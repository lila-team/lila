import asyncio
import json
import os
import tempfile
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from browser_use import Agent, Browser, BrowserConfig, Controller
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel

from lila.config import Config
from lila.const import MAX_LOGS_DISPLAY
from lila.utils import (
    dump_browser_state,
    render_template_to_file,
    replace_vars_in_content,
    run_command,
)


@dataclass
class ReportLog:
    log: str
    screenshot_b64: str


class FailedStepError(RuntimeError):
    pass


@dataclass
class StepResult:
    success: bool
    msg: str


class AgentResult(BaseModel):
    success: bool
    msg: str


controller = Controller()


@dataclass
class TestCase:
    name: str
    steps: List[Dict[str, str]]

    tags: List[str] = field(default_factory=list)
    raw_content: str = ""

    status: str = "pending"
    steps_results: List[StepResult] = field(default_factory=list)

    logs: List[Dict[str, Any]] = field(default_factory=list)

    duration: float = 0.0

    _should_stop: bool = False

    @classmethod
    def from_yaml(cls, name: str, content: str):
        data = yaml.safe_load(content)
        logger.debug(f"Loaded test case: {name}")
        return cls(
            name=name,
            steps=[step for step in data["steps"]],
            tags=data.get("tags", []),
            status="pending",
            raw_content=content,
        )

    def dump_report(self, output_dir: str, report: Dict[int, List[ReportLog]]) -> None:
        # Writes a markdown report for the test case
        template_data = {
            "name": self.name,
            "steps": self.steps,
            "report": report,
            "now": datetime.utcnow(),
        }
        template_path = Path(__file__).parent / "assets" / "report.html"
        output = Path(output_dir) / f"{self.name}.html"
        render_template_to_file(template_path, output, template_data)
        logger.debug(f"Report for {self.name} saved at {output}")

    def _update_state(self, run_id: str, server_url: str):
        ret = requests.get(
            f"{server_url}/api/v1/remote/runs/{run_id}/status",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
            },
        )
        ret.raise_for_status()
        data = ret.json()
        logger.debug(f"Succesfully fetched state for run {run_id}")
        self.steps_results = [
            StepResult(success=step["success"], msg=step["msg"])
            for step in data["step_results"]
        ]
        self.logs = [
            {"level": log["level"], "msg": log["msg"]} for log in data["logs"]
        ][-MAX_LOGS_DISPLAY:]
        self.status = data.get("conclusion", data["run_status"])
        logger.debug("Updated update queue for new state to render")

    async def handle_verification(
        self,
        thread_id: str,
        verification: str,
        step_type: str,
        step_content: str,
        context: BrowserContext,
        run_id: str,
        server_url: str,
        report_logs: List[ReportLog],
    ) -> None:
        state = await context.get_state()
        dumped_state = dump_browser_state(state)
        ret = requests.post(
            f"{server_url}/api/v1/remote/runs/{run_id}/threads/{thread_id}/verifications",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
            },
            json={
                "verification": replace_vars_in_content(verification),
                "state": dumped_state,
                "step_type": step_type,
                "step_content": replace_vars_in_content(step_content),
            },
        )
        ret.raise_for_status()
        logger.debug(
            f"Successfully sent browser state for verification for thread {thread_id} run {run_id}"
        )

        result = ret.json()["result"]
        if result["status"] == "failure":
            logger.debug(f"Verification failed '{verification}': {result['log']}")
            report_logs.append(
                ReportLog(
                    log=f"Verification failed: {result['log']}",
                    screenshot_b64=await context.take_screenshot(),
                )
            )
            raise FailedStepError(f"Failed to execute step: {result['log']}")

        if result["status"] == "complete":
            logger.debug(f"Verification passed for '{verification}': {result['log']}")
            logger.info(f"Verification passed: {result['log']}")
            report_logs.append(
                ReportLog(
                    log=f"Verification passed: {result['log']}",
                    screenshot_b64=await context.take_screenshot(),
                )
            )
        else:
            raise RuntimeError(f"Unknown verification status: {result['status']}")

    async def _handle_agent_step(
        self,
        run_id: str,
        thread_id: str,
        step_type: str,
        step_content: str,
        context: BrowserContext,
        report_logs: List[ReportLog],
        llm: BaseChatModel,
    ) -> None:
        controller = Controller(output_model=AgentResult)

        agent = Agent(
            browser_context=context,
            task=f"Handle the following step: {step_type} {step_content}",
            llm=llm,
            controller=controller,
        )
        history = await agent.run()
        result = history.final_result()

        if result:
            parsed: AgentResult = AgentResult.model_validate_json(result)

            if parsed.success:
                logger.debug(f"Agent step completed successfully: {parsed.msg}")
                report_logs.append(
                    ReportLog(
                        log=f"Agent step completed successfully: {parsed.msg}",
                        screenshot_b64=await context.take_screenshot(),
                    )
                )
            else:
                logger.debug(f"Agent step failed: {parsed.msg}")
                report_logs.append(
                    ReportLog(
                        log=f"Agent step failed: {parsed.msg}",
                        screenshot_b64=await context.take_screenshot(),
                    )
                )
                raise FailedStepError(f"Failed to execute step: {parsed.msg}")
        else:
            raise RuntimeError("Agent step returned no result")

    async def handle_step(
        self,
        step_type: str,
        step_content: str,
        verifications: List[str],
        context: BrowserContext,
        run_id: str,
        report_logs: List[ReportLog],
        llm: BaseChatModel,
    ) -> None:
        thread_id = str(uuid.uuid4())
        logger.debug(f"Initializing thread {thread_id} to handle step")

        page = await context.get_current_page()
        if step_type == "goto":
            await page.goto(replace_vars_in_content(step_content))
            await context._wait_for_page_and_frames_load()
            logger.info("Navigation completed successfully")
            report_logs.append(
                ReportLog(
                    log="Navigation completed successfully",
                    screenshot_b64=await context.take_screenshot(),
                )
            )
        elif step_type == "wait":
            seconds = int(replace_vars_in_content(step_content))
            logger.info(f"Waiting for {step_content} seconds")
            await page.wait_for_timeout(seconds * 1000)
            logger.info("Wait completed successfully")
            report_logs.append(
                ReportLog(
                    log="Wait completed successfully",
                    screenshot_b64=await context.take_screenshot(),
                )
            )
        elif step_type == "exec":
            cmds = replace_vars_in_content(step_content).split("\n")
            for cmd in cmds:
                if cmd:
                    logger.info(f"Executing command: {cmd}")
                    stdout, stderr, rc = await run_command(cmd)
                    if rc != 0:
                        logger.error(f"Failed to execute command: {stderr}")
                        logger.error(f"Command output: {stdout.decode()}")
                        logger.error(f"Command error: {stderr.decode()}")
                        logger.error(f"Command return code: {rc}")
                        report_logs.append(
                            ReportLog(
                                log=f"Failed to execute command {cmd}: {stderr.decode()}",
                                screenshot_b64=await context.take_screenshot(),
                            )
                        )
                        raise FailedStepError(f"Failed to execute command: code {rc}")
            logger.info("Commands executed successfully")
            report_logs.append(
                ReportLog(
                    log="Commands executed successfully",
                    screenshot_b64=await context.take_screenshot(),
                )
            )
        else:
            await self._handle_agent_step(
                run_id,
                thread_id,
                step_type,
                step_content,
                context,
                report_logs,
                llm,
            )

        if verifications:
            logger.debug(f"Running step verifications: {len(verifications)}")
        #     for verification in verifications:
        #         logger.debug(f"Running verification: {verification}")
        #         await self.handle_verification(
        #             thread_id,
        #             verification,
        #             step_type,
        #             step_content,
        #             context,
        #             run_id,
        #             report_logs,
        #         )
        #         logger.debug(f"Verification passed for {verification}")
        else:
            logger.info("No verifications for this step")

    @staticmethod
    def initialize_browser_context(
        config: Config, browser_state: Optional[str] = None
    ) -> BrowserContext:
        storage_state = None
        if browser_state:
            logger.info(f"Loading browser state from {browser_state}")
            with open(browser_state, "r") as f:
                storage_state = json.load(f)

        browser_config = BrowserConfig(headless=config.browser.headless)
        browser = Browser(config=browser_config)

        cookies_file = None
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            if storage_state and "cookies" in storage_state:
                json.dump(storage_state["cookies"], f)
                cookies_file = f.name

            # Ref https://docs.browser-use.com/customize/browser-settings#context-configuration
            context_config = BrowserContextConfig(
                wait_for_network_idle_page_load_time=3,
                browser_window_size={
                    "width": config.browser.width,
                    "height": config.browser.height,
                },
                # pass -1 for highlighting all elements in the page
                viewport_expansion=0,
                cookies_file=cookies_file,
            )

            return BrowserContext(browser=browser, config=context_config)

    @staticmethod
    async def teardown(context: BrowserContext, name: str, config: Config) -> None:
        path = Path(config.runtime.output_dir) / f"{name}.json"
        os.makedirs(path.parent, exist_ok=True)
        await context.session.context.storage_state(path=path)
        logger.debug(f"Browser state saved for {name} at {path}")
        await context.close()
        await context.browser.close()

    async def run(
        self,
        run_id: str,
        config: Config,
        browser_state: Optional[str],
        name: str,
        llm: BaseChatModel,
    ) -> bool:
        with logger.contextualize(test_name=name):
            context = self.initialize_browser_context(config, browser_state)
            logger.debug(f"Browser context initialized for {run_id}: {context}")
            success = True
            report: Dict[int, List[ReportLog]] = {}
            for idx, step in enumerate(self.steps):
                report[idx] = []
                step_type, step_content = [(k, v) for k, v in step.items()][0]
                with logger.contextualize(step=f"{step_type} {step_content}"):
                    verifications: str | List[str] = step.get("verify", [])
                    if isinstance(verifications, str):
                        verifications_list: List[str] = [verifications]
                    else:
                        verifications_list = verifications

                    logger.debug(
                        f"Running step {idx}: {step_type} {step_content} [{len(verifications_list)} verifications]"
                    )
                    try:
                        await self.handle_step(
                            step_type,
                            step_content,
                            verifications_list,
                            context,
                            run_id,
                            report[idx],
                            llm,
                        )
                        logger.success("Step completed successfully")
                    except FailedStepError as e:
                        success = False
                        logger.error(f"Failed to execute step: {str(e)}")
                        break
                    except Exception as e:
                        logger.exception(f"Unexpected error: {e}")
                        await self.teardown(context, name, config)
                        raise

            await self.teardown(context, name, config)
            self.dump_report(config.runtime.output_dir, report)
            report_path = Path(config.runtime.output_dir) / f"{name}.html"
            logger.info(f"Report saved at {report_path}")
            return success


def collect_test_cases(
    test_files: List[str], tags: List[str], exclude_tags: List[str]
) -> List[TestCase]:
    testcases = []
    for path in test_files:
        with open(path, "r") as f:
            content = f.read()
            # Remove extension for filename
            name = os.path.splitext(path)[0]
            test = TestCase.from_yaml(name, content)

        if tags:
            if not set(tags).intersection(test.tags):
                continue

            if exclude_tags:
                if set(exclude_tags).intersection(test.tags):
                    continue

        testcases.append(test)

    return testcases


class TestRunner:
    def __init__(self, testcases: List[TestCase]):
        self.testcases = testcases

    def run_tests(
        self,
        config: Config,
        browser_state: Optional[str],
        llm: BaseChatModel,
    ) -> bool:
        future_to_test = {}

        with ThreadPoolExecutor(
            max_workers=config.runtime.concurrent_workers
        ) as executor:
            for idx, testcase in enumerate(self.testcases):
                with logger.contextualize(test_name=testcase.name):
                    run_id = str(uuid.uuid4())

                    # For debuggin purposes
                    def run_wrapper(*args, **kwargs):
                        try:
                            return asyncio.run(testcase.run(*args, **kwargs))
                        except Exception:
                            print(traceback.format_exc())
                            raise

                    # Submit all tests
                    key = executor.submit(
                        run_wrapper,
                        run_id,
                        config,
                        browser_state,
                        name=testcase.name,
                        llm=llm,
                    )
                    future_to_test[key] = testcase

            for future in as_completed(future_to_test.keys()):
                result = future.result()
                testcase = future_to_test[future]

                testcase.status = "success" if result else "failure"

        # Show summary and failed test details
        total = len(self.testcases)
        passed = sum(1 for t in self.testcases if t.status == "success")
        failed = sum(1 for t in self.testcases if t.status == "failure")

        if failed:
            logger.error(
                f"Test Report - Passed: {passed}, Failed: {failed}, Total: {total}"
            )
            return False
        else:
            logger.success(
                f"Test Report - Passed: {passed}, Failed: {failed}, Total: {total}"
            )
            return True
