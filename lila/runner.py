import asyncio
import json
import os
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.controller.service import Controller
from browser_use.controller.views import (
    ClickElementAction,
    InputTextAction,
    ScrollAction,
)
from loguru import logger

from lila.config import Config
from lila.const import MAX_ATTEMPTS, MAX_LOGS_DISPLAY
from lila.utils import (
    PressKeyWithFocus,
    dump_browser_state,
    get_completion,
    press_key,
    pretty_step_formatter,
    render_template_to_file,
    send_state,
)


@dataclass
class ReportLog:
    log: str
    screenshot_b64: str


class FailedStepError(RuntimeError):
    pass


def report_test_status(run_id: str, payload: Dict, server_url: str) -> None:
    ret = requests.patch(
        f"{server_url}/api/v1/remote/runs/{run_id}",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
        },
        json=payload,
    )
    ret.raise_for_status()
    logger.debug(f"Successfully reported test status for run {run_id}")


def report_step_result(
    run_id: str, idx: int, result: str, msg: str, server_url: str
) -> None:
    ret = requests.put(
        f"{server_url}/api/v1/remote/runs/{run_id}/steps/{idx}/result",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
        },
        json={"result": result, "msg": msg},
    )
    ret.raise_for_status()
    logger.debug(f"Successfully reported step result for run {run_id}")


def send_step_screenshot(
    run_id: str, idx: int, screenshot_b64: str, server_url: str
) -> None:
    ret = requests.post(
        f"{server_url}/api/v1/remote/runs/{run_id}/steps/{idx}/screenshots",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
        },
        json={"screenshot_b64": screenshot_b64},
    )
    ret.raise_for_status()
    logger.debug(f"Successfully sent screenshot for run {run_id}")


def send_step_log(run_id: str, idx: int, level: str, msg: str, server_url: str) -> None:
    ret = requests.post(
        f"{server_url}/api/v1/remote/runs/{run_id}/steps/{idx}/logs",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
        },
        json={"level": level, "msg": msg},
    )
    ret.raise_for_status()
    logger.debug(f"Successfully sent log for run {run_id}")


@dataclass
class StepResult:
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

    async def handle_verifications(
        self,
        idx: int,
        step_content: str,
        context: BrowserContext,
        run_id: str,
        server_url: str,
        report_logs: List[ReportLog],
        dry_run: bool = False,
    ) -> None:
        state = await context.get_state()
        dumped_state = dump_browser_state(state)
        ret = requests.post(
            f"{server_url}/api/v1/remote/runs/{run_id}/steps/{idx}/verifications",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
            },
            json=dumped_state,
        )
        ret.raise_for_status()
        logger.debug(
            f"Successfully sent browser state for verification for step {idx } run {run_id}"
        )

        data = ret.json()

        if not data["verifications"]:
            logger.debug(f"No verifications found for step {idx}")
            return

        for verification in data["verifications"]:
            if verification["status"] == "failure":
                logger.debug(
                    f"Verification failed for step {pretty_step_formatter(self.steps[idx])}: {verification['log']}"
                )
                report_logs.append(
                    ReportLog(
                        log=f"Verification failed: {verification['log']}",
                        screenshot_b64=await context.take_screenshot(),
                    )
                )
                raise FailedStepError(f"Failed to execute step: {verification['log']}")

            logger.debug(f"Verification passed for step {idx}: {verification['log']}")
            report_logs.append(
                ReportLog(
                    log=f"Verification passed: {verification['log']}",
                    screenshot_b64=await context.take_screenshot(),
                )
            )

    async def _handle_complex_step(
        self,
        idx: int,
        context: BrowserContext,
        run_id: str,
        server_url: str,
        report_logs: List[ReportLog],
        dry_run: bool = False,
    ) -> None:
        done = False
        prev_interaction = None
        scrolling = False
        attempts = 0
        while not done:
            action_result = None
            attempts += 1
            if attempts > MAX_ATTEMPTS:
                raise FailedStepError("Step took too long to complete")

            state = await context.get_state()
            send_state(idx, run_id, server_url, state, prev_interaction)
            logger.debug(f"Sent state for step {idx}")

            data = get_completion(idx, run_id, server_url)["completion"]

            report_logs.append(
                ReportLog(
                    log=data["log"],
                    screenshot_b64=await context.take_screenshot(),
                )
            )

            logger.debug(f"Got completion data for step {idx}")

            logger.info(data["log"])
            if "action" in data:
                logger.debug(f"Completion requires action: {data['action']}")
                if data["action"] == "click":
                    fn = controller.registry.registry.actions["click_element"]
                    action_result = await fn.function(
                        ClickElementAction(index=int(data["element"])), context
                    )
                    # Hack. There is a bug in browser use that reports a click
                    # as a failure when the context is destroyed due to a navigation
                    # event. This is a workaround to handle that.
                    if (
                        action_result.error
                        and "Execution context was destroyed" in action_result.error
                    ):
                        action_result.error = None

                    if action_result.error:
                        logger.error(f"Click failed: {action_result.error}")
                    else:
                        logger.info("Click performed successfully")

                    prev_interaction = {
                        "status": "failure" if action_result.error else "success",
                        "msg": action_result.error
                        or "Successfully clicked requested element",
                    }
                elif data["action"] == "input":
                    fn = controller.registry.registry.actions["input_text"]
                    action_result = await fn.function(
                        InputTextAction(index=int(data["element"]), text=data["text"]),
                        context,
                    )

                    if action_result.error:
                        logger.error(f"Input failed: {action_result.error}")
                    else:
                        logger.info("Input performed successfully")

                    prev_interaction = {
                        "status": "failure" if action_result.error else "success",
                        "msg": action_result.error
                        or "Successfully inputted text into requested element",
                    }
                elif data["action"] == "press-enter":
                    action_result = await press_key(
                        PressKeyWithFocus(index=int(data["element"]), key="Enter"),
                        context,
                    )
                    logger.debug("Pressed Enter key")
                    if action_result.error:
                        logger.error(f"Press Enter failed: {action_result.error}")
                    else:
                        logger.info("Press Enter performed successfully")

                    prev_interaction = {
                        "status": "failure" if action_result.error else "success",
                        "msg": action_result.error or "Successfully pressed Enter key",
                    }
                elif data["action"] == "native-dropdown-open":
                    fn = controller.registry.registry.actions["get_dropdown_options"]
                    action_result = await fn.function(
                        index=int(data["element"]),
                        browser=context,
                    )

                    if action_result.error:
                        logger.error(f"Failed to open dropdown: {action_result.error}")
                    else:
                        logger.info("Opened dropdown and got options")
                    prev_interaction = {
                        "status": "failure" if action_result.error else "success",
                        "msg": action_result.error
                        or f"Successfully opened dropdown and got options:\n{action_result.extracted_content}",
                    }
                    logger.debug("Opened dropdown and got options")
                elif data["action"] == "native-dropdown-select":
                    fn = controller.registry.registry.actions["select_dropdown_option"]
                    action_result = await fn.function(
                        index=int(data["element"]),
                        text=data["text"],
                        browser=context,
                    )

                    if action_result.error:
                        logger.error(
                            f"Failed to select option from dropdown: {action_result.error}"
                        )
                    else:
                        logger.info("Selected option from dropdown")

                    prev_interaction = {
                        "status": "failure" if action_result.error else "success",
                        "msg": action_result.error
                        or "Successfully selected option from dropdown",
                    }
                    logger.debug("Opened dropdown and got options")
                else:
                    raise RuntimeError(f"Unknown action {data['action']}")

                await context._wait_for_page_and_frames_load()
                report_logs.append(
                    ReportLog(
                        log=action_result.error or "Action performed successfully",
                        screenshot_b64=await context.take_screenshot(),
                    )
                )

            if data["status"] == "complete" and (
                action_result is None or not action_result.error
            ):
                done = True
                logger.debug("Completion reports step completed.")
                report_logs.append(
                    ReportLog(
                        log="Step completed successfully",
                        screenshot_b64=await context.take_screenshot(),
                    )
                )
            if data["status"] == "wip":
                logger.debug("Completion reports step still in progress.")
            elif data["status"] == "not-found":
                logger.debug("Completion reports element not found. Running scrolling.")
                if scrolling or state.pixels_above == 0:
                    if state.pixels_below == 0:
                        logger.debug("No more elements to uncover. Failing step.")
                        raise FailedStepError(f"Failed to execute step: {data['log']}")

                    fn = controller.registry.registry.actions["scroll_down"]
                    await fn.function(ScrollAction(), context)
                    logger.debug("Scrolled one page down")
                    prev_interaction = {
                        "status": "success",
                        "msg": "Scrolled down one page to uncover more elements.",
                    }
                    report_logs.append(
                        ReportLog(
                            log="Scrolled down one page to uncover more elements.",
                            screenshot_b64=await context.take_screenshot(),
                        )
                    )
                else:
                    fn = controller.registry.registry.actions["scroll_up"]
                    await fn.function(ScrollAction(state.pixels_above), context)
                    logger.debug("Moved all the way up")
                    prev_interaction = {
                        "status": "success",
                        "msg": "Scrolled to the top of the page to uncover more elements.",
                    }
                    report_logs.append(
                        ReportLog(
                            log="Scrolled to the top of the page to uncover more elements.",
                            screenshot_b64=await context.take_screenshot(),
                        )
                    )

                scrolling = True
                logger.debug("Activated scrolling mode")

                await context._wait_for_page_and_frames_load()
            elif data["status"] == "failure":
                logger.debug(f"Completion reports step failed: {data['log']}")
                report_logs.append(
                    ReportLog(
                        log="Failed to execute step: {data['log']}",
                        screenshot_b64=await context.take_screenshot(),
                    )
                )
                raise FailedStepError(f"Failed to execute step: {data['log']}")

    async def handle_step(
        self,
        idx: int,
        step_type: str,
        step_content: str,
        verifications: List[str],
        context: BrowserContext,
        run_id: str,
        server_url: str,
        report_logs: List[ReportLog],
        dry_run: bool = False,
    ) -> None:
        page = await context.get_current_page()
        if step_type == "goto":
            await page.goto(step_content)
            await context._wait_for_page_and_frames_load()
            logger.info("Navigation completed successfully")
            report_logs.append(
                ReportLog(
                    log="Navigation completed successfully",
                    screenshot_b64=await context.take_screenshot(),
                )
            )
        elif step_type == "wait":
            seconds = int(step_content)
            logger.info(f"Waiting for {step_content} seconds")
            await page.wait_for_timeout(seconds * 1000)
            logger.info("Wait completed successfully")
            report_logs.append(
                ReportLog(
                    log="Wait completed successfully",
                    screenshot_b64=await context.take_screenshot(),
                )
            )
        else:
            await self._handle_complex_step(
                idx, context, run_id, server_url, report_logs, dry_run
            )

        if verifications:
            logger.info("Running step verifications")
            await self.handle_verifications(
                idx, step_content, context, run_id, server_url, report_logs, dry_run
            )
            logger.debug("Verifications completed successfully")
        else:
            logger.info("No verifications for this step")

    def start(self, server_url: str, batch_id: str) -> str:
        ret = requests.post(
            f"{server_url}/api/v1/remote/runs",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
            },
            json={
                "name": self.name,
                "content": self.raw_content,
                "batch_id": batch_id,
            },
        )
        ret.raise_for_status()
        if ret.status_code != 201:
            raise RuntimeError(f"Failed to start run: {ret.json()}")

        data = ret.json()
        logger.info(f"Successfully started running test {self.name} [{data['run_id']}]")
        return data["run_id"]

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
        dry_run: bool = False,
    ) -> bool:
        with logger.contextualize(test_name=name):
            server_url = config.runtime.server_url
            report_test_status(
                run_id, {"status": "running" if not dry_run else "dry-run"}, server_url
            )
            logger.debug(f"Test {run_id} started running")

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
                            idx,
                            step_type,
                            step_content,
                            verifications_list,
                            context,
                            run_id,
                            server_url,
                            report[idx],
                            dry_run=dry_run,
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

            if not dry_run:
                report_test_status(
                    run_id,
                    {
                        "status": "finished",
                        "conclusion": "success" if success else "failure",
                    },
                    server_url,
                )

            await self.teardown(context, name, config)
            self.dump_report(config.runtime.output_dir, report)
            report_path = Path(config.runtime.output_dir) / f"{name}.html"
            logger.info(f"Report saved at {report_path}")
            return success


def collect_test_cases(test_files: List[str], tags: List[str], exclude_tags: List[str]):
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
        self, config: Config, browser_state: Optional[str], batch_id: str, dry_run: bool
    ) -> bool:
        future_to_test = {}

        with ThreadPoolExecutor(
            max_workers=config.runtime.concurrent_workers
        ) as executor:
            for idx, testcase in enumerate(self.testcases):
                with logger.contextualize(test_name=testcase.name):
                    run_id = testcase.start(config.runtime.server_url, batch_id)

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
                        dry_run=True,
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
