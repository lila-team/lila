import asyncio
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, List, Optional

import requests
import yaml
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from browser_use.controller.views import (
    ClickElementAction,
    InputTextAction,
)
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lila.config import Config
from lila.const import INCLUDE_ATTRIBUTES, MAX_LOGS_DISPLAY
from lila.utils import get_vars, get_vars_from_env


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

    def _update_state(self, run_id: str, server_url: str, update_queue: Queue):
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
        update_queue.put(True)
        logger.debug("Updated update queue for new state to render")

    async def handle_step(
        self,
        idx: int,
        step_type: str,
        step_content: str,
        context: BrowserContext,
        update_queue: Queue,
        run_id: str,
        server_url: str,
        dry_run: bool = False,
    ) -> None:
        logger.info(f"Handling step {idx}: {step_type} {step_content}")
        page = await context.get_current_page()
        if step_type == "goto":
            await page.goto(step_content)
            await page.wait_for_load_state()
            logger.debug(f"Successfully navigated to {step_content}")
            if not dry_run:
                send_step_log(run_id, idx, "info", "Navigated to page", server_url)
                self._update_state(run_id, server_url, update_queue)
        else:
            done = False
            prev_report = None
            while not done:
                state = await context.get_state()
                dumped_state = {
                    "dom": state.element_tree.clickable_elements_to_string(
                        include_attributes=INCLUDE_ATTRIBUTES
                    ),
                    "screenshot_b64": state.screenshot,
                }
                if prev_report:
                    dumped_state["prev_report"] = prev_report

                ret = requests.post(
                    f"{server_url}/api/v1/remote/runs/{run_id}/steps/{idx}/states",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
                    },
                    json=dumped_state,
                )
                ret.raise_for_status()
                logger.debug(f"Successfully sent browser state for run {run_id}")

                ret = requests.get(
                    f"{server_url}/api/v1/remote/runs/{run_id}/steps/{idx}/completion",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
                    },
                )
                ret.raise_for_status()
                logger.debug(f"Successfully fetched completion for run {run_id}")

                data = ret.json()["completion"]
                logger.info(f"New completion data: {data}")
                if "action" in data:
                    logger.debug("Completion requires action")
                    if data["action"] == "click":
                        fn = controller.registry.registry.actions["click_element"]
                        await fn.function(
                            ClickElementAction(index=int(data["element"])), context
                        )
                    elif data["action"] == "input":
                        fn = controller.registry.registry.actions["input_text"]
                        await fn.function(
                            InputTextAction(
                                index=int(data["element"]), text=data["text"]
                            ),
                            context,
                        )

                    await page.wait_for_load_state()
                    prev_report = {"status": "success"}
                    logger.info(
                        f"Handled action {data['action']} required by completion for step {idx}"
                    )

                if data["status"] == "complete":
                    done = True
                    logger.info("Completion reports step is done")
                elif data["status"] == "failure":
                    if not dry_run:
                        send_step_log(run_id, idx, "info", data["log"], server_url)
                    logger.info("Completion reports step failed")
                    raise RuntimeError(f"Failed to execute step: {data['log']}")

                if not done:
                    # Submit a clean screenshot
                    await context.remove_highlights()
                    screenshot = await context.take_screenshot()
                    if not dry_run:
                        send_step_log(run_id, idx, "info", data["log"], server_url)
                        send_step_screenshot(run_id, idx, screenshot, server_url)
                        self._update_state(run_id, server_url, update_queue)

        if not dry_run:
            report_step_result(run_id, idx, "success", "Step completed", server_url)

        # Submit a clean screenshot
        await context.remove_highlights()
        screenshot = await context.take_screenshot()
        if not dry_run:
            send_step_screenshot(run_id, idx, screenshot, server_url)
            self._update_state(run_id, server_url, update_queue)

        logger.info(f"Step {idx} completed successfully")

    def start(self, server_url: str, batch_id: str) -> str:
        required_secrets = get_vars(self.raw_content)
        given_secrets = get_vars_from_env(required_secrets, fail_if_missing=False)

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
                "secrets": given_secrets,
                "batch_id": batch_id,
            },
        )
        ret.raise_for_status()
        if ret.status_code != 201:
            raise RuntimeError(f"Failed to start run: {ret.json()}")

        data = ret.json()
        logger.info(f"Successfully started run {data['run_id']} for test {self.name}")
        return data["run_id"]

    async def run(
        self,
        run_id: str,
        update_queue: Queue,
        config: Config,
        browser_state: str,
        dry_run: bool = False,
    ) -> None:
        server_url = config.runtime.server_url
        report_test_status(run_id, {"status": "running"}, server_url)
        update_queue.put(True)

        config = BrowserConfig(headless=config.browser.headless)
        browser = Browser(config=config)
        async with await browser.new_context() as context:
            for idx, step in enumerate(self.steps):
                step_type, step_content = [(k, v) for k, v in step.items()][0]
                await self.handle_step(
                    idx,
                    step_type,
                    step_content,
                    context,
                    update_queue,
                    run_id,
                    server_url,
                    dry_run=dry_run,
                )

            report_test_status(
                run_id, {"status": "finished", "conclusion": "success"}, server_url
            )
            if not dry_run:
                self._update_state(run_id, server_url, update_queue)


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
        self.console = Console()
        self._update_queue: Queue = Queue()

    def create_table(self) -> Table:
        table = Table(show_header=True, header_style="bold", show_lines=True)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Test Name", justify="left", width=30, overflow="fold")
        table.add_column("Progress", justify="left", width=10)
        table.add_column("Recent Logs", justify="left", width=60)

        status_colors = {
            "pending": "white",
            "enqueued": "white",
            "running": "deep_sky_blue1",
            "success": "green",
            "failure": "red",
            "skipped": "yellow",
            "cancelled": "yellow",
            "error": "red",
            "remote_timeout": "red",
        }

        for testcase in self.testcases:
            status_style = f"{status_colors[testcase.status]}"

            # Calculate step progress
            total_steps = len(testcase.steps)
            completed_steps = len(testcase.steps_results)
            progress = f"{completed_steps}/{total_steps}"

            # Format log entries
            log_text = Text()
            pending_logs = MAX_LOGS_DISPLAY - len(testcase.logs)
            if pending_logs:
                log_text.append(
                    " \n"
                    * (
                        pending_logs
                        if pending_logs < MAX_LOGS_DISPLAY
                        else (MAX_LOGS_DISPLAY - 1)
                    )
                )

            for i, log in enumerate(testcase.logs):
                level_colors = {
                    "info": "cyan",
                    "warn": "yellow",
                    "error": "red",
                }
                log_text.append(
                    f'[{log["level"].upper()}] {log["msg"]}',
                    style=level_colors.get(log["level"], "white"),
                )
                if i < len(testcase.logs) - 1:
                    log_text.append("\n")

            table.add_row(
                f"[{status_style}]{testcase.status}[/]",
                testcase.name,
                progress,
                log_text,
            )

        return table

    def run_tests(
        self, config: Config, browser_state: Optional[str], batch_id: str, dry_run: bool
    ) -> bool:
        def update_display(live: Live):
            """Background thread to update the display"""
            while True:
                update = self._update_queue.get()
                if update is None:  # Sentinel value to stop the thread
                    break
                live.update(self.create_table())

        failed_tests = []
        future_to_test = {}

        if dry_run:
            with ThreadPoolExecutor(
                max_workers=config.runtime.concurrent_workers
            ) as executor:
                for idx, testcase in enumerate(self.testcases):
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
                        self._update_queue,
                        config,
                        browser_state,
                        dry_run=True,
                    )
                    future_to_test[key] = testcase

                ret = True
                for future in as_completed(future_to_test.keys()):
                    test = future_to_test[future]
                    print(f"Test {test.name} finished with status {test.status}")

                    if test.status == "failure":
                        ret = False

            return ret

        # Create live display
        with Live(self.create_table(), refresh_per_second=5) as live:
            # Start display update thread
            display_thread = threading.Thread(target=update_display, args=(live,))
            display_thread.start()

            # Run tests in parallel
            with ThreadPoolExecutor(
                max_workers=config.runtime.concurrent_workers
            ) as executor:
                for idx, testcase in enumerate(self.testcases):
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
                        run_wrapper, run_id, self._update_queue, config, browser_state
                    )
                    future_to_test[key] = testcase

                for future in as_completed(future_to_test.keys()):
                    test = future_to_test[future]

                    if test.status == "failure":
                        failed_tests.append(test)

                        if config.runtime.fail_fast:
                            # This will stop the running futures
                            # Thread share memory, so we can update its state
                            # and the thread will read it.
                            for testcase in future_to_test.values():
                                testcase._should_stop = True

            # Stop the display update thread
            self._update_queue.put(None)
            display_thread.join()

        # Show summary and failed test details
        total = len(self.testcases)
        passed = sum(1 for t in self.testcases if t.status == "success")
        failed = len(failed_tests)

        if failed_tests:
            self.console.print("\n=========== Failures ===========\n", style="bold red")
            for test in failed_tests:
                # Create detailed step report
                steps_text = Text()
                for i, step in enumerate(test.steps):
                    status_color = "white"
                    if len(test.steps_results) > i:
                        status_color = (
                            "green" if test.steps_results[i].success else "red"
                        )
                    action, content = [(k, v) for k, v in step.items()][0]
                    steps_text.append(f"\n{action}: {content} ", style=status_color)

                steps_text.append(
                    f"\n\nError: {test.steps_results[-1].msg}", style="bold red"
                )

                panel = Panel(
                    steps_text,
                    title=f"[red]{test.name}[/] (Duration: {test.duration:.2f}s)",
                    border_style="red",
                )
                self.console.print(panel)

        if not failed:
            self.console.print(
                f"\n=========== [bold green]{passed}[/] tests passed ===========\n",
                style="bold green",
            )
        else:
            success_rate = passed / total * 100
            self.console.print(
                f"========== [bold red]{failed} failed[/], [bold green]{passed} passed[/], [bold purple]{success_rate:.2f}% success rate[/] =========="
            )

        return failed == 0
