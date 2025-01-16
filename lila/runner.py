import base64
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, TypedDict

import requests
import yaml
from PIL import Image
from playwright.sync_api import Page, Playwright, sync_playwright
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lila.config import Config
from lila.const import HEIGHT, MAX_LOGS_DISPLAY, WIDTH
from lila.utils import Pointer, get_vars, get_vars_from_env


def initialize_page(
    p: Playwright, config: Config, browser_state: Optional[str]
) -> Page:
    if config.browser.type == "firefox":
        browser = p.firefox.launch(
            headless=config.browser.headless,
        )
    elif config.browser.type == "webkit":
        browser = p.webkit.launch(
            headless=config.browser.headless,
        )
    elif config.browser.type == "chromium":
        browser = p.chromium.launch(
            headless=config.browser.headless,
        )
    else:
        raise RuntimeError(f"Unsupported browser type: {config.browser.type}")

    if browser_state:
        context = browser.new_context(storage_state=browser_state)
    else:
        context = browser.new_context()

    page = context.new_page()
    page.set_viewport_size(
        {"width": config.browser.width, "height": config.browser.height}
    )
    return page


@dataclass
class ActionResult:
    # base64 encoded screenshot
    screenshot: str

    success: bool
    msg: Optional[str] = None


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: Dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


def scale_coordinates(x: int, y: int):
    """Scale coordinates to a target maximum resolution."""
    ratio = WIDTH / HEIGHT
    target_dimension = None
    for dimension in MAX_SCALING_TARGETS.values():
        # allow some error in the aspect ratio - not ratios are exactly 16:9
        if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
            if dimension["width"] < WIDTH:
                target_dimension = dimension
            break
    if target_dimension is None:
        return x, y
    # should be less than 1
    x_scaling_factor = target_dimension["width"] / WIDTH
    y_scaling_factor = target_dimension["height"] / HEIGHT
    return round(x * x_scaling_factor), round(y * y_scaling_factor)


def base64_screenshot(page: Page, pointer: Pointer, wait: float = 0) -> str:
    if wait:
        time.sleep(wait)

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = f"{tmpdirname}/screenshot.png"
        page.screenshot(path=path)

        img = Image.open(path)

        # get path of current file, go back to root
        # and appens assets/pointer.png
        image_path = Path(__file__).parent / "assets" / "pointer.png"
        pointer_img = Image.open(image_path)
        img.paste(pointer_img, (pointer.x, pointer.y), pointer_img)
        img.save(path)

        x, y = scale_coordinates(WIDTH, HEIGHT)
        img = Image.open(path)
        img = img.resize((x, y))
        img.save(path)

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def handle_action(page: Page, payload: Dict, pointer: Pointer) -> ActionResult:
    action = payload["action"]
    wait: float = 1
    if action == "goto":
        page.goto(payload["url"])
        wait = 3
    elif action == "screenshot":
        wait = 0
    elif action == "left_click":
        page.mouse.down()
        page.mouse.up()
    elif action == "right_click":
        page.mouse.down(button="right")
        page.mouse.up(button="right")
    elif action == "double_click":
        page.mouse.down()
        page.mouse.up()
        page.mouse.down()
        page.mouse.up()
    elif action == "mouse_move":
        page.mouse.move(payload["x"], payload["y"])
        pointer.x = payload["x"]
        pointer.y = payload["y"]
    elif action == "left_click_drag":
        page.mouse.down(button="left")
        page.mouse.move(payload["x"], payload["y"])
        page.mouse.up(button="left")
        pointer.x = payload["x"]
        pointer.y = payload["y"]
    elif action == "key":
        text_mapping = {
            "Return": "Enter",
        }
        page.keyboard.press(text_mapping.get(payload["text"], payload["text"]))
        wait = 0.5
    elif action == "type":
        page.keyboard.type(payload["text"], delay=100)
        wait = 0.5
    else:
        raise RuntimeError(f"Unknown action: {action}")

    return ActionResult(
        success=True, screenshot=base64_screenshot(page, pointer, wait=wait)
    )


def report_action_result(
    run_id: str, action_id: str, result: ActionResult, server_url: str
) -> None:
    ret = requests.patch(
        f"{server_url}/api/v1/remote/runs/{run_id}/actions/{action_id}",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
        },
        json={
            "result": "success" if result.success else "error",
            "screenshot": result.screenshot,
        },
    )
    ret.raise_for_status()


@dataclass
class StepResult:
    success: bool
    msg: str


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
        return cls(
            name=name,
            steps=[step for step in data["steps"]],
            tags=data.get("tags", []),
            status="pending",
            raw_content=content,
        )

    def loop_until_done(
        self, page: Page, run_id: str, update_queue: Queue, server_url: str
    ) -> None:
        done = False
        pointer = Pointer(WIDTH // 2, HEIGHT // 2)
        while not done and not self._should_stop:
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
            self.steps_results = [
                StepResult(success=step["success"], msg=step["msg"])
                for step in data["step_results"]
            ]
            self.logs = [
                {"level": log["level"], "msg": log["msg"]} for log in data["logs"]
            ][-MAX_LOGS_DISPLAY:]

            update_queue.put(True)

            if data["run_status"] in [
                "finished",
                "cancelled",
                "error",
                "remote_timeout",
            ]:
                done = True
                self.status = data.get("conclusion") or data["run_status"]
                self.duration = data["duration"]
            elif data["run_status"] == "processing":
                self.status = "running"
                time.sleep(0.5)
            elif data["run_status"] == "requested_action":
                payload = data["payload"]
                result = handle_action(page, payload, pointer)
                report_action_result(run_id, data["id"], result, server_url)
            else:
                raise RuntimeError(f"Unknown status: {data['status']}")

        if self._should_stop:
            self.status = "cancelled"
            update_queue.put(True)
            return

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

        return ret.json()["run_id"]

    def run(
        self, run_id, update_queue: Queue, config: Config, browser_state: str
    ) -> None:
        with sync_playwright() as p:
            page = initialize_page(p, config, browser_state)
            self.status = "pending"
            update_queue.put(True)
            self.loop_until_done(page, run_id, update_queue, config.runtime.server_url)
            update_queue.put(True)

            # Save state of the page
            output_file = f"{config.runtime.output_dir}/{self.name}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            page.context.storage_state(path=output_file)
            page.close()


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
        self, config: Config, browser_state: Optional[str], batch_id: str
    ) -> bool:
        def update_display(live: Live):
            """Background thread to update the display"""
            while True:
                update = self._update_queue.get()
                if update is None:  # Sentinel value to stop the thread
                    break
                live.update(self.create_table())

        failed_tests = []

        # Create live display
        with Live(self.create_table(), refresh_per_second=5) as live:
            # Start display update thread
            display_thread = threading.Thread(target=update_display, args=(live,))
            display_thread.start()

            future_to_test = {}

            # Run tests in parallel
            with ThreadPoolExecutor(
                max_workers=config.runtime.concurrent_workers
            ) as executor:
                for idx, testcase in enumerate(self.testcases):
                    run_id = testcase.start(config.runtime.server_url, batch_id)

                    # For debuggin purposes
                    def run_wrapper(*args, **kwargs):
                        try:
                            return testcase.run(*args, **kwargs)
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

        # slowest_test = sorted(self.testcases, key=lambda t: t.duration, reverse=True)[0]
        # fastest_test = sorted(self.testcases, key=lambda t: t.duration)[0]
        # avg = sum(t.duration for t in self.testcases) / total

        # self.console.print(
        #     f"Slowest test: {slowest_test.name} [white]({slowest_test.duration:.2f}s)[/]"
        # )
        # self.console.print(
        #     f"Fastest test: {fastest_test.name} [white]({fastest_test.duration:.2f}s)[/]"
        # )
        # self.console.print(f"Average duration: [white]{avg:.2f}s[/]")
