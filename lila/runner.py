import asyncio
import time
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
from lila.models import Step, TestSteps, TestCaseDef


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
class TestCaseRun:
    id: str
    path: str
    test_def: TestCaseDef

    status: str = "pending"
    steps_results: List[StepResult] = field(default_factory=list)

    logs: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0

    # def dump_report(self, output_dir: str, report: Dict[int, List[ReportLog]]) -> None:
    #     # Writes a markdown report for the test case
    #     template_data = {
    #         "name": self.name,
    #         "steps": self.steps,
    #         "report": report,
    #         "now": datetime.utcnow(),
    #     }
    #     template_path = Path(__file__).parent / "assets" / "report.html"
    #     output = Path(output_dir) / f"{self.name}.html"
    #     render_template_to_file(template_path, output, template_data)
    #     logger.debug(f"Report for {self.name} saved at {output}")


class TestRunner:
    def __init__(self, testcases: Dict[str, TestCaseDef]):
        self.testcases = testcases

    def run_tests(
        self,
        config: Config,
        browser_state: Optional[str],
        llm: BaseChatModel,
    ) -> bool:
        future_to_test = {}
        results = []

        with ThreadPoolExecutor(
            max_workers=config.runtime.concurrent_workers
        ) as executor:
            for path, testcase in self.testcases.items():

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
                    path,
                    config,
                    browser_state,
                    llm=llm,
                )
                future_to_test[key] = testcase

            for future in as_completed(future_to_test.keys()):
                testcase_run = future.result()
                results.append(testcase_run)
