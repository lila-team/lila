import json
import os
import re
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from browser_use import Agent, Browser, Controller
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from lila.config import Config
from pydantic import BaseModel
from browser_use import Agent, Browser, BrowserConfig, Controller
from langchain_core.language_models.chat_models import BaseChatModel
from lila.utils import (
    dump_browser_state,
    render_template_to_file,
    replace_vars_in_content,
    run_command,
)

import yaml
from dacite import from_dict
from loguru import logger


TIMEOUT = timedelta(hours=1)
MINUTE_IN_SECS = 60


class Verification(BaseModel):
    passed: bool
    reason: str


class ActionResult(BaseModel):
    success: bool
    reason: str


@dataclass
class StepResult:
    action: ActionResult
    screenshot_b64: str
    verifications: List[Verification]


@dataclass
class Step:
    verify: str | List[str]

    def validate(self):
        pass

    def __str__(self):
        # Build a string representation of the step name and the content
        # Get the name of the instance attribute first
        step_name = self.__class__.__name__.lower()
        return f"{step_name} {self.__dict__[step_name]} (self.verify)"

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Step":
        raise NotImplementedError("from_content not implemented")

    @classmethod
    def get_type(cls) -> str:
        return cls.__name__.lower()

    def get_value(self) -> str:
        return self.__dict__[self.__class__.__name__.lower()]

    def get_secrets(self) -> List[str]:
        value = self.get_value()
        return get_vars(value)

    @staticmethod
    async def run_verification(context: BrowserContext, verification: str, llm: BaseChatModel) -> Verification:
        controller = Controller(output_model=Verification)
        agent = Agent(
            browser_context=context,
            task=f"Verify the following verification: {verification}. You are not allowed to perform any action.",
            llm=llm,
            controller=controller,
        )
        history = await agent.run()
        result = history.final_result()

        if result:
            try:
                return Verification.model_validate_json(result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse verification result: {e}")
                raise ValueError(
                    "Invalid JSON format in verification result"
                ) from e
        else:
            logger.error("No result returned from verification agent")
            raise ValueError("No result returned from verification agent")


    async def run_verifications(self, context: BrowserContext, llm: BaseChatModel) -> List[Verification]:
        if not self.verify:
            return []

        verifications = [self.verify] if isinstance(self.verify, str) else self.verify

        ret = []
        for verification in verifications:
            logger.debug(f"Verifying {verification}")
            result = await self.run_verification(context, verification, llm)
            if result.passed:
                logger.info(f"Verification passed: {result.reason}")
            else:
                logger.error(f"Verification failed: {result.reason}")

            ret.append(result)

        return ret

    async def _handle_action(self, context: BrowserContext, llm: BaseChatModel) -> ActionResult:
        controller = Controller(output_model=ActionResult)
        agent = Agent(
            browser_context=context,
            task=f"Do the following: {self.get_type()} {self.get_value()}",
            llm=llm,
            controller=controller,
        )
        history = await agent.run()
        result = history.final_result()

        if result:
            try:
                return ActionResult.model_validate_json(result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse action result: {e}")
                raise ValueError(
                    "Invalid JSON format in action result"
                ) from e
        else:
            logger.error("No result returned from action agent")
            raise ValueError("No result returned from action agent")


    async def handle(self, context: BrowserContext, config: Config, llm: BaseChatModel) -> StepResult:
        action_result = await self._handle_action(context, llm)
        if not action_result.success:
            logger.error(f"Action failed: {action_result.reason}")
            return StepResult(
                action=action_result,
                screenshot_b64=await context.take_screenshot(),
                verifications=[],
            )
        else:
            logger.info(f"Action completed successfully: {action_result.reason}")
            verifications = await self.run_verifications(context, llm)
            return StepResult(
                action=action_result,
                screenshot_b64=await context.take_screenshot(),
                verifications=verifications,
            )

    def dump(self):
        key = self.__class__.__name__.lower()
        value = self.__dict__[key]
        return {key: value, "verify": self.verify}

    def data_with_secrets(self, secrets_mapping: Optional[Dict[str, str]]) -> str:
        # Replace secrets in the step content
        secrets_mapping = secrets_mapping or {}
        data = self.__dict__[self.__class__.__name__.lower()]
        new_data = replace_vars(data, secrets_mapping)
        logger.debug(f"Secrets mapping: {secrets_mapping.keys()}")
        logger.debug(f"Replacing secrets in {data}")
        return new_data


@dataclass
class Pick(Step):
    pick: str

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Pick":
        return cls(pick=content, verify=verify or [])


@dataclass
class Submit(Step):
    submit: str

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Submit":
        return cls(submit=content, verify=verify or [])

@dataclass
class Input(Step):
    input: str

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Input":
        return cls(input=content, verify=verify or [])

@dataclass
class Click(Step):
    click: str

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Click":
        return cls(click=content, verify=verify or [])


@dataclass
class Wait(Step):
    wait: str | int

    @classmethod
    def from_content(
        cls, content: int | str, verify: Optional[List[str]] = None
    ) -> "Wait":
        return cls(wait=content, verify=verify or [])

    def validate(self):
        try:
            wait = int(self.wait)
        except ValueError:
            raise ValueError("Wait value must be an integer")

        if wait < 0:
            raise ValueError("Wait time must be a positive integer")

        return super().validate()

    async def _handle_action(self, context: BrowserContext, llm: BaseChatModel) -> ActionResult:
        # Wait for the specified time
        wait_time = int(self.wait)
        page = await context.get_current_page()
        await page.wait_for_timeout(wait_time * 1000)  # Convert seconds to milliseconds
        return ActionResult(success=True, reason="Wait completed successfully")


@dataclass
class Exec(Step):
    exec: str

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Exec":
        return cls(exec=content, verify=verify or [])


    async def _handle_action(self, context: BrowserContext, llm: BaseChatModel) -> ActionResult:
        cmds = replace_vars_in_content(self.exec).split("\n")
        for cmd in cmds:
            if cmd:
                logger.info(f"Executing command: {cmd}")
                stdout, stderr, rc = await run_command(cmd)
                if rc != 0:
                    logger.error(f"Failed to execute command: {stderr}")
                    logger.error(f"Command output: {stdout.decode()}")
                    logger.error(f"Command error: {stderr.decode()}")
                    logger.error(f"Command return code: {rc}")
                    return ActionResult(
                        success=False,
                        reason=f"Command failed with return code {rc}",
                    )

        return ActionResult(
            success=True, reason="Command executed successfully"
        )


@dataclass
class Goto(Step):
    goto: str

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Goto":
        return cls(goto=content, verify=verify or [])

    @staticmethod
    def _is_valid_url(string):
        # Define a regex pattern for matching a valid URL
        url_pattern = re.compile(
            r"^(https?://)"  # Match the scheme (http or https)
            r"([a-zA-Z0-9.-]+)"  # Match the domain
            r"(\.[a-zA-Z]{2,})"  # Match the top-level domain
            r"(:\d+)?"  # Optionally match a port number
            r"(/[^\s]*)?$",  # Optionally match the path, query, and fragment
            re.IGNORECASE,
        )
        # Check if the string matches the URL pattern exactly
        return bool(url_pattern.match(string))

    def validate(self):
        # Validate the URL but first replace potential env vars
        goto_with_fake_env_vars = fake_vars(self.goto, lambda _: "foobar")

        # goto value must be a valid URL
        # it shouldn't contain any other string
        if not self._is_valid_url(goto_with_fake_env_vars):
            raise ValueError(
                "Invalid URL. Please provide a valid URL with http or https scheme and no additional content"
            )

        return super().validate()

    async def _handle_action(self, context: BrowserContext, llm: BaseChatModel) -> ActionResult:
        page = await context.get_current_page()
        await page.goto(replace_vars_in_content(self.goto))
        await context._wait_for_page_and_frames_load()
        return ActionResult(success=True, reason="Navigation completed successfully")


TestSteps = List[Goto | Click | Input | Pick | Submit | Wait | Exec]


@dataclass
class TestCaseDef:
    steps: TestSteps
    tags: List[str] = field(default_factory=list)

    async def run(
        self,
        path: str,
        config: Config,
        browser_state: Optional[str],
        llm: BaseChatModel,
    ) -> "TestCaseRun":
        id = str(uuid.uuid4())
        step_results = []
        ckpt = time.time()
        with logger.contextualize(test_name=path):
            context = self.initialize_browser_context(config, browser_state)
            logger.debug(f"Browser context initialized for {id}: {context}")
            success = True

            for idx, step in enumerate(self.steps):
                step_type= step.get_type()
                step_content = step.get_value()
                with logger.contextualize(step=f"{step_type} {step_content}"):
                    logger.debug(
                        f"Starting step {idx}"
                    )
                    step_result = await step.handle(
                        context,
                        config,
                        llm,
                    )
                    step_results.append(step_result)

                    if not step_result.action.success:
                        logger.error(f"Step failed: {step_result.action.reason}")
                        await self.teardown(context, id, config)
                        return TestCaseRun(
                            id=id,
                            path=path,
                            test_def=self,
                            status="failed",
                            steps_results=step_results,
                            duration=time.time() - ckpt,
                        )

                    if step_result.verifications and any(not v.passed for v in step_result.verifications):
                        logger.error(f"Step verifications failed")
                        await self.teardown(context, id, config)
                        return TestCaseRun(
                            id=id,
                            path=path,
                            test_def=self,
                            status="failed",
                            steps_results=step_results,
                            duration=time.time() - ckpt,
                        )

                    logger.info(f"Step completed successfully")

            logger.info("All steps completed successfully")

            await self.teardown(context, id, config)
            return TestCaseRun(
                id=id,
                path=path,
                test_def=self,
                status="success",
                steps_results=step_results,
                duration=time.time() - ckpt,
            )


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
    async def teardown(context: BrowserContext, run_id: str, config: Config) -> None:
        path = Path(config.runtime.output_dir) / f"{run_id}.json"
        os.makedirs(path.parent, exist_ok=True)
        await context.session.context.storage_state(path=path)
        logger.debug(f"Browser state saved for {run_id} at {path}")
        await context.close()
        await context.browser.close()
            # self.dump_report(config.runtime.output_dir, report)
            # report_path = Path(config.runtime.output_dir) / f"{name}.html"
            # logger.info(f"Report saved at {report_path}")
            # return success


@dataclass
class TestCaseRun:
    id: str
    path: str
    test_def: TestCaseDef

    status: str = "pending"
    steps_results: List[StepResult] = field(default_factory=list)

    duration: float = 0.0


