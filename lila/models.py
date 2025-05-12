import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, List, Optional

from browser_use import Agent, Browser, BrowserConfig, Controller
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel

from lila.config import Config
from lila.utils import (
    fake_vars,
    get_vars,
    replace_vars_in_content,
    run_command,
)

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

    @staticmethod
    def task_instructions() -> str:
        """Generic instructions for the LLM on how to perform this type of action."""
        return "Perform the action as instructed. Focus only on completing this one action."

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
    async def run_verification(
        context: BrowserContext,
        verification: str,
        llm_factory: Callable[[], BaseChatModel],
    ) -> Verification:
        controller = Controller(output_model=Verification)
        agent = Agent(
            enable_memory=False,
            browser_context=context,
            task=f"<INSTRUCTIONS>\nPerform a VISUAL VERIFICATION ONLY of: {verification}.\nDO NOT perform any actions.\n</INSTRUCTIONS>\n\n<NOTES>\n- Keep reason under 100 chars, be concise.\n- Only report on what you can visually confirm is present or absent on the screen.\n</NOTES>",
            llm=llm_factory(),
            controller=controller,
        )
        history = await agent.run()
        result = history.final_result()

        if result:
            try:
                return Verification.model_validate_json(result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse verification result: {e}")
                raise ValueError("Invalid JSON format in verification result") from e
        else:
            logger.error(f"No result returned from verification agent: {history}")
            raise ValueError(f"No result returned from verification agent {history}")

    # Verifications are now handled directly in the handle method

    async def _handle_action(
        self,
        context: BrowserContext,
        llm_factory: Callable[[], BaseChatModel],
        prev_steps: Optional[List[str]] = None,
        next_steps: Optional[List[str]] = None,
    ) -> ActionResult:
        controller = Controller(output_model=ActionResult)

        agent = Agent(
            enable_memory=False,
            browser_context=context,
            task=f"""Your job is to perform ONLY this action: {self.get_type()} {self.get_value()}.

<INSTRUCTIONS>
{self.task_instructions()}
</INSTRUCTIONS>

<NOTES>
- You need to respond if you were able to perform the action or not, with a reason.
- Not being able to complete the action can be due to:
    - The action is not possible to perform in the current context.
    - The elements were not found
    - Elements are not interactable
    - or any other reason preventing you from performing the action.
- If you encounter a popup, banner, captcha, etc. that blocks you from attempting the action, you are allowed to perform other actions to unblock the main action.
- In all other cases, you are not allowed to perform any other action than the one specified.
- Keep the output reason under 100 chars, be concise, regardless if the actions was successful or not.
</NOTES>
""",
            llm=llm_factory(),
            controller=controller,
        )
        history = await agent.run()
        result = history.final_result()

        if result:
            try:
                return ActionResult.model_validate_json(result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse action result: {e}")
                raise ValueError("Invalid JSON format in action result") from e
        else:
            logger.error(f"No result returned from action agent: {history}")
            raise ValueError(f"No result returned from action agent {history}")

    async def handle(
        self,
        context: BrowserContext,
        config: Config,
        llm_factory: Callable[[], BaseChatModel],
        prev_steps: Optional[List[str]] = None,
        next_steps: Optional[List[str]] = None,
    ) -> StepResult:
        try:
            action_result = await self._handle_action(
                context, llm_factory, prev_steps, next_steps
            )
        except ValueError as e:
            logger.error(f"Error handling action: {e}")
            return StepResult(
                action=ActionResult(success=False, reason=str(e)),
                screenshot_b64=await context.take_screenshot(),
                verifications=[],
            )

        if not action_result.success:
            logger.error(f"Action failed: {action_result.reason}")
            return StepResult(
                action=action_result,
                screenshot_b64=await context.take_screenshot(),
                verifications=[],
            )
        else:
            logger.info(f"Completed successfully: {action_result.reason}")
            # Use simpler verification flag
            verifications = []
            for verification in (
                [self.verify] if isinstance(self.verify, str) else self.verify
            ):
                with logger.contextualize(verify=True, verification_text=verification):
                    logger.debug(f"Verifying: {verification}")
                    try:
                        result = await self.run_verification(
                            context, verification, llm_factory
                        )
                        verifications.append(result)
                        if result.passed:
                            logger.info(f"Verification passed: {result.reason}")
                        else:
                            logger.error(f"Verification failed: {result.reason}")
                    except ValueError as e:
                        verifications.append(Verification(passed=False, reason=str(e)))

            return StepResult(
                action=action_result,
                screenshot_b64=await context.take_screenshot(),
                verifications=verifications,
            )

    def dump(self):
        key = self.__class__.__name__.lower()
        value = self.__dict__[key]
        return {key: value, "verify": self.verify}


@dataclass
class Pick(Step):
    pick: str

    @staticmethod
    def task_instructions() -> str:
        """Instructions for the LLM on how to perform a pick action."""
        return "Select from a dropdown menu or list. Find and pick the specified option. Do not submit after picking."

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Pick":
        return cls(pick=content, verify=verify or [])


@dataclass
class Submit(Step):
    submit: str

    @staticmethod
    def task_instructions() -> str:
        """Instructions for the LLM on how to perform a submit action."""
        return "Find and submit a form by clicking the specified submit button or equivalent submission element."

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Submit":
        return cls(submit=content, verify=verify or [])


@dataclass
class Input(Step):
    input: str

    @staticmethod
    def task_instructions() -> str:
        """Instructions for the LLM on how to perform an input action."""
        return "Input the specified text into the appropriate input field. Do not submit after entering the text, just complete the input action."

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Input":
        return cls(input=content, verify=verify or [])


@dataclass
class Click(Step):
    click: str

    @staticmethod
    def task_instructions() -> str:
        """Instructions for the LLM on how to perform a click action."""
        return "Find and click on the specified element. This could be a button, link, or other clickable element."

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Click":
        return cls(click=content, verify=verify or [])


@dataclass
class Wait(Step):
    wait: str | int

    @staticmethod
    def task_instructions() -> str:
        """Instructions for the LLM on how to perform a wait action."""
        return "Wait for the specified number of seconds without performing any actions. This allows time for page elements to load or animations to complete."

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

    async def _handle_action(
        self, context: BrowserContext, *args, **kwargs
    ) -> ActionResult:
        # Wait for the specified time
        wait_time = int(self.wait)
        await asyncio.sleep(wait_time)
        return ActionResult(success=True, reason="Wait completed successfully")


@dataclass
class Exec(Step):
    exec: str

    @staticmethod
    def task_instructions() -> str:
        """Instructions for the LLM on how to perform an exec action."""
        return "Execute a shell command on the server. This is a system operation that runs in the background and does not involve browser interaction."

    @classmethod
    def from_content(cls, content: str, verify: Optional[List[str]] = None) -> "Exec":
        return cls(exec=content, verify=verify or [])

    async def _handle_action(
        self, context: BrowserContext, *args, **kwargs
    ) -> ActionResult:
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

        return ActionResult(success=True, reason="Command executed successfully")


@dataclass
class Goto(Step):
    goto: str

    @staticmethod
    def task_instructions() -> str:
        """Instructions for the LLM on how to perform a goto action."""
        return "Navigate to the specified URL. This is a browser navigation action that loads a new page."

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

    async def _handle_action(
        self, context: BrowserContext, *args, **kwargs
    ) -> ActionResult:
        await context.navigate_to(self.goto)
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
        llm_factory: Callable[[], BaseChatModel],
    ) -> "TestCaseRun":
        id = str(uuid.uuid4())
        step_results = []
        ckpt = time.time()
        # Use shorter path for test name context
        test_id = Path(path).stem
        with logger.contextualize(test_name=test_id):
            context = self.initialize_browser_context(config, browser_state)
            logger.debug(f"Browser context initialized for {id}: {context}")

            for idx, step in enumerate(self.steps):
                step_type = step.get_type()
                step_content = step.get_value()
                # Use a simplified step description for logging
                step_desc = f"{step_type}:{step_content}"
                with logger.contextualize(step=step_desc):
                    logger.debug(f"Step {idx+1}")
                    # Add action context flag for logging
                    with logger.contextualize(action=True):
                        # Get all previous and next steps for context
                        prev_steps = []
                        next_steps = []

                        # Collect all previous steps
                        for i in range(0, idx):
                            prev_step_obj = self.steps[i]
                            prev_steps.append(
                                f"{prev_step_obj.get_type()} {prev_step_obj.get_value()}"
                            )

                        # Collect all next steps
                        for i in range(idx + 1, len(self.steps)):
                            next_step_obj = self.steps[i]
                            next_steps.append(
                                f"{next_step_obj.get_type()} {next_step_obj.get_value()}"
                            )

                        step_result = await step.handle(
                            context,
                            config,
                            llm_factory,
                            prev_steps=prev_steps,
                            next_steps=next_steps,
                        )
                    step_results.append(step_result)

                    if not step_result.action.success:
                        logger.error(f"Failed: {step_result.action.reason}")
                        await self.teardown(context, id, config)
                        return TestCaseRun(
                            id=id,
                            path=path,
                            test_def=self,
                            status="failed",
                            steps_results=step_results,
                            duration=time.time() - ckpt,
                        )

                    if step_result.verifications and any(
                        not v.passed for v in step_result.verifications
                    ):
                        logger.error("Step verifications failed")
                        await self.teardown(context, id, config)
                        return TestCaseRun(
                            id=id,
                            path=path,
                            test_def=self,
                            status="failed",
                            steps_results=step_results,
                            duration=time.time() - ckpt,
                        )

                    logger.debug("Step completed successfully")

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
                window_width=config.browser.width,
                window_height=config.browser.height,
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
