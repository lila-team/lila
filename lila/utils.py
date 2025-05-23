import asyncio
import copy
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import (
    BrowserState,
)
from browser_use.dom.views import (
    DOMElementNode,
)
from jinja2 import Template
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from lila.const import INCLUDE_ATTRIBUTES


def parse_tags(value: str) -> List[str]:
    ret = []
    if value:
        tags = value.split(",")
        for tag in tags:
            tag = tag.strip()
            if not re.match(r"^[a-zA-Z0-9-_]+$", tag.strip()):
                raise ValueError(
                    f"Invalid tag '{tag}'. Tags must be alphanumeric (with optional hyphens or underscores)."
                )
            ret.append(tag)

    return ret


def get_vars(content: str) -> List[str]:
    # Searches for all ${VAR_NAME} and returns the list
    # of strings VAR_NAME
    regex = re.compile(r"\${(.*?)}")
    ret = regex.findall(content)
    return ret


def get_missing_vars(vars_list: List[str]) -> List[str]:
    # Returns a list of variables in vars_list that are not present
    # in the environment variables
    ret = []
    for var in vars_list:
        if os.environ.get(var) is None:
            ret.append(var)

    return ret


def replace_vars_in_content(content: str) -> str:
    # Replaces all ${VAR_NAME} with the os env var value
    ret = copy.deepcopy(content)
    for var in get_vars(content):
        value = os.environ.get(var)
        if value is None:
            raise RuntimeError(f"Environment variable {var} not found")

        ret = ret.replace(f"${{{var}}}", value)

    return ret


def dump_browser_state(state: BrowserState) -> Dict:
    return {
        "dom": state.element_tree.clickable_elements_to_string(
            include_attributes=INCLUDE_ATTRIBUTES
        ),
        "url": state.url,
        "title": state.title,
        "screenshot_b64": state.screenshot,
        "pixels_above": state.pixels_above,
        "pixels_below": state.pixels_below,
    }


def generate_completion(
    run_id: str,
    thread_id: str,
    server_url: str,
    state: BrowserState,
    step_type: str,
    step_content: str,
    prev_interaction: Optional[Dict] = None,
) -> Dict:
    dumped_state = dump_browser_state(state)
    data = {
        "step_type": step_type,
        "step_content": step_content,
        "state": dumped_state,
    }

    if prev_interaction:
        data["prev_interaction"] = prev_interaction

    ret = requests.post(
        f"{server_url}/api/v1/remote/runs/{run_id}/threads/{thread_id}/completion",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
        },
        json=data,
    )
    ret.raise_for_status()
    return ret.json()


### Custom implementation of key press over an element
### Based on the original input controller action
### https://github.com/browser-use/browser-use/blob/87f4d74e063f3c6236a53238c7178564158f3966/browser_use/controller/service.py#L227
@dataclass
class PressKeyWithFocus:
    index: int
    key: str


async def _press_key_element_node(
    context: BrowserContext, element_node: DOMElementNode, key: str
):
    try:
        # Highlight before typing
        if element_node.highlight_index is not None:
            await context._update_state(focus_element=element_node.highlight_index)

        page = await context.get_current_page()
        element_handle = await context.get_locate_element(element_node)

        if element_handle is None:
            raise Exception(f"Element: {repr(element_node)} not found")

        await element_handle.scroll_into_view_if_needed(timeout=2500)
        await element_handle.focus()
        await element_handle.press(key)
        await page.wait_for_load_state()

    except Exception as e:
        raise Exception(
            f"Failed to press key into element: {repr(element_node)}. Error: {str(e)}"
        )


async def press_key(params: PressKeyWithFocus, context: BrowserContext):
    session = await context.get_session()
    state = session.cached_state

    if params.index not in state.selector_map:
        raise Exception(
            f"Element index {params.index} does not exist - retry or use alternative actions"
        )

    element_node = state.selector_map[params.index]
    await _press_key_element_node(context, element_node, params.key)
    msg = f"⌨️  Pressed {params.key} into index {params.index}"
    logger.debug(msg)
    logger.debug(f"Element xpath: {element_node.xpath}")
    return ActionResult(extracted_content=msg, include_in_memory=True)


###


def setup_logging(debug: bool):
    # Cleanup existing config
    # logger.remove()

    def formatter(record):
        ret = ""
        if debug:
            ret += (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            )

        if "test_name" in record["extra"]:
            MAX_STEP_LENGTH = 20
            test_name = record["extra"]["test_name"]

            if "step" in record["extra"]:
                step = (
                    record["extra"]["step"]
                    .replace("\n", " ")
                    .replace("{", "")
                    .replace("}", "")
                )
                step_text = f"{step[:MAX_STEP_LENGTH]}{'...' if len(step) > MAX_STEP_LENGTH else ''}"

                # Context formatting with action/verification flags
                action_tag = " | ▶" if "action" in record["extra"] else ""

                # Add verification flag and text if present
                verification_tag = ""
                if "verify" in record["extra"]:
                    verification_tag = " | ✓"
                    if "verification_text" in record["extra"]:
                        v_text = record["extra"]["verification_text"]
                        # Truncate verification text if too long
                        MAX_VERIFY_LENGTH = 30
                        v_text = v_text.replace("\n", " ")
                        v_text = f"{v_text[:MAX_VERIFY_LENGTH]}{'...' if len(v_text) > MAX_VERIFY_LENGTH else ''}"
                        verification_tag = f" | ✓ {v_text}"

                ret += f"<cyan>[{test_name} | {step_text}{action_tag}{verification_tag}]</cyan> "
            else:
                ret += f"<cyan>[{test_name}]</cyan> "

        ret += f'<level>{record["message"]}</level>\n'
        return ret

    logger.add(
        sink=sys.stderr,
        level="DEBUG" if debug else "INFO",
        format=formatter,
    )

    logger.add(
        sink=".lila.log",
        level="DEBUG",
        format="{time} {level} {message}",
        rotation="1MB",
    )


def pretty_step_formatter(step: Dict) -> str:
    action, content = list(step.items())[0]
    verifications = step.get("verify", [])
    if isinstance(verifications, str):
        verifications = [verifications]

    return f"{action}: {content} | {len(verifications)} verifications"


def render_template_to_file(template_path, output_file, data):
    try:
        # Read the template file
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Create template object
        template = Template(template_content)

        template.globals["pretty_step"] = pretty_step_formatter

        # Render the template
        rendered_html = template.render(**data)

        # Write the rendered HTML to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rendered_html)

        return True

    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return False


async def run_command(command):
    # Create the subprocess
    process = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Read stdout and stderr
    stdout, stderr = await process.communicate()

    # Get the return code
    return_code = process.returncode
    return stdout, stderr, return_code


def get_langchain_chat_model(model: str, provider: str) -> BaseChatModel:
    # TODO -> add additional kwargs
    # to configure the model
    model = init_chat_model(
        model=model,
        model_provider=provider,
    )
    logger.debug(f"Using model: {model}")
    return model


def fake_vars(content: str, fake_env_f: Callable[[str], str]) -> str:
    # Searches for all ${VAR_NAME} and replaces them with the value of the environment variable VAR_NAME
    regex = re.compile(r"\${(.*?)}")
    for match in regex.findall(content):
        logger.debug(f"Matched: {match}")
        value = fake_env_f(match)
        if value is None:
            logger.warning(f"Mapping variable {match} not found")
            continue
        content = content.replace(f"${{{match}}}", value)

    return content
