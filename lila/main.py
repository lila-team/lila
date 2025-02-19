import json
import os
import pathlib
import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import click
import requests
import yaml
from loguru import logger

from lila.config import Config
from lila.const import API_KEY_URL
from lila.runner import TestRunner, collect_test_cases
from lila.utils import parse_tags, setup_logging


def validate_content(content: str, server_url: str) -> None:
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Provided content is not a valid YAML: {e}")

    ret = requests.post(
        f"{server_url}/api/v1/testcase-validations",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
        },
        json={"content": content},
    )
    raise_for_status(ret)
    if ret.status_code == 200:
        data = ret.json()
        if data["valid"]:
            return

        raise ValueError(
            f"Provided content is not a valid Lila test case: {data['message']}"
        )


def find_parsing_errors(test_paths: List[str], server_url: str) -> Dict[str, str]:
    invalid_tests = {}
    for path in test_paths:
        with open(path, "r") as f:
            content = f.read()
            try:
                validate_content(content, server_url)
            except ValueError as e:
                invalid_tests[path] = str(e)

    return invalid_tests


def raise_for_status(response: requests.Response):
    try:
        response.raise_for_status()
    except requests.RequestException as e:
        try:
            data = response.json()
            raise RuntimeError(f"Error: {data}") from e
        except json.JSONDecodeError:
            raise RuntimeError(f"Error: {response.text}") from e


@click.group()
def cli():
    """Lila CLI tool."""
    pass


def collect(path) -> List[str]:
    """
    Find all YAML files in a given path. If path is a file, return it if it's a YAML file.
    If path is a directory, recursively search for all YAML files within it.

    Args:
        path (str): Path to file or directory

    Returns:
        list: List of paths to YAML files found
    """
    yaml_extensions = (".yaml", ".yml")
    result = []

    # Convert path to Path object for easier handling
    path_obj = pathlib.Path(path)

    # If path is a file, check if it's a YAML file
    if path_obj.is_file():
        if path_obj.suffix.lower() in yaml_extensions:
            return [str(path_obj)]
        return []

    # If path is a directory, walk through it recursively
    if path_obj.is_dir():
        for root, _, files in os.walk(path):
            for file in files:
                file_path = pathlib.Path(root) / file
                if file_path.suffix.lower() in yaml_extensions:
                    result.append(str(file_path))

    return sorted(result)  # Sort for consistent output


def _get_config(config_file: Optional[str]) -> Config:
    if not config_file:
        if os.path.exists("lila.toml"):
            config_file = "lila.toml"
        else:
            return Config.default()
    elif not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    return Config.from_toml_file(config_file)


@cli.command()
@click.argument("path", type=str, required=True)
@click.option(
    "--tags",
    type=str,
    help="Comma-separated list of tags",
    required=False,
)
@click.option("--exclude-tags", type=str, help="Exclude tests by tags")
@click.option("--config", type=str, help="Path to the Lila config file", required=False)
@click.option(
    "--browser-state", type=str, help="Path to the browser state file", required=False
)
@click.option(
    "--output-dir", type=str, help="Override config output directory", required=False
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode",
    default=False,
)
@click.option(
    "--headless",
    is_flag=True,
    help="Run tests in headless mode",
    default=False,
)
def run(
    path: str,
    tags: str,
    exclude_tags: str,
    config: Optional[str],
    browser_state: Optional[str],
    output_dir: Optional[str],
    debug: bool = False,
    headless: bool = False,
):
    """Run a Lila test suite."""
    setup_logging(debug=debug)
    # This were previously run flags,
    # will add the later once webapp
    # is more stable.
    dry_run = True
    batch_id = None

    if "LILA_API_KEY" not in os.environ:
        logger.error(
            f"Please set the LILA_API_KEY environment variable. You can find it in the Lila app: {API_KEY_URL}"
        )
        return

    try:
        config_obj = _get_config(config)
    except FileNotFoundError as e:
        logger.error(str(e))

    if output_dir:
        config_obj.runtime.output_dir = output_dir

    config_obj.browser.headless = headless

    tag_list = []
    if tags:
        try:
            tag_list = parse_tags(tags)
        except ValueError as e:
            logger.error(str(e))
            return

    exclude_tag_list = []
    if exclude_tags:
        try:
            exclude_tag_list = parse_tags(exclude_tags)
        except ValueError as e:
            logger.error(str(e))
            return

    # If intersection of tags and exclude_tags is not empty, raise an error
    if set(tag_list) & set(exclude_tag_list):
        logger.error(
            f"Tags and exclude-tags cannot have common elements: {tag_list} and {exclude_tag_list}"
        )
        return

    if browser_state and not os.path.exists(browser_state):
        logger.error(f"Browser state file not found: {browser_state}")
        return

    test_files = collect(path)
    if not test_files:
        logger.error(f"No YAML files found in the provided path: {path}")
        return

    invalid_files = find_parsing_errors(test_files, config_obj.runtime.server_url)
    if invalid_files:
        logger.error("Parsing errors found")
        for path, error in invalid_files.items():
            logger.error(f"File {path}: {error}")
        return

    testcases = collect_test_cases(test_files, tag_list, exclude_tag_list)

    if not testcases:
        logger.error(
            "No test cases found with the provided path and the provided params"
        )
        return

    if not batch_id:
        batch_id = str(uuid.uuid4())

    runner = TestRunner(testcases)
    success = runner.run_tests(config_obj, browser_state, batch_id, dry_run)
    if not success:
        sys.exit(1)


@cli.command()
def init():
    """Initialize a Lila testing template."""
    if os.path.exists("lila.toml"):
        logger.error(
            "Config file lila.toml already exists, it seems like the application is already initialized."
        )
        return

    # Create a lila.toml file
    config_path = Path(__file__).parent / "assets" / "lila.toml"
    shutil.copy(config_path, "lila.toml")
    logger.info("Config file created: lila.toml")

    # Create a lila directory
    os.makedirs("lila-output", exist_ok=True)
    logger.info("Output directory created for artifacts: lila-output/")

    # Create an example test case
    example_path = Path(__file__).parent.parent / "tests" / "google-maps.yaml"
    shutil.copy(example_path, "demo.yaml")
    logger.info("Example test case created: demo.yaml")
    logger.success("All set! Run your first test: lila run demo.yaml")


@click.option("--config", type=str, help="Path to the Lila config file", required=False)
@cli.command()
def check(
    config: Optional[str],
) -> None:
    """Check if Lila is properly set up and ready to run tests."""
    setup_logging(debug=False)
    if "LILA_API_KEY" not in os.environ:
        logger.error(
            f"Please set the LILA_API_KEY environment variable. You can find it in the Lila app: {API_KEY_URL}"
        )
        return

    config_obj = _get_config(config)
    server_url = config_obj.runtime.server_url
    ret = requests.get(
        f"{server_url}/api/v1/readiness",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['LILA_API_KEY']}",
        },
    )
    if ret.status_code == 401:
        logger.error("Invalid LILA_API_KEY. Please set the correct API key.")
        sys.exit(1)
        return

    if ret.status_code != 200:
        logger.error(f"Lila is not ready to run tests [code: {ret.status_code}]")
        sys.exit(1)
        return

    logger.success("Lila is properly set up and ready to run tests.")
