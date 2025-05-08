import base64
import datetime
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import yaml
from dacite import from_dict
from jinja2 import Template
from loguru import logger

from lila.config import Config
from lila.models import TestCaseDef
from lila.runner import TestRunner
from lila.utils import (
    get_langchain_chat_model,
    get_missing_vars,
    get_vars,
    parse_tags,
    setup_logging,
)


@dataclass
class TestCollection:
    valid: Dict[str, TestCaseDef]
    invalid: Dict[str, str]


def generate_html_report(test_results, output_dir: str) -> Tuple[str, float]:
    """
    Generate an HTML report for test results and save it to the output directory.

    Args:
        test_results: List of test results
        output_dir: Directory to save the report

    Returns:
        Tuple of (report_path, total_duration)
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate summary metrics
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result.status == "success")
    failed_tests = total_tests - passed_tests
    total_duration = sum(result.duration for result in test_results)

    # Try to load the logo
    logo_base64 = ""
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Load the template
    template_path = Path(__file__).parent / "assets" / "test_results.html"
    with open(template_path, "r") as f:
        template_content = f.read()

    # Create Jinja template
    template = Template(template_content)

    # Prepare timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Render the template
    html_content = template.render(
        test_results=test_results,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        total_duration=round(total_duration, 2),
        timestamp=timestamp,
        logo_base64=logo_base64,
        enumerate=enumerate,
        len=len,
    )

    # Generate the report filename
    report_filename = (
        f"lila_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    report_path = os.path.join(output_dir, report_filename)

    # Write the report file
    with open(report_path, "w") as f:
        f.write(html_content)

    return report_path, total_duration


def parse_collection(test_paths: List[str]) -> TestCollection:
    invalid_tests = {}
    valid_tests = {}

    for path in test_paths:
        with open(path, "r") as f:
            content = f.read()
            logger.debug(f"Read file: {path}")

            vars_list = get_vars(content)
            logger.debug(f"Found variables: {vars}")

            missing_vars = get_missing_vars(vars_list)
            if missing_vars:
                logger.debug(f"Missing environment variables: {missing_vars}")
                invalid_tests[path] = f"Missing environment variables: {missing_vars}"
            else:
                try:
                    parsed = yaml.safe_load(content)
                except yaml.YAMLError:
                    invalid_tests[path] = "Provided content is not a valid YAML: {e}"
                else:
                    logger.debug("Content is a valid YAML")
                    try:
                        test = from_dict(data_class=TestCaseDef, data=parsed)
                    except Exception as e:
                        invalid_tests[path] = str(e)
                    else:
                        valid_tests[path] = test

    return TestCollection(valid=valid_tests, invalid=invalid_tests)


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
@click.option(
    "--model",
    type=str,
    help="The LLM model to use",
    default="gpt-4o",
)
@click.option(
    "--provider",
    type=str,
    help="The LLM provider to use",
    default="openai",
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
    model: str = "gpt-4o",
    provider: str = "openai",
):
    """Run a Lila test suite."""
    setup_logging(debug=debug)

    llm = get_langchain_chat_model(model=model, provider=provider)

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

    test_collection = parse_collection(test_files)
    if test_collection.invalid:
        logger.error("Parsing errors found")
        for path, error in test_collection.invalid.items():
            logger.error(f"File {path}: {error}")
        return

    if not test_collection.valid:
        logger.error(
            "No test cases found with the provided path and the provided params"
        )
        return

    # Print test collection in a pytest-like format
    test_count = len(test_collection.valid)
    print(f"\nCollected {test_count} test{'s' if test_count != 1 else ''}")
    print("=" * 70)
    for i, (path, test_def) in enumerate(test_collection.valid.items(), 1):
        # Extract file name from path for cleaner display
        file_name = path.split("/")[-1]
        # Show tags if present
        tags_str = ""
        if hasattr(test_def, "tags") and test_def.tags:
            tags_str = f" [tags: {', '.join(test_def.tags)}]"
        print(f"{i}) {file_name}{tags_str}")
    print("=" * 70)

    runner = TestRunner(test_collection.valid)
    test_results = runner.run_tests(config_obj, browser_state, llm)

    # Print test results summary
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result.status == "success")
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 70)
    print(f"TEST RESULTS SUMMARY: {passed_tests}/{total_tests} passed")
    print("=" * 70)

    # Print detailed results for each test
    for result in test_results:
        test_status = "✅ PASSED" if result.status == "success" else "❌ FAILED"
        duration = f"{result.duration:.2f}s"
        print(f"\n{test_status} {result.path} ({duration})")

        if result.status == "failed":
            # Find the failed step
            for i, step_result in enumerate(result.steps_results):
                if not step_result.action.success:
                    step_type = result.test_def.steps[i].get_type()
                    step_value = result.test_def.steps[i].get_value()
                    print(f"  Failed at step {i+1}: {step_type} {step_value}")
                    print(f"  Reason: {step_result.action.reason}")
                    break
                elif step_result.verifications and any(
                    not v.passed for v in step_result.verifications
                ):
                    step_type = result.test_def.steps[i].get_type()
                    step_value = result.test_def.steps[i].get_value()
                    print(f"  Failed at step {i+1}: {step_type} {step_value}")
                    for j, v in enumerate(step_result.verifications):
                        if not v.passed:
                            print(f"  Verification {j+1} failed: {v.reason}")
                    break

    # Generate HTML report
    output_dir = config_obj.runtime.output_dir
    report_path, total_duration = generate_html_report(test_results, output_dir)

    # Convert to absolute path if needed
    if not os.path.isabs(report_path):
        report_path = os.path.abspath(report_path)

    # Get a file:// URL for the report
    report_url = f"file://{report_path}"

    print("\n" + "=" * 70)
    print(f"📊 DETAILED HTML REPORT: {report_path}")
    print(f"📈 Open in browser: {report_url}")
    print(f"⏱️ Total test duration: {total_duration:.2f}s")
    print("=" * 70)

    print("\n")
    if failed_tests > 0:
        sys.exit(1)


@cli.command()
def init():
    """Initialize a Lila testing template."""
    setup_logging(debug=False)

    if os.path.exists("lila.toml"):
        logger.error(
            "Config file lila.toml already exists, it seems like the application is already initialized."
        )
        return

    # Create a lila.toml file
    config_path = Path(__file__).parent / "assets" / "lila.toml"
    shutil.copy(config_path, "lila.toml")
    logger.info("Config file created: lila.toml")

    # Create a gitignore file
    gitignore_path = Path(__file__).parent / "assets" / "gitignore"
    shutil.copy(gitignore_path, ".gitignore")
    logger.info("Generated gitignore file")

    dotenv_template = Path(__file__).parent / "assets" / "env"
    if not os.path.exists(os.path.join(os.getcwd(), ".env")):
        shutil.copy(dotenv_template, ".env")
        logger.info("Generated .env file")

    # Create a lila directory
    os.makedirs("lila-output", exist_ok=True)
    logger.info("Output directory created for artifacts: lila-output/")

    # Create an example test case
    example_path = Path(__file__).parent / "assets" / "google-maps.yaml"
    shutil.copy(example_path, "demo.yaml")
    logger.info("Example test case created: demo.yaml")
    logger.success("All set! Run your first test: lila run demo.yaml")
