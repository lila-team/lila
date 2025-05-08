import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from lila.config import Config
from lila.models import TestCaseDef


class TestRunner:
    def __init__(self, testcases: Dict[str, TestCaseDef]):
        self.testcases = testcases

    def run_tests(
        self,
        config: Config,
        browser_state: Optional[str],
        llm: BaseChatModel,
    ) -> None:
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
