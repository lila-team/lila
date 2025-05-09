import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from lila.config import Config
from lila.models import TestCaseDef, TestCaseRun


class TestRunner:
    def __init__(self, testcases: Dict[str, TestCaseDef]):
        self.testcases = testcases

    def run_tests(
        self,
        config: Config,
        browser_state: Optional[str],
        llm_factory: Callable[[], BaseChatModel],
    ) -> List[TestCaseRun]:
        future_to_test = {}
        results = []

        with ThreadPoolExecutor(
            max_workers=config.runtime.concurrent_workers
        ) as executor:
            for path, testcase in self.testcases.items():
                # For debugging purposes
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
                    llm_factory,
                )
                future_to_test[key] = testcase

            for future in as_completed(future_to_test.keys()):
                testcase_run = future.result()
                results.append(testcase_run)

                # If fail_fast is enabled and the test failed, stop executing remaining tests
                if config.runtime.fail_fast and testcase_run.status == "failed":
                    for f in list(future_to_test.keys()):
                        if not f.done():
                            f.cancel()
                    break

        return results
