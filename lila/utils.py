import os
import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Pointer:
    x: int
    y: int


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


def get_vars_from_env(
    vars_list: List[str], fail_if_missing: bool = True
) -> Dict[str, str]:
    # Returns a dictionary with the values of the environment variables
    # specified in vars_list
    ret = {}
    for var in vars_list:
        if os.environ.get(var) is None:
            if fail_if_missing:
                raise RuntimeError(f"Environment variable {var} not found")
        else:
            ret[var] = os.environ[var]

    return ret
