import tomllib
from dataclasses import dataclass

from dacite import from_dict

from lila import const


@dataclass
class Browser:
    type: str
    width: int
    height: int
    headless: bool

    @classmethod
    def default(cls):
        return cls(type="chromium", width=const.WIDTH, height=const.HEIGHT)


@dataclass
class Runtime:
    concurrent_workers: int
    fail_fast: bool
    output_dir: str
    server_url: str

    @classmethod
    def default(cls):
        return cls(
            concurrent_workers=5,
            fail_fast=False,
            output_dir="lila-results",
            server_url=const.BASE_URL,
        )


@dataclass
class Config:
    browser: Browser
    runtime: Runtime

    @classmethod
    def from_toml_file(cls, toml_file: str) -> "Config":
        with open(toml_file, "rb") as file:
            toml_dict = tomllib.load(file)
            return from_dict(data_class=cls, data=toml_dict)

    @classmethod
    def default(cls):
        return cls(browser=Browser.default(), runtime=Runtime.default())
