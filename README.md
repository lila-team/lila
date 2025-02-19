[<img src="assets/logo.png" width=300>](https://lila.dev/)

[![PyPI version](https://badge.fury.io/py/lilacli.svg)](https://badge.fury.io/py/lilacli)
[![Documentation Status](https://readthedocs.org/projects/lila/badge/?version=latest)](https://docs.lila.dev)
[![CI](https://github.com/lila-team/lila/actions/workflows/daily-run.yml/badge.svg)](https://github.com/lila-team/lila/actions/workflows/daily-run.yml)
[![Twitter](https://img.shields.io/twitter/follow/lila__dev?style=social)](https://twitter.com/lila__dev)
[![Discord](https://img.shields.io/discord/1303067047931936809?label=Discord)](https://discord.gg/kZ7TEmxH)
![GitHub Repo stars](https://img.shields.io/github/stars/lila-team/lila)

**[Lila](https://lila.dev) is the best testing framework for fast moving teams developing webapps.**

Lila CLI is a powerful tool for running end-to-end tests written in human-readable plain text using YAML files. It simplifies the testing process by allowing anyone in the team to write tests in a natural, easy-to-understand format.

**No coding required.**


### How does it work?

Lila runs your app in a **local browser** with [Playwright](https://playwright.dev/python/) and uses an LLM-powered engine to guide the CLI run the high level instructions.

### Why Lila?

* No coding required.
* Self healing tests, does not rely on low level implementation.
* Anyone in the team can implement tests.
* Integrates natively with Playwright storage states.
* Runs browser locally, making it ideal for localdev or staging environments.

### Test example

```yaml
steps:
  - goto: https://www.google.com/maps

  - input: Empire State on the main search bar
    verify: a dropdown with suggestions appears

  - click: on the first suggestion from the dropdown
    verify: the map shows the Empire State Building in NY
```

For more information on how to build tests, [checkout the guides](https://docs.lila.dev/guides/intro)

For test examples, Lila runs daily a suite of tests over public URLs. Check them out [here](https://github.com/lila-team/examples)

## Getting started

### Create a free account

You will need a free account to try Lila. Create your free account at [https://lila.dev](https://lila.dev)

### Installation

Lila CLI requires Python 3.11 or higher. Install it using pip:

```bash
pip install lilacli
```

Install Chromium web drivers for Playwright

```bash
playwright install chromium
```

### Quick Start

Initialize a new Lila test project:

```bash
lila init
```

This command creates a template structure with example tests to help you get started.

### Running Tests

Fetch your `LILA_API_KEY` from the [app](https://app.lila.dev) and run your tests using the `run` command:

```bash
LILA_API_KEY=<your-api-key> lila run <folder/file>
```

#### Command Line Options

* `--browser-state`: Initialize the browser with a Playwright JSON storage session
* `--tags`: Filter tests by specified comma-separated tags
* `--exclude-tags`: Filter out tests by specified comma-separated tags
* `--output-dir`: Specify the output directory for the browser states
* `--config`: Point to the `lila.toml` config file if not using the root location
* `--headless`: If present, does not open browser
* `--debug`: Displays debug logs

#### Example runs

```bash
lila run tests/
lila run tests/one-specific-test.yaml
lila run tests/ --tags foo,bar
lila run tests/subtests --exclude-tags baz
lila run tests/user-login.yaml
lila run tests/user.yaml --browser-state tests/user-login.json
```

### Configuring Lila

Configuration can be set in `lila.toml` file. For the full spec, [checkout the documentation](https://docs.lila.dev/docs/guides/configuration)

Configuration example

```toml
[browser]
type = "chromium"
width = 1536
height = 1152

[runtime]
fail_fast = true
output_dir = "lila-output"
concurrent_workers = 4
server_url = 'https://app.lila.dev'
```

## Documentation

For comprehensive documentation, visit [docs.lila.dev](https://docs.lila.dev).

## Examples

Find a variety of test examples in our [examples repository](https://github.com/lila-team/examples), showcasing different testing scenarios and best practices.

## Contact

Reach out at info@lila.dev

Join our [Discord server](https://discord.gg/6rRfZUqh)

Follow us at [Twitter - X](https://x.com/lila__dev)
