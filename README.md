[<img src="assets/logo.png" width=300>](https://lila.dev/)

<h1>The best testing framework for startups 🚀</h1>

[![PyPI - Version](https://img.shields.io/pypi/v/lilacli?color=blue)](https://pypi.org/project/lilacli/)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.lila.dev)
[![CI](https://github.com/lila-team/lila/actions/workflows/daily-run.yml/badge.svg)](https://github.com/lila-team/lila/actions/workflows/daily-run.yml)
[![Twitter](https://img.shields.io/twitter/follow/lila__dev?style=social)](https://twitter.com/lila__dev)
[![Discord](https://img.shields.io/discord/1303067047931936809?label=Discord)](https://discord.gg/kZ7TEmxH)
![GitHub Repo stars](https://img.shields.io/github/stars/lila-team/lila)

Lila CLI is a powerful tool for running end-to-end tests written in human-readable plain text using YAML files. It simplifies the testing process by allowing anyone in the team to write tests in a natural, easy-to-understand format.

**No coding required.**

## Quick start

With pip (Python>=3.11)

```
pip install lilacli
```

Install Playwright Chromium web drivers

```
playwright install chromium
```

Create your first testcase in a `demo.yaml` file

```yaml
steps:
  - goto: https://www.google.com/maps

  - input: Empire State on the main search bar
    verify: a dropdown with suggestions appears

  - click: on the first suggestion from the dropdown
    verify: the map shows the Empire State Building in NY
```

Fetch an API Key from [Lila app](https://app.lila.dev) and add it to an `.env` file:

```
LILA_API_KEY=...
```

Run

```
lila run demo.yaml
```

## How does it work?

Lila runs your app in a **local browser** with [Playwright](https://playwright.dev/python/) and uses an LLM-powered engine to guide the CLI run high level instructions. Lila extracts information from the DOM using the awesome library [browser-use](https://github.com/browser-use/browser-use)

### Why Lila?

* No coding required.
* Self healing tests, does not rely on low level implementation.
* Anyone in the team can implement tests.
* Integrates natively with Playwright storage states.
* Runs browser locally, making it ideal for localdev or staging environments.

### Writing tests

Tests are just YAML files. Here is the same example as the quick start.

```yaml
steps:
  - goto: https://www.google.com/maps

  - input: Empire State on the main search bar
    verify: a dropdown with suggestions appears

  - click: on the first suggestion from the dropdown
    verify: the map shows the Empire State Building in NY
```

### Step types

**goto**: navigate to a URL

```yaml
- goto: https://private.staging.my-app.com
```

**click**: perform a click on a described element

```yaml
- click: on the checkout cart
# or
- click: on the direction item
# or
- click: on button 'Create'
```

**input**: type text into an input field

```yaml
- input: foo@bar.com as email
# or
- input: iPad mini in the main search bar
```

**submit**: submits a form by clicking a button or pressing enter

```yaml
- submit: the login form
# or
- submit: the search
```

**wait**: just waits N seconds

```yaml
- wait: 10  # waits 10 seconds
```

**pick**: pick an element from a dropdown

```yaml
- pick: EST from the timezone dropdown
```

**exec**: run a bash command

```yaml
- exec: |  # multiline
    curl -X POST ...

# or
- exec: psql -c "INSERT ..." # single line
```

### Verifications

Each step can have one or more `verify` assertions.

```yaml
- goto: https://my-app.com/login
  verify: there is a login form with username and password  # Single verification

- click: the checkout cart
  verify:  # multi verifications
    - 5 items appear in the purchase summary
    - the total is 500 USD
```

### More on building tests

For more information on how to build tests, [checkout the guides](https://docs.lila.dev/guides/intro)

### More examples

For test examples, Lila runs daily a suite of tests over public URLs. Check them out [here](https://github.com/lila-team/examples)

## Creating an account

Lila is free for beta users. Create your free account at [https://lila.dev](https://lila.dev).
You will need the account to fetch an API key to run lila CLI.


## Reusing state

Lila runs save the browser state in the output folder. To re-use where another test left off, just pass `--browser-state` parameter.

For example:

**login.yaml**:

```yaml
steps:
  - goto: https://my-app.com/login
  - input: foobar as username
  - input: barbaz as password
  - submit: the login form
    verify: login was successful
```

`lila run login.yaml` -> This will generate `lila-output/login.json`

**dashboard.yaml** (needs a logged in user)

```yaml
steps:
  - goto: https://my-app.com/dashboard
    verify: there is a panel displaying metrics
```

`lila run dashboard.yaml --browser-state lila-output/login.json`

## CLI Parameters

* `--browser-state`: Initialize the browser with a Playwright JSON storage session
* `--tags`: Filter tests by specified comma-separated tags
* `--exclude-tags`: Filter out tests by specified comma-separated tags
* `--output-dir`: Specify the output directory for the browser states
* `--config`: Point to the `lila.toml` config file if not using the root location
* `--headless`: If present, does not open browser
* `--debug`: Displays debug logs

## CI/CD ready

Checkout [an example for a daily run using GHA](.github/workflows/daily-run.yaml)

## Documentation

For comprehensive documentation, visit [docs.lila.dev](https://docs.lila.dev).

## Examples

Find a variety of test examples in our [examples repository](https://github.com/lila-team/examples), showcasing different testing scenarios and best practices.

## Contact

Reach out at info@lila.dev

Join our [Discord server](https://discord.gg/6rRfZUqh)

Follow us at [Twitter - X](https://x.com/lila__dev)
