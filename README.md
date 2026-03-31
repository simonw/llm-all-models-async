# llm-all-models-async

[![PyPI](https://img.shields.io/pypi/v/llm-all-models-async.svg)](https://pypi.org/project/llm-all-models-async/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-all-models-async?include_prereleases&label=changelog)](https://github.com/simonw/llm-all-models-async/releases)
[![Tests](https://github.com/simonw/llm-all-models-async/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-all-models-async/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-all-models-async/blob/main/LICENSE)

Make all LLM sync models available as async

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-all-models-async
```
## Usage

Usage instructions go here.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-all-models-async
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
python -m pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
