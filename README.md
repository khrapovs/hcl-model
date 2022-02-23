# HCL Model

![PyTest](https://github.com/khrapovs/hcl-model/actions/workflows/pytest.yaml/badge.svg)
![Docs](https://github.com/khrapovs/hcl-model/actions/workflows/docs.yaml/badge.svg)
[![!pypi](https://img.shields.io/pypi/v/hcl-model)](https://pypi.org/project/hcl-model/)
[![codecov](https://codecov.io/gh/khrapovs/hcl-model/branch/main/graph/badge.svg?token=KC0XT6R18H)](https://codecov.io/gh/khrapovs/hcl-model)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![!python-versions](https://img.shields.io/pypi/pyversions/hcl-model)](https://pypi.org/project/hcl-model/)

Simple time series forecasting based on multiple linear regression.

## Documentation

Full documentations hosted on GitHub pages: [khrapovs.github.io/hcl-model](https://khrapovs.github.io/hcl-model/).

## Installation

```shell
pip install hcl-model
```

## Contribute

Create a virtual environment and activate it
```shell
python -m venv venv
source venv/bin/activate
```
Install the development packages
```shell
pip install -e .[dev]
```
and use pre-commit to make sure that your code is formatted using [black](https://github.com/PyCQA/isort) and [isort](https://pycqa.github.io/isort/index.html) automatically:
```shell
pre-commit install
```
Run tests:
```shell
pip install -e .[test]
pytest
```
Build documentation:
```shell
pip install -e .[doc]
mkdocs build
```
or use
```shell
mkdocs serve
```
if you prefer a live, self-refreshing, documentation.
