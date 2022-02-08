# HCL Model

![PyTest](https://github.com/khrapovs/hcl-model/actions/workflows/pytest.yaml/badge.svg)
![Docs](https://github.com/khrapovs/hcl-model/actions/workflows/docs.yaml/badge.svg)
[![!pypi](https://img.shields.io/pypi/v/hcl-model?color=orange)](https://pypi.org/project/hcl-model/)

Simple time series forecasting based on multiple linear regression.

[Full documentation](https://khrapovs.github.io/hcl-model/)

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
and use pre-commit to make sure that your code is blackified automatically (used the `black` package):
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
