# hcl-model

## [Full documentation](https://nexus.signintra.com/repository/GDSA-static/packages/hcl-model/docs/index.html)

This package implements Hand Crafted Linear (HCL) model for time series forecasting. It is used in [Ocean TME project](https://git.signintra.com/trade-management-platform).

This package is hosted on [TSC GitLab](https://git.signintra.com/gdsa/python-libs/hcl-model).

## Installation

Simply run while being inside of Schenker network:
```shell
export PIP_EXTRA_INDEX_URL="https://${NEXUS_LOGIN}:${NEXUS_PASSWORD}@nexus.signintra.com/repository/pypi-all/simple"
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
Build documentation (see more details [here](https://www.mkdocs.org/#getting-started)):
```shell
pip install -e .[doc]
mkdocs build
```
or use
```shell
mkdocs serve
```
if you prefer a live, self-refreshing, documentation.

**Note:** Do not push directly to master! Please, submit a MR for review, make sure that Gitlab CI/CD pipelines pass.
