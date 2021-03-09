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
Install development packages:
```shell
pip install -e .
```
Run tests:
```shell
pip install -e .[testing]
pytest
```
Build documentation:
```shell
pip install -e .[docs]
python setup.py docs -W
```
Install pre-commit, that will automatically apply black to all your modified python files at commit time:
```shell
pip install -e .[dev]
pip install pre-commit
pre-commit install
```
Notice that if black modifies some files, the commit will fail, and you will need to commit again.

**Note:** Do not push directly to master! Please, submit a MR for review, make sure that Gitlab CI/CD pipelines pass.