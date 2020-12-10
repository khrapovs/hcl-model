# hcl-model

## [Full documentation](https://nexus.signintra.com/repository/GDSA-static/packages/hcl-model/docs/index.html)

This package implements Hand Crafted Linear (HCL) model for time series forecasting. It is used in [Ocean TME project](https://git.signintra.com/trade-management-platform).

This package is hosted on [TSC GitLab](https://git.signintra.com/gdsa/python-libs/hcl-model).

## Installation

Install from [private Nexus PyPi repository](https://nexus.signintra.com/#browse/browse:pypi-all):
```bash
pip install -extra-index-url https://nexus.signintra.com/repository/pypi-all/simple hcl-model
```

If you use `pipenv`, then in `Pipfile` add the following lines:
```text
[[source]]
name = "nexus"
url = "https://nexus.signintra.com/repository/pypi-all/simple"
verify_ssl = true
```

## Contribute

Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate
```

Install development packages:
```bash
pip install -e .
```

Run tests:
```bash
pip install -e .[testing]
python setup.py test
```

Build documentation:
```bash
pip install -e .[docs]
python setup.py docs -W
```

**Note:** Do not push directly to master! Please, submit a MR for review, make sure that Gitlab CI/CD pipelines pass.