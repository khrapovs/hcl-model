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

## Testing

Run
```bash
python setup.py test
```
