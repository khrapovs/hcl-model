# hcl-model

This package implements Hand Crafted Linear (HCL) model for time series forecasting. It is used in [Ocean TME project](https://git.signintra.com/trade-management-platform/popeyethesailor).

This package is hosted on [TSC GitLab](https://git.signintra.com/trade-management-platform/hcl-model).

[**Documentation**](https://nexus.signintra.com/repository/GDSA-static/packages/hcl-model/docs/index.html)

## Installation

Install from [private Nexus PyPi repository](https://nexus.signintra.com/#browse/browse:GDSA-PyPi):
```bash
pip install -extra-index-url https://${NEXUS_LOGIN}:${NEXUS_PASSWORD}@nexus.signintra.com/repository/GDSA-PyPi/simple hcl-model
```

If you use `pipenv`, then in `Pipfile` add the following lines:
```text
[[source]]
name = "nexus"
url = "https://${NEXUS_LOGIN}:${NEXUS_PASSWORD}@nexus.signintra.com/repository/GDSA-PyPi/simple"
verify_ssl = true
```
with `NEXUS_LOGIN` and `NEXUS_PASSWORD` credentials as environmental variables (ask developers).

## Testing

Run
```bash
python setup.py test
```
