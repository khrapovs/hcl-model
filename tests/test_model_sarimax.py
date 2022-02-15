import pandas as pd

from hcl_model.model_sarimax import SARIMAXModel
from tests.test_model_common import TestModelCommon


class TestSARIMAX(TestModelCommon):
    def test_model_fit(self):

        endog, exog = self.generate_data()

        model = SARIMAXModel(trend="n")
        y_train = endog[self.lbl_value]
        model.fit(y=y_train)
        parameters = model.get_parameters()

        assert list(parameters.index) == ["sigma2"]
        assert model._trend_fit is None
        assert set(model.summary()[model.lbl_params].keys()) == {"sigma2"}

        model = SARIMAXModel(trend="c")
        model.fit(y=y_train)
        parameters = model.get_parameters()

        assert list(parameters.index) == ["sigma2"]
        assert set(model._trend_fit.params.index.values) == {"const"}
        # trend is extracted before fitting SARIMAX, hence no 'const' among parameters
        assert set(model.summary()[model.lbl_params].keys()) == {"sigma2"}

        model = SARIMAXModel(trend="ct")
        model.fit(y=y_train)
        parameters = model.get_parameters()

        assert list(parameters.index) == ["sigma2"]
        assert set(model._trend_fit.params.index.values) == {"const", "trend"}
        assert set(model.summary()[model.lbl_params].keys()) == {"sigma2"}

        model = SARIMAXModel(trend="t")
        model.fit(y=y_train)
        parameters = model.get_parameters()

        assert list(parameters.index) == ["sigma2"]
        assert set(model._trend_fit.params.index.values) == {"trend"}
        assert set(model.summary()[model.lbl_params].keys()) == {"sigma2"}

        model = SARIMAXModel(trend="n")
        model.fit(y=y_train, X=exog)
        parameters = model.get_parameters()

        assert list(parameters.index) == ["const", "time", "sigma2"]
        assert model._trend_fit is None
        assert set(model.summary()[model.lbl_params].keys()) == {"const", "time", "sigma2"}

    def test_model_prediction(self):
        endog, exog = self.generate_data()
        model = SARIMAXModel(trend="ct")
        num_steps = 10
        y_train = endog.loc[endog.index[:-num_steps], self.lbl_value]
        x_train = exog.loc[endog.index[:-num_steps]]
        x_test = exog.loc[endog.index[-num_steps:]]
        model.fit(y=y_train, X=x_train)
        forecast = model.predict(num_steps=num_steps, X=x_test)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == self.lbl_value
        assert forecast.index.name == self.lbl_date
        assert isinstance(forecast.index, pd.DatetimeIndex)

    def test_model_simulation(self):
        endog, exog = self.generate_data()
        model = SARIMAXModel(trend="ct")
        num_steps = 10
        num_simulations = 5

        y_train = endog.loc[endog.index[:-num_steps], self.lbl_value]
        x_train = exog.loc[endog.index[:-num_steps]]
        x_test = exog.loc[endog.index[-num_steps:]]
        model.fit(y=y_train, X=x_train)
        simulations = model.simulate(num_steps=num_steps, num_simulations=num_simulations, X=x_test)

        assert isinstance(simulations, pd.DataFrame)
        assert simulations.shape == (num_steps, num_simulations)
        assert simulations.index.name == self.lbl_date
        assert isinstance(simulations.index, pd.DatetimeIndex)

    def test_model_percentiles(self):
        endog, exog = self.generate_data()
        model = SARIMAXModel(trend="ct")
        num_steps = 10
        num_simulations = 5
        quantile_levels = [5, 95]

        y_train = endog.loc[endog.index[:-num_steps], self.lbl_value]
        x_train = exog.loc[endog.index[:-num_steps]]
        model.fit(y=y_train, X=x_train)
        x_test = exog.loc[endog.index[-num_steps:]]
        forecast = model.predict(
            num_steps=num_steps, quantile_levels=quantile_levels, num_simulations=num_simulations, X=x_test
        )

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (num_steps, len(quantile_levels) + 1)
        assert forecast.columns[0] == self.lbl_value
        assert forecast.index.name == self.lbl_date
        assert isinstance(forecast.index, pd.DatetimeIndex)

    def test_model_summary(self):
        endog, exog = self.generate_data()

        model = SARIMAXModel(trend="n")
        model.fit(y=endog[self.lbl_value])

        assert set(model.summary().index) >= {
            model.lbl_aic,
            model.lbl_r2,
            model.lbl_mape,
            model.lbl_resid_mean,
            model.lbl_resid_std,
            model.lbl_resid_skewness,
            model.lbl_resid_kurtosis,
            model.lbl_params,
        }
