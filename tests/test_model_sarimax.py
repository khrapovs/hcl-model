import numpy as np
import pandas as pd

from hcl_model.model_sarimax import SARIMAXModel


class TestSARIMAX:

    @staticmethod
    def generate_data():

        nobs = 30
        index = pd.date_range('2019-01-01', periods=nobs, freq='W-FRI', name='date')
        endog = pd.DataFrame({'value': np.arange(1, nobs+1) + np.random.normal(size=nobs)}, index=index)
        exog = pd.DataFrame({'const': np.ones(nobs), 'time': np.arange(1, nobs+1)}, index=index)
        return endog, exog

    def test_model_fit(self):

        endog, exog = self.generate_data()

        model = SARIMAXModel(trend='n')
        model.fit(endog=endog['value'])
        parameters = model._get_parameters()

        # print(parameters)
        # Some random test. No good logic here
        assert list(parameters.index) == ['sigma2']
        assert model._trend_fit is None

        model = SARIMAXModel(trend='c')
        model.fit(endog=endog['value'])
        parameters = model._get_parameters()

        # print(parameters)
        # Some random test. No good logic here
        assert list(parameters.index) == ['sigma2']
        assert set(model._trend_fit.params.index.values) == {'const'}

        model = SARIMAXModel(trend='ct')
        model.fit(endog=endog['value'])
        parameters = model._get_parameters()

        # print(parameters)
        # Some random test. No good logic here
        assert list(parameters.index) == ['sigma2']
        assert set(model._trend_fit.params.index.values) == {'const', 'trend'}

        model = SARIMAXModel(trend='t')
        model.fit(endog=endog['value'])
        parameters = model._get_parameters()

        # print(parameters)
        # Some random test. No good logic here
        assert list(parameters.index) == ['sigma2']
        assert set(model._trend_fit.params.index.values) == {'trend'}

        model = SARIMAXModel(trend='n')
        model.fit(endog=endog['value'], exog=exog)
        parameters = model._get_parameters()

        # print(parameters)
        # Some random test. No good logic here
        assert list(parameters.index) == ['const', 'time', 'sigma2']
        assert model._trend_fit is None

    def test_model_prediction(self):

        endog, exog = self.generate_data()
        model = SARIMAXModel(trend='ct')
        num_steps = 10
        lbl_value = 'value'
        model.fit(endog=endog.loc[endog.index[:-num_steps], lbl_value], exog=exog)
        forecast = model.predict(num_steps=num_steps)

        # print(forecast)
        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == lbl_value

    def test_model_simulation(self):

        endog, exog = self.generate_data()
        model = SARIMAXModel(trend='ct')
        num_steps = 10
        num_simulations = 5

        model.fit(endog=endog.loc[endog.index[:-num_steps], 'value'], exog=exog)
        simulations = model.simulate(num_steps=num_steps, num_simulations=num_simulations)

        # print(simulations)
        assert isinstance(simulations, pd.DataFrame)
        assert simulations.shape == (num_steps, num_simulations)

    def test_model_percentiles(self):
        endog, exog = self.generate_data()
        model = SARIMAXModel(trend='ct')
        num_steps = 10
        num_simulations = 5
        quantile_levels = [5, 95]
        lbl_value = 'value'

        model.fit(endog=endog.loc[endog.index[:-num_steps], lbl_value], exog=exog)
        forecast = model.predict(num_steps=num_steps, quantile_levels=quantile_levels,
                                 num_simulations=num_simulations)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (num_steps, len(quantile_levels) + 1)
        assert forecast.columns[0] == lbl_value

    def test_model_summary(self):
        endog, exog = self.generate_data()

        model = SARIMAXModel(trend='n')
        model.fit(endog=endog['value'])

        assert set(model.summary().index) >= {model._lbl_aic, model._lbl_r2, model._lbl_mape,
                                              model._lbl_resid_mean, model._lbl_resid_std,
                                              model._lbl_resid_skewness, model._lbl_resid_kurtosis}
