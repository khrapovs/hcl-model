import functools

import numpy as np
import pandas as pd
import statsmodels.tsa.tsatools as tst

from hcl_model.model_hcl_generic import HandCraftedLinearModel


class TestHCL:

    @staticmethod
    def generate_data():

        nobs = 30
        index = pd.date_range('2019-01-01', periods=nobs, freq='W-FRI', name='date')
        endog = pd.DataFrame({'value': np.arange(1, nobs+1) + np.random.normal(size=nobs)}, index=index)
        exog = pd.DataFrame({'const': np.ones(nobs), 'time': np.arange(1, nobs+1)}, index=index)
        return endog, exog

    def test_model_fit(self):

        endog, exog = self.generate_data()

        model = HandCraftedLinearModel()
        model.fit(endog=endog['value'], exog=exog)
        parameters = model._get_parameters()

        # Some random test. No good logic here
        assert list(parameters.index) == ['const', 'time']

    def test_model_prediction(self):

        endog, exog = self.generate_data()
        model = HandCraftedLinearModel()
        num_steps = 10
        lbl_value = 'value'
        model.fit(endog=endog.loc[endog.index[:-num_steps], lbl_value], exog=exog)
        forecast = model.predict(num_steps=num_steps)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == lbl_value

    def test_model_simulation(self):

        endog, exog = self.generate_data()
        model = HandCraftedLinearModel()
        num_steps = 10
        num_simulations = 5

        model.fit(endog=endog.loc[endog.index[:-num_steps], 'value'], exog=exog)
        simulations = model.simulate(num_steps=num_steps, num_simulations=num_simulations)

        assert isinstance(simulations, pd.DataFrame)
        assert simulations.shape == (num_steps, num_simulations)

    def test_model_percentiles(self):
        endog, exog = self.generate_data()
        model = HandCraftedLinearModel()
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

    def test_summary(self):
        endog, exog = self.generate_data()

        model = HandCraftedLinearModel()
        model.fit(endog=endog['value'], exog=exog)

        assert set(model.summary().index) >= {model._lbl_aic, model._lbl_r2, model._lbl_mape,
                                              model._lbl_resid_mean, model._lbl_resid_std,
                                              model._lbl_resid_skewness, model._lbl_resid_kurtosis}


class TestHCLTransforms:

    @staticmethod
    def generate_input():
        nobs = 30
        endog = pd.Series(np.arange(1, nobs + 1) + np.random.normal(size=nobs, scale=1e-1), name='value',
                          index=pd.date_range('2019-01-01', periods=nobs, freq='W-FRI', name='date'))

        data = tst.add_trend(endog, trend='ct')
        data['x3'] = 999

        f = {'lag1': lambda y: y.shift(1),
             'local_mean': lambda y: y.shift(1).ewm(span=5).mean()}

        g = {'const': lambda x: x + 10,
             'trend': lambda x: -x}

        return data, f, g

    def test_transform_lags(self):
        lags = [1, 2, 10]
        col_name = 'lag_{}'

        f = {col_name.format(lag): functools.partial(lambda lag, y: y.shift(lag), lag) for lag in lags}
        model = HandCraftedLinearModel(endog_transform=f)
        endog = pd.Series(np.arange(5))

        df = model._transform_endog(endog)
        df_expected = pd.DataFrame()
        for lag in lags:
            df_expected = df_expected.assign(**{col_name.format(lag): endog.shift(lag)})

        pd.testing.assert_frame_equal(df, df_expected)

    def test_model_fit(self):
        data, f, g = self.generate_input()

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(endog=data['value'], exog=data.iloc[:, 1:])

        parameters = model._get_parameters()

        # Some random test. No good logic here
        assert list(parameters.index) == ['lag1', 'local_mean', 'const', 'trend']
        assert parameters.isna().sum() == 0

    def test_model_prediction(self):
        data, f, g = self.generate_input()

        num_steps = 5
        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(endog=data.loc[data.index[:-num_steps], 'value'], exog=data.iloc[:-num_steps, 1:])

        forecast = model.predict(exog=data.iloc[:, 1:], num_steps=num_steps)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == 'value'
        assert forecast.isna().sum().sum() == 0

    def test_model_percentiles(self):
        data, f, g = self.generate_input()

        num_steps = 5
        num_simulations = 5
        quantile_levels = [5, 95]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(endog=data.loc[data.index[:-num_steps], 'value'], exog=data.iloc[:-num_steps, 1:])

        forecast = model.predict(exog=data.iloc[:, 1:], num_steps=num_steps, quantile_levels=quantile_levels,
                                 num_simulations=num_simulations)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (num_steps, len(quantile_levels) + 1)
        assert forecast.columns[0] == 'value'
        assert forecast.isna().sum().sum() == 0


class TestHCLWeightedTransforms:

    @staticmethod
    def generate_input():
        nobs = 30
        endog = pd.Series(np.arange(1, nobs + 1) + np.random.normal(size=nobs, scale=1e-1), name='value',
                          index=pd.date_range('2019-01-01', periods=nobs, freq='W-FRI', name='date'))

        data = tst.add_trend(endog, trend='ct')
        data['x3'] = 999

        f = {'lag1': lambda y: y.shift(1),
             'local_mean': lambda y: y.shift(1).ewm(span=5).mean()}

        g = {'const': lambda x: x + 10,
             'trend': lambda x: -x}

        weights = np.power(np.arange(start=0, stop=1, step=1/len(endog)), 2)

        return data, f, g, weights

    def test_model_fit(self):
        data, f, g, weights = self.generate_input()

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(endog=data['value'], exog=data.iloc[:, 1:], weights=weights)

        parameters = model._get_parameters()

        # Some random test. No good logic here
        assert list(parameters.index) == ['lag1', 'local_mean', 'const', 'trend']
        assert parameters.isna().sum() == 0

    def test_model_prediction(self):
        data, f, g, weights = self.generate_input()

        num_steps = 5
        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(endog=data.loc[data.index[:-num_steps], 'value'],
                  exog=data.iloc[:-num_steps, 1:],
                  weights=weights[:-num_steps])

        forecast = model.predict(exog=data.iloc[:, 1:], num_steps=num_steps)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == 'value'
        assert forecast.isna().sum().sum() == 0
