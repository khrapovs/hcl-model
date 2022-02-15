import functools
from typing import Dict, Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import add_trend

from hcl_model.model_hcl_generic import HandCraftedLinearModel
from hcl_model.transformers.calendar import CalendarTransformer
from tests.test_model_common import TestModelCommon


class TestHCL(TestModelCommon):
    def test_model_fit(self):
        endog, x_train = self.generate_data()
        y_train = endog[self.lbl_value]

        model = HandCraftedLinearModel()
        model.fit(y=y_train, X=x_train)
        parameters = model.get_parameters()

        params_expected = ["{} const".format(model.lbl_original_exog), "{} time".format(model.lbl_original_exog)]
        assert list(parameters.index) == params_expected
        assert set(model.summary()[model.lbl_params].keys()) == set(params_expected)

    def test_model_prediction(self):
        endog, exog = self.generate_data()
        model = HandCraftedLinearModel()
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
        model = HandCraftedLinearModel()
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
        model = HandCraftedLinearModel()
        num_steps = 10
        num_simulations = 5
        quantile_levels = [5, 95]

        y_train = endog.loc[endog.index[:-num_steps], self.lbl_value]
        x_train = exog.loc[endog.index[:-num_steps]]
        x_test = exog.loc[endog.index[-num_steps:]]

        model.fit(y=y_train, X=x_train)
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

        model = HandCraftedLinearModel()
        model.fit(y=endog[self.lbl_value], X=exog)

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


class TestHCLTransforms:
    lbl_date = "date"
    lbl_value = "value"

    def generate_input(self) -> pd.DataFrame:
        nobs = 30
        endog = pd.Series(
            np.arange(1, nobs + 1) + np.random.normal(size=nobs, scale=1e-1),
            name=self.lbl_value,
            index=pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name=self.lbl_date),
        )
        data = endog.to_frame()
        data["x2"] = np.random.normal(size=nobs)
        data["x3"] = np.random.normal(size=nobs)
        return data

    @staticmethod
    def _get_exog_transform() -> Dict[str, Callable]:
        return {"const": lambda x: x + 10, "trend": lambda x: -x.shift(2)}

    @staticmethod
    def _get_endog_transform() -> Dict[str, Callable]:
        return {
            "lag1": lambda y: y.shift(1),
            "local_mean": lambda y: y.shift(1).ewm(span=5).mean(),
        }

    def test_transform_lags(self):
        lags = [1, 2, 10]
        col_name = "lag_{}"

        f = {col_name.format(lag): functools.partial(lambda lag, y: y.shift(lag), lag) for lag in lags}
        model = HandCraftedLinearModel(endog_transform=f)
        endog = pd.Series(np.arange(5))

        transformed = model._transform_data(data=endog, transform=f)
        transformed_expected = {col_name.format(lag): endog.shift(lag) for lag in lags}

        for key, val in transformed.items():
            pd.testing.assert_series_equal(val, transformed_expected[key])

    def test_transform_data(self):
        data = self.generate_input()
        f = self._get_endog_transform()
        g = self._get_exog_transform()

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        endog = data[self.lbl_value]
        exog = data.iloc[:, 1:]

        transformed = model._transform_data(data=endog, transform=f)
        transformed_expected = {key: endog.transform(f[key]) for key in f.keys()}

        for key, val in transformed.items():
            pd.testing.assert_series_equal(val, transformed_expected[key])

        transformed = model._transform_data(data=exog, transform=g)
        transformed_expected = {key: exog.transform(g[key]) for key in g.keys()}

        for key, val in transformed.items():
            pd.testing.assert_frame_equal(val, transformed_expected[key])

        transformed = model._transform_all_data(exog=exog, endog=endog)
        transformed_endog_expected = {key: endog.transform(f[key]) for key in f.keys()}
        transformed_exog_expected = {key: exog.transform(g[key]) for key in g.keys()}

        for key, val in transformed_endog_expected.items():
            pd.testing.assert_series_equal(val, transformed[key])
        for key, val in transformed_exog_expected.items():
            pd.testing.assert_frame_equal(val, transformed[key])

    def test_convert_transformed_dict_to_frame(self):
        data = self.generate_input()
        f = self._get_endog_transform()
        g = self._get_exog_transform()

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        endog = data[self.lbl_value]
        exog = data.iloc[:, 1:]

        transformed = model._transform_all_data(exog=exog, endog=endog)
        transformed_df = model._convert_transformed_dict_to_frame(transformed=transformed)

        keys = set(f.keys())
        for key in g.keys():
            keys.update({"{} {}".format(key, col) for col in exog.columns})

        assert set(transformed_df.columns) == keys

    def test_model_fit(self):
        data = self.generate_input()
        f = self._get_endog_transform()
        g = self._get_exog_transform()
        y_train = data[self.lbl_value]
        x_train = data.iloc[:, 1:]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(y=y_train, X=x_train)

        parameters = model.get_parameters()

        keys = set(f.keys())
        for key in g.keys():
            keys.update({"{} {}".format(key, col) for col in x_train.columns})

        # Some random test. No good logic here
        assert set(parameters.index) == keys
        assert parameters.isna().sum() == 0
        assert set(model.summary()[model.lbl_params].keys()) == keys

    def test_model_prediction(self):
        data = self.generate_input()
        f = self._get_endog_transform()
        g = self._get_exog_transform()
        num_steps = 5
        y_train = data.loc[data.index[:-num_steps], self.lbl_value]
        x_train = data.iloc[:-num_steps, 1:]
        x_test = data.iloc[-num_steps:, 1:]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(y=y_train, X=x_train)

        forecast = model.predict(num_steps=num_steps, X=x_test)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == self.lbl_value
        assert forecast.isna().sum().sum() == 0
        assert forecast.index.name == self.lbl_date
        assert isinstance(forecast.index, pd.DatetimeIndex)

    def test_model_percentiles(self):
        data = self.generate_input()
        f = self._get_endog_transform()
        g = self._get_exog_transform()

        num_steps = 5
        num_simulations = 5
        quantile_levels = [5, 95]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        y_train = data.loc[data.index[:-num_steps], self.lbl_value]
        x_train = data.iloc[:-num_steps, 1:]
        x_test = data.iloc[-num_steps:, 1:]

        model.fit(y=y_train, X=x_train)

        forecast = model.predict(
            num_steps=num_steps, X=x_test, quantile_levels=quantile_levels, num_simulations=num_simulations
        )

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (num_steps, len(quantile_levels) + 1)
        assert forecast.columns[0] == self.lbl_value
        assert forecast.isna().sum().sum() == 0
        assert forecast.index.name == self.lbl_date
        assert isinstance(forecast.index, pd.DatetimeIndex)

    def test_with_calendar_transform(self):
        data = self.generate_input()
        f = self._get_endog_transform()
        degrees_of_freedom = 4
        lbl_splines = "splines"
        lbl_squared = "squared"
        g = {
            lbl_splines: lambda df: CalendarTransformer().add_periodic_splines(
                df=df, degrees_of_freedom=degrees_of_freedom
            ),
            lbl_squared: lambda df: df ** 2,
        }
        num_steps = 5
        x_train = data.iloc[:-num_steps, 1:]
        y_train = data.loc[data.index[:-num_steps], self.lbl_value]
        x_test = data.iloc[-num_steps:, 1:]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        transformed = model._transform_data(data=x_train, transform=g)
        transformed_df = model._convert_transformed_dict_to_frame(transformed=transformed)
        model.fit(y=y_train, X=x_train)

        forecast = model.predict(num_steps=num_steps, X=x_test)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == self.lbl_value
        assert forecast.isna().sum().sum() == 0
        assert forecast.index.name == self.lbl_date
        assert isinstance(forecast.index, pd.DatetimeIndex)
        assert {lbl_splines, lbl_squared} == set(transformed.keys())
        assert len(transformed[lbl_splines].columns) == x_train.shape[1] + degrees_of_freedom
        assert len(transformed[lbl_squared].columns) == x_train.shape[1]
        assert len(transformed_df.columns) == 2 * x_train.shape[1] + degrees_of_freedom


class TestHCLWeightedTransforms:
    lbl_date = "date"
    lbl_value = "value"

    def generate_input(self):
        nobs = 30
        endog = pd.Series(
            np.arange(1, nobs + 1) + np.random.normal(size=nobs, scale=1e-1),
            name=self.lbl_value,
            index=pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name=self.lbl_date),
        )

        data = add_trend(endog, trend="ct")
        data["x3"] = 999

        f = {
            "lag1": lambda y: y.shift(1),
            "local_mean": lambda y: y.shift(1).ewm(span=5).mean(),
        }

        g = {"const": lambda x: x + 10, "trend": lambda x: -x}

        weights = np.power(np.arange(start=0, stop=1, step=1 / len(endog)), 2)

        return data, f, g, weights

    def test_model_fit(self):
        data, f, g, weights = self.generate_input()

        y_train = data[self.lbl_value]
        x_train = data.iloc[:, 1:]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(y=y_train, X=x_train, weights=weights)

        parameters = model.get_parameters()

        keys = set(f.keys())
        for key in g.keys():
            keys.update({"{} {}".format(key, col) for col in x_train.columns})

        # Some random test. No good logic here
        assert set(parameters.index) == keys
        assert parameters.isna().sum() == 0
        assert set(model.summary()[model.lbl_params].keys()) == keys

    def test_model_prediction(self):
        data, f, g, weights = self.generate_input()

        num_steps = 5
        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        y_train = data.loc[data.index[:-num_steps], self.lbl_value]
        x_train = data.iloc[:-num_steps, 1:]
        x_test = data.iloc[-num_steps:, 1:]
        weights_train = weights[:-num_steps]

        model.fit(y=y_train, X=x_train, weights=weights_train)
        forecast = model.predict(num_steps=num_steps, X=x_test)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == self.lbl_value
        assert forecast.isna().sum().sum() == 0
        assert forecast.index.name == self.lbl_date
        assert isinstance(forecast.index, pd.DatetimeIndex)
