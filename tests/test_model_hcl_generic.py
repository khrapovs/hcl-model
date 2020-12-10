import functools

import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import add_trend

from hcl_model.model_hcl_generic import HandCraftedLinearModel
from tests.test_model_common import TestModelCommon


class TestHCL(TestModelCommon):
    def test_model_fit(self):
        endog, exog = self.generate_data()

        model = HandCraftedLinearModel()
        model.fit(endog=endog["value"], exog=exog)
        parameters = model._get_parameters()

        params_expected = [
            "{} const".format(model.lbl_original_exog),
            "{} time".format(model.lbl_original_exog),
        ]
        assert list(parameters.index) == params_expected
        assert set(model.summary()[model.lbl_params].keys()) == set(params_expected)

    def test_model_prediction(self):
        endog, exog = self.generate_data()
        model = HandCraftedLinearModel()
        num_steps = 10
        lbl_value = "value"
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

        model.fit(endog=endog.loc[endog.index[:-num_steps], "value"], exog=exog)
        simulations = model.simulate(
            num_steps=num_steps, num_simulations=num_simulations
        )

        assert isinstance(simulations, pd.DataFrame)
        assert simulations.shape == (num_steps, num_simulations)

    def test_model_percentiles(self):
        endog, exog = self.generate_data()
        model = HandCraftedLinearModel()
        num_steps = 10
        num_simulations = 5
        quantile_levels = [5, 95]
        lbl_value = "value"

        model.fit(endog=endog.loc[endog.index[:-num_steps], lbl_value], exog=exog)
        forecast = model.predict(
            num_steps=num_steps,
            quantile_levels=quantile_levels,
            num_simulations=num_simulations,
        )

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (num_steps, len(quantile_levels) + 1)
        assert forecast.columns[0] == lbl_value

    def test_model_summary(self):
        endog, exog = self.generate_data()

        model = HandCraftedLinearModel()
        model.fit(endog=endog["value"], exog=exog)

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
    @staticmethod
    def generate_input():
        nobs = 30
        endog = pd.Series(
            np.arange(1, nobs + 1) + np.random.normal(size=nobs, scale=1e-1),
            name="value",
            index=pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date"),
        )

        data = endog.to_frame()
        data["x2"] = np.random.normal(size=nobs)
        data["x3"] = np.random.normal(size=nobs)

        f = {
            "lag1": lambda y: y.shift(1),
            "local_mean": lambda y: y.shift(1).ewm(span=5).mean(),
        }

        g = {"const": lambda x: x + 10, "trend": lambda x: -x.shift(2)}

        return data, f, g

    def test_transform_lags(self):
        lags = [1, 2, 10]
        col_name = "lag_{}"

        f = {
            col_name.format(lag): functools.partial(lambda lag, y: y.shift(lag), lag)
            for lag in lags
        }
        model = HandCraftedLinearModel(endog_transform=f)
        endog = pd.Series(np.arange(5))

        transformed = model._transform_data(data=endog, transform=f)
        transformed_expected = {col_name.format(lag): endog.shift(lag) for lag in lags}

        for key, val in transformed.items():
            pd.testing.assert_series_equal(val, transformed_expected[key])

    def test_transform_data(self):
        data, f, g = self.generate_input()

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        endog = data["value"]
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
        data, f, g = self.generate_input()

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        endog = data["value"]
        exog = data.iloc[:, 1:]

        transformed = model._transform_all_data(exog=exog, endog=endog)
        transformed_df = model._convert_transformed_dict_to_frame(
            transformed=transformed
        )

        keys = set(f.keys())
        for key in g.keys():
            keys.update({"{} {}".format(key, col) for col in exog.columns})

        assert set(transformed_df.columns) == keys

    def test_model_fit(self):
        data, f, g = self.generate_input()
        endog = data["value"]
        exog = data.iloc[:, 1:]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(endog=endog, exog=exog)

        parameters = model._get_parameters()

        keys = set(f.keys())
        for key in g.keys():
            keys.update({"{} {}".format(key, col) for col in exog.columns})

        # Some random test. No good logic here
        assert set(parameters.index) == keys
        assert parameters.isna().sum() == 0
        assert set(model.summary()[model.lbl_params].keys()) == keys

    def test_model_prediction(self):
        data, f, g = self.generate_input()

        num_steps = 5
        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(
            endog=data.loc[data.index[:-num_steps], "value"],
            exog=data.iloc[:-num_steps, 1:],
        )

        forecast = model.predict(exog=data.iloc[:, 1:], num_steps=num_steps)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == "value"
        assert forecast.isna().sum().sum() == 0

    def test_model_percentiles(self):
        data, f, g = self.generate_input()

        num_steps = 5
        num_simulations = 5
        quantile_levels = [5, 95]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        endog = data.loc[data.index[:-num_steps], "value"]
        exog = data.iloc[:, 1:]
        model.fit(endog=endog, exog=exog)

        forecast = model.predict(
            exog=exog,
            num_steps=num_steps,
            quantile_levels=quantile_levels,
            num_simulations=num_simulations,
        )

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (num_steps, len(quantile_levels) + 1)
        assert forecast.columns[0] == "value"
        assert forecast.isna().sum().sum() == 0


class TestHCLWeightedTransforms:
    @staticmethod
    def generate_input():
        nobs = 30
        endog = pd.Series(
            np.arange(1, nobs + 1) + np.random.normal(size=nobs, scale=1e-1),
            name="value",
            index=pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date"),
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

        endog = data["value"]
        exog = data.iloc[:, 1:]

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(endog=endog, exog=exog, weights=weights)

        parameters = model._get_parameters()

        keys = set(f.keys())
        for key in g.keys():
            keys.update({"{} {}".format(key, col) for col in exog.columns})

        # Some random test. No good logic here
        assert set(parameters.index) == keys
        assert parameters.isna().sum() == 0
        assert set(model.summary()[model.lbl_params].keys()) == keys

    def test_model_prediction(self):
        data, f, g, weights = self.generate_input()

        num_steps = 5
        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)
        model.fit(
            endog=data.loc[data.index[:-num_steps], "value"],
            exog=data.iloc[:-num_steps, 1:],
            weights=weights[:-num_steps],
        )

        forecast = model.predict(exog=data.iloc[:, 1:], num_steps=num_steps)

        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape[0] == num_steps
        assert forecast.columns[0] == "value"
        assert forecast.isna().sum().sum() == 0
