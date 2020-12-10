from abc import abstractmethod
from typing import Tuple

import pandas as pd
import numpy as np
from statsmodels.tsa.tsatools import add_trend

from hcl_model.model_hcl_generic import HandCraftedLinearModel
from hcl_model.model_sarimax import SARIMAXModel


class TestModelCommon:
    @staticmethod
    def generate_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        nobs = 30
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        endog = pd.DataFrame(
            {"value": np.arange(1, nobs + 1) + np.random.normal(size=nobs)}, index=index
        )
        exog = pd.DataFrame(
            {"const": np.ones(nobs), "time": np.arange(1, nobs + 1)}, index=index
        )
        return endog, exog

    @abstractmethod
    def test_model_fit(self):
        pass

    @abstractmethod
    def test_model_prediction(self):
        pass

    @abstractmethod
    def test_model_simulation(self):
        pass

    @abstractmethod
    def test_model_percentiles(self):
        pass

    @abstractmethod
    def test_model_summary(self):
        pass


class TestPredictionsSanity:
    def test_constant(self):
        nobs = 30
        num_steps = 10
        lbl_value = "value"
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, nobs + 1), index=index, name=lbl_value)
        endog = data.iloc[:-num_steps]
        exog = add_trend(data, trend="c", has_constant="add").drop(lbl_value, axis=1)

        expected_forecast = pd.DataFrame(
            {lbl_value: endog.mean()}, index=pd.Index(np.arange(nobs - num_steps, nobs))
        )

        model = SARIMAXModel(trend="c")
        forecast_sarimax = model.predict(endog=endog, num_steps=num_steps)

        model = HandCraftedLinearModel()
        forecast_hcl = model.predict(endog=endog, exog=exog, num_steps=num_steps)

        pd.testing.assert_frame_equal(forecast_sarimax, expected_forecast)
        pd.testing.assert_frame_equal(forecast_hcl, expected_forecast)

    def test_linear_trend(self):
        nobs = 30
        num_steps = 10
        lbl_value = "value"
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(
            np.arange(1, nobs + 1), index=index, name=lbl_value, dtype=float
        )
        endog = data.iloc[:-num_steps]
        exog = add_trend(data, trend="ct", has_constant="add").drop(lbl_value, axis=1)

        expected_forecast = pd.DataFrame(
            {lbl_value: data.iloc[-num_steps:].values},
            index=pd.Index(np.arange(nobs - num_steps, nobs)),
        )

        model = SARIMAXModel(trend="ct")
        model.fit(endog=endog)
        forecast_sarimax = model.predict(endog=endog, num_steps=num_steps)

        model = HandCraftedLinearModel()
        model.fit(endog=endog, exog=exog)
        forecast_hcl = model.predict(endog=endog, exog=exog, num_steps=num_steps)

        pd.testing.assert_frame_equal(forecast_sarimax, expected_forecast)
        pd.testing.assert_frame_equal(forecast_hcl, expected_forecast)

    def test_ar1_with_const(self):
        nobs = 30
        num_steps = 10
        lbl_value = "value"
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(
            np.arange(1, nobs + 1), index=index, name=lbl_value, dtype=float
        )
        endog = data.iloc[:-num_steps]
        exog = add_trend(data, trend="c", has_constant="add").drop(lbl_value, axis=1)
        endog_transform = {"lag1": lambda y: y.shift(1)}

        model = SARIMAXModel(trend="n", order=(1, 0, 0), enforce_stationarity=False)
        model.fit(endog=endog, exog=exog)
        forecast_sarimax = model.predict(endog=endog, exog=exog, num_steps=num_steps)

        model = HandCraftedLinearModel(endog_transform=endog_transform)
        model.fit(endog=endog, exog=exog)
        forecast_hcl = model.predict(endog=endog, exog=exog, num_steps=num_steps)

        pd.testing.assert_frame_equal(forecast_sarimax, forecast_hcl, rtol=1e-1)

    def test_ar1_with_linear_trend(self):
        nobs = 30
        num_steps = 10
        lbl_value = "value"
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(
            np.arange(1, nobs + 1), index=index, name=lbl_value, dtype=float
        )
        endog = data.iloc[:-num_steps]
        exog = add_trend(data, trend="ct", has_constant="add").drop(lbl_value, axis=1)
        endog_transform = {"lag1": lambda y: y.shift(1)}

        model = SARIMAXModel(trend="n", order=(1, 0, 0))
        model.fit(endog=endog, exog=exog)
        forecast_sarimax = model.predict(endog=endog, exog=exog, num_steps=num_steps)

        model = HandCraftedLinearModel(endog_transform=endog_transform)
        model.fit(endog=endog, exog=exog)
        forecast_hcl = model.predict(endog=endog, exog=exog, num_steps=num_steps)

        pd.testing.assert_frame_equal(forecast_sarimax, forecast_hcl, rtol=1e-1)
