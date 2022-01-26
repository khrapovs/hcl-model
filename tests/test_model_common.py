from abc import abstractmethod
from typing import Tuple

import pandas as pd
import numpy as np
from statsmodels.tsa.tsatools import add_trend

from hcl_model.model_hcl_generic import HandCraftedLinearModel
from hcl_model.model_sarimax import SARIMAXModel


class TestModelCommon:
    lbl_date = "date"
    lbl_value = "value"

    def generate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        nobs = 30
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name=self.lbl_date)
        endog = pd.DataFrame({self.lbl_value: np.arange(1, nobs + 1) + np.random.normal(size=nobs)}, index=index)
        exog = pd.DataFrame({"const": np.ones(nobs), "time": np.arange(1, nobs + 1)}, index=index)
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
    lbl_value = "value"

    def test_constant(self):
        nobs = 30
        num_steps = 10

        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, nobs + 1), index=index, name=self.lbl_value)
        exog = add_trend(data, trend="c", has_constant="add").drop(self.lbl_value, axis=1)
        y_train = data.iloc[:-num_steps]
        x_train = exog.iloc[:-num_steps]
        x_test = exog.iloc[-num_steps:]

        expected_forecast = pd.DataFrame({self.lbl_value: y_train.mean()}, index=index[-num_steps:])

        model = SARIMAXModel(trend="c")
        model.fit(y=y_train)
        forecast_sarimax = model.predict(num_steps=num_steps)

        model = HandCraftedLinearModel()
        model.fit(y=y_train, X=x_train)
        forecast_hcl = model.predict(num_steps=num_steps, X=x_test)

        pd.testing.assert_frame_equal(forecast_sarimax, expected_forecast)
        pd.testing.assert_frame_equal(forecast_hcl, expected_forecast)

    def test_linear_trend(self):
        nobs = 30
        num_steps = 10
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, nobs + 1), index=index, name=self.lbl_value, dtype=float)
        exog = add_trend(data, trend="ct", has_constant="add").drop(self.lbl_value, axis=1)
        y_train = data.iloc[:-num_steps]
        x_train = exog.iloc[:-num_steps]
        x_test = exog.iloc[-num_steps:]

        expected_forecast = pd.DataFrame({self.lbl_value: data.iloc[-num_steps:].values}, index=index[-num_steps:])

        model = SARIMAXModel(trend="ct")
        model.fit(y=y_train)
        forecast_sarimax = model.predict(num_steps=num_steps)

        model = HandCraftedLinearModel()
        model.fit(y=y_train, X=x_train)
        forecast_hcl = model.predict(num_steps=num_steps, X=x_test)

        pd.testing.assert_frame_equal(forecast_sarimax, expected_forecast)
        pd.testing.assert_frame_equal(forecast_hcl, expected_forecast)

    def test_ar1_with_const(self):
        nobs = 30
        num_steps = 10
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, nobs + 1), index=index, name=self.lbl_value, dtype=float)
        exog = add_trend(data, trend="c", has_constant="add").drop(self.lbl_value, axis=1)
        y_train = data.iloc[:-num_steps]
        x_train = exog.iloc[:-num_steps]
        x_test = exog.iloc[-num_steps:]
        endog_transform = {"lag1": lambda y: y.shift(1)}

        model = SARIMAXModel(trend="n", order=(1, 0, 0), enforce_stationarity=False)
        model.fit(y=y_train, X=x_train)
        forecast_sarimax = model.predict(num_steps=num_steps, X=x_test)

        model = HandCraftedLinearModel(endog_transform=endog_transform)
        model.fit(y=y_train, X=x_train)
        forecast_hcl = model.predict(num_steps=num_steps, X=x_test)

        pd.testing.assert_frame_equal(forecast_sarimax, forecast_hcl, rtol=1e-1)

    def test_ar1_with_linear_trend(self):
        nobs = 30
        num_steps = 10
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, nobs + 1), index=index, name=self.lbl_value, dtype=float)
        exog = add_trend(data, trend="ct", has_constant="add").drop(self.lbl_value, axis=1)
        y_train = data.iloc[:-num_steps]
        x_train = exog.iloc[:-num_steps]
        x_test = exog.iloc[-num_steps:]
        endog_transform = {"lag1": lambda y: y.shift(1)}

        model = SARIMAXModel(trend="n", order=(1, 0, 0))
        model.fit(y=y_train, X=x_train)
        forecast_sarimax = model.predict(num_steps=num_steps, X=x_test)

        model = HandCraftedLinearModel(endog_transform=endog_transform)
        model.fit(y=y_train, X=x_train)
        forecast_hcl = model.predict(num_steps=num_steps, X=x_test)

        pd.testing.assert_frame_equal(forecast_sarimax, forecast_hcl, rtol=1e-1)
