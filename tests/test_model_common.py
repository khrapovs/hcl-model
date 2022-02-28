from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest
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
    def test_model_fit(self) -> None:
        pass

    @abstractmethod
    def test_model_prediction(self) -> None:
        pass

    @abstractmethod
    def test_model_simulation(self) -> None:
        pass

    @abstractmethod
    def test_model_percentiles(self) -> None:
        pass

    @abstractmethod
    def test_model_summary(self) -> None:
        pass


@pytest.mark.parametrize("num_steps", [None, 10])
@pytest.mark.parametrize("y_type", ["series", "ndarray"])
class TestPredictionsSanity:
    lbl_value = "value"
    nobs = 30
    split_num_steps = 10

    def test_constant(self, num_steps: Optional[int], y_type: str) -> None:
        index = pd.date_range("2019-01-01", periods=self.nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, self.nobs + 1), index=index, name=self.lbl_value)
        exog = add_trend(data, trend="c", has_constant="add").drop(self.lbl_value, axis=1)
        y_train = data.iloc[: -self.split_num_steps]
        x_train = exog.iloc[: -self.split_num_steps]
        x_test = exog.iloc[-self.split_num_steps :]

        expected_forecast = pd.DataFrame({self.lbl_value: y_train.mean()}, index=index[-self.split_num_steps :])

        model_sarimax = SARIMAXModel(trend="c")
        model_sarimax.fit(y=y_train, X=None)
        with pytest.raises(ValueError, match="Either `num_steps` or `X` should be provided"):
            model_sarimax.predict(num_steps=None, X=None)
        forecast_sarimax = model_sarimax.predict(num_steps=self.split_num_steps, X=None)

        model_hcl = HandCraftedLinearModel()
        model_hcl.fit(y=y_train.values if y_type == "ndarray" else y_train, X=x_train)
        with pytest.raises(ValueError, match="Either `num_steps` or `X` should be provided"):
            model_hcl.predict(num_steps=None, X=None)
        forecast_hcl = model_hcl.predict(num_steps=num_steps, X=x_test)

        pd.testing.assert_frame_equal(forecast_sarimax, expected_forecast)
        pd.testing.assert_frame_equal(forecast_hcl, expected_forecast)

    def test_linear_trend(self, num_steps: Optional[int], y_type: str) -> None:
        index = pd.date_range("2019-01-01", periods=self.nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, self.nobs + 1), index=index, name=self.lbl_value, dtype=float)
        exog = add_trend(data, trend="ct", has_constant="add").drop(self.lbl_value, axis=1)
        y_train = data.iloc[: -self.split_num_steps]
        x_train = exog.iloc[: -self.split_num_steps]
        x_test = exog.iloc[-self.split_num_steps :]

        expected_forecast = pd.DataFrame(
            {self.lbl_value: data.iloc[-self.split_num_steps :].values}, index=index[-self.split_num_steps :]
        )

        model_sarimax = SARIMAXModel(trend="ct")
        model_sarimax.fit(y=y_train, X=None)
        with pytest.raises(ValueError, match="Either `num_steps` or `X` should be provided"):
            model_sarimax.predict(num_steps=None, X=None)
        forecast_sarimax = model_sarimax.predict(num_steps=self.split_num_steps, X=None)

        model_hcl = HandCraftedLinearModel()
        model_hcl.fit(y=y_train.values if y_type == "ndarray" else y_train, X=x_train)
        with pytest.raises(ValueError, match="Either `num_steps` or `X` should be provided"):
            model_hcl.predict(num_steps=None, X=None)
        forecast_hcl = model_hcl.predict(num_steps=num_steps, X=x_test)

        pd.testing.assert_frame_equal(forecast_sarimax, expected_forecast)
        pd.testing.assert_frame_equal(forecast_hcl, expected_forecast)

    def test_ar1_with_const(self, num_steps: Optional[int], y_type: str) -> None:
        nobs = 30
        index = pd.date_range("2019-01-01", periods=nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, nobs + 1), index=index, name=self.lbl_value, dtype=float)
        exog = add_trend(data, trend="c", has_constant="add").drop(self.lbl_value, axis=1)
        y_train = data.iloc[: -self.split_num_steps]
        x_train = exog.iloc[: -self.split_num_steps]
        x_test = exog.iloc[-self.split_num_steps :]
        endog_transform = {"lag1": lambda y: y.shift(1)}

        model_sarimax = SARIMAXModel(trend="n", order=(1, 0, 0), enforce_stationarity=False)
        model_sarimax.fit(y=y_train.values if y_type == "ndarray" else y_train, X=x_train)
        with pytest.raises(ValueError, match="Either `num_steps` or `X` should be provided"):
            model_sarimax.predict(num_steps=None, X=None)
        forecast_sarimax = model_sarimax.predict(num_steps=num_steps, X=x_test)

        model_hcl = HandCraftedLinearModel(endog_transform=endog_transform)
        model_hcl.fit(y=y_train.values if y_type == "ndarray" else y_train, X=x_train)
        with pytest.raises(ValueError, match="Either `num_steps` or `X` should be provided"):
            model_hcl.predict(num_steps=None, X=None)
        forecast_hcl = model_hcl.predict(num_steps=num_steps, X=x_test)

        pd.testing.assert_frame_equal(forecast_sarimax, forecast_hcl, rtol=1e-1)

    def test_ar1_with_linear_trend(self, num_steps: Optional[int], y_type: str) -> None:
        index = pd.date_range("2019-01-01", periods=self.nobs, freq="W-FRI", name="date")
        data = pd.Series(np.arange(1, self.nobs + 1), index=index, name=self.lbl_value, dtype=float)
        exog = add_trend(data, trend="ct", has_constant="add").drop(self.lbl_value, axis=1)
        y_train = data.iloc[: -self.split_num_steps]
        x_train = exog.iloc[: -self.split_num_steps]
        x_test = exog.iloc[-self.split_num_steps :]
        endog_transform = {"lag1": lambda y: y.shift(1)}

        model_sarimax = SARIMAXModel(trend="n", order=(1, 0, 0))
        model_sarimax.fit(y=y_train.values if y_type == "ndarray" else y_train, X=x_train)
        with pytest.raises(ValueError, match="Either `num_steps` or `X` should be provided"):
            model_sarimax.predict(num_steps=None, X=None)
        forecast_sarimax = model_sarimax.predict(num_steps=num_steps, X=x_test)

        model_hcl = HandCraftedLinearModel(endog_transform=endog_transform)
        model_hcl.fit(y=y_train.values if y_type == "ndarray" else y_train, X=x_train)
        with pytest.raises(ValueError, match="Either `num_steps` or `X` should be provided"):
            model_hcl.predict(num_steps=None, X=None)
        forecast_hcl = model_hcl.predict(num_steps=num_steps, X=x_test)

        pd.testing.assert_frame_equal(forecast_sarimax, forecast_hcl, rtol=1e-1)
