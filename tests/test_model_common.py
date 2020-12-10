from abc import abstractmethod
from typing import Tuple

import pandas as pd
import numpy as np


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
