from __future__ import annotations

import pandas as pd
import pytest
from pandas._testing import assert_series_equal
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from hcl_model.transformers.estimator_to_transformer import EstimatorToTransformer


class ModelForTests(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.indicator = "A"

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> ModelForTests:
        self.indicator = "B"
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series([1.0, 5.0])


def test_estimator_to_transformer() -> None:
    custom_model = ModelForTests()

    assert custom_model.indicator == "A"

    sklearn_model = EstimatorToTransformer(model=custom_model)
    x_data = pd.DataFrame({"x": [1.0, 3.0]})

    with pytest.raises(NotFittedError, match="instance is not fitted yet"):
        sklearn_model.transform(X=x_data)

    sklearn_model.fit(X=x_data, y=pd.Series([4.0]))

    assert custom_model.indicator == "B"

    assert_series_equal(sklearn_model.transform(X=x_data), pd.Series([1.0, 5.0]))
