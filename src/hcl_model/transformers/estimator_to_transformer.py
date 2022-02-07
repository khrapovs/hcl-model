from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class EstimatorToTransformer(BaseEstimator, TransformerMixin):
    """Convert sklearn estimator to transformer.

    Effectively, it replaces `predict` method with `transform`,
    so that the object is usable as an intermediate step in `Pipeline`.

    :param model: object with fit and predict methods
    """

    def __init__(self, model: BaseEstimator) -> None:
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> EstimatorToTransformer:
        self.model.fit(X=X, y=y)
        self.model_ = self.model
        return self

    def transform(self, X: pd.DataFrame) -> Optional[pd.Series]:
        check_is_fitted(self)
        return self.model_.predict(X=X)
