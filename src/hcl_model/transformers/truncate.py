from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

X_TYPE = Union[np.ndarray, pd.Series, pd.DataFrame]


class TruncateTransformer(BaseEstimator, TransformerMixin):
    """Truncate data.

    Leave only a specific number of past weeks.
    """

    def __init__(self, full_weight_obs: int, truncate: bool = True) -> None:
        self.truncate = truncate
        self.full_weight_obs = full_weight_obs

    def fit(self, X: X_TYPE, y: pd.Series = None) -> TruncateTransformer:
        return self

    def transform(self, X: X_TYPE) -> X_TYPE:
        if self.truncate and self.full_weight_obs > 0:
            return X[-self.full_weight_obs :]
        else:
            return X

    @staticmethod
    def inverse_transform(X: X_TYPE) -> X_TYPE:
        return X
