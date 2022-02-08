from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SeriesToFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.Series, y: pd.Series = None) -> SeriesToFrameTransformer:
        return self

    @staticmethod
    def transform(X: pd.Series) -> pd.DataFrame:
        return X.to_frame()
