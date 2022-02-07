from __future__ import annotations

from typing import Union

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from sklearn.base import BaseEstimator, TransformerMixin

from hcl_model.transformers.feature_union import make_union_of_datetime_indexed_features


class TransformerForTests(BaseEstimator, TransformerMixin):
    def __init__(self, dates: Union[list[int], pd.DatetimeIndex], cols: Union[str, list[str]]) -> None:
        self._dates = dates
        self._cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> TransformerForTests:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[self._dates, self._cols]


def test_datetime_indexed_feature_union() -> None:
    date_index = pd.date_range(start="2021-01-01", periods=4)
    df = pd.DataFrame({"a": [1, 3, 6, 9], "b": [2, 5, 7, 10]}, index=date_index)
    feature_transformers = [
        TransformerForTests(dates=date_index, cols=["a"]),
        TransformerForTests(dates=date_index, cols=["b"]),
    ]
    feature_union = make_union_of_datetime_indexed_features(*feature_transformers)
    assert_frame_equal(feature_union.transform(X=df), df)
    assert_frame_equal(feature_union.fit_transform(X=df), df)

    feature_transformers = [
        TransformerForTests(dates=date_index[[0, 2]], cols=["a"]),
        TransformerForTests(dates=date_index[[1, 3]], cols=["b"]),
    ]
    feature_union = make_union_of_datetime_indexed_features(*feature_transformers)
    assert_frame_equal(feature_union.transform(X=df), df.loc[[]].set_index(pd.Index([], dtype="object")))
    assert_frame_equal(feature_union.fit_transform(X=df), df.loc[[]].set_index(pd.Index([], dtype="object")))

    feature_transformers = [
        TransformerForTests(dates=date_index[[0, 2]], cols=["a"]),
        TransformerForTests(dates=date_index, cols=["b"]),
    ]
    feature_union = make_union_of_datetime_indexed_features(*feature_transformers)
    assert_frame_equal(feature_union.transform(X=df), df.loc[date_index[[0, 2]]])
    assert_frame_equal(feature_union.fit_transform(X=df), df.loc[date_index[[0, 2]]])

    feature_transformers = [
        TransformerForTests(dates=date_index[[0, 2]], cols="a"),
        TransformerForTests(dates=date_index, cols=["b"]),
    ]
    feature_union = make_union_of_datetime_indexed_features(*feature_transformers)
    assert_frame_equal(feature_union.transform(X=df), df.loc[date_index[[0, 2]]])
    assert_frame_equal(feature_union.fit_transform(X=df), df.loc[date_index[[0, 2]]])

    feature_transformers = [
        TransformerForTests(dates=date_index[[0, 2]], cols="a"),
        TransformerForTests(dates=date_index, cols="b"),
    ]
    feature_union = make_union_of_datetime_indexed_features(*feature_transformers)
    assert_frame_equal(feature_union.transform(X=df), df.loc[date_index[[0, 2]]])
    assert_frame_equal(feature_union.fit_transform(X=df), df.loc[date_index[[0, 2]]])

    feature_transformers = [
        TransformerForTests(dates=[0, 2], cols="a"),
        TransformerForTests(dates=[0, 1, 2, 3], cols="b"),
    ]
    feature_union = make_union_of_datetime_indexed_features(*feature_transformers)
    error_msg = "All features should be in datetime-indexed Series or DataFrames"
    with pytest.raises(ValueError, match=error_msg):
        feature_union.transform(X=df.reset_index(drop=True))
    with pytest.raises(ValueError, match=error_msg):
        feature_union.fit_transform(X=df.reset_index(drop=True))
