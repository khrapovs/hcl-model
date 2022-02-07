from functools import reduce
from typing import Union, List

import pandas as pd
from sklearn.pipeline import FeatureUnion, _name_estimators


class DatetimeIndexedFeatureUnion(FeatureUnion):
    """A replacement for FeatureUnion from sklearn.

    Standard sklearn FeatureUnion concatenates features as numpy arrays.
    Here features are expected to be pd.Series or pd.DataFrame,
    and instead of concatenation, they are merged on index.
    """

    def _hstack(self, Xs: List[Union[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
        if not all([isinstance(df.index, pd.DatetimeIndex) for df in Xs]):
            raise ValueError("All features should be in datetime-indexed Series or DataFrames")
        return reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), Xs)


def make_union_of_datetime_indexed_features(*transformers, n_jobs=None, verbose=False):
    """Construct a DatetimeIndexedFeatureUnion from the given transformers.

    Simple reimplementation of `sklearn.pipeline.make_union`
    """
    return DatetimeIndexedFeatureUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
