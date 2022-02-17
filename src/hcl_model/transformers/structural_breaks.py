from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.base import BaseEstimator, TransformerMixin

X_TYPE = Union[pd.Series, np.ndarray]


class TargetStructuralBreakCorrectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, structural_break_correction: bool = True) -> None:
        self.structural_break_correction = structural_break_correction

    def fit(self, X: X_TYPE, y: pd.Series = None) -> TargetStructuralBreakCorrectionTransformer:
        return self

    def transform(self, X: X_TYPE) -> X_TYPE:
        if self.structural_break_correction:
            if isinstance(X, np.ndarray):
                return self._get_series_without_structural_breaks(signal=pd.Series(X.flatten())).values
            else:
                return self._get_series_without_structural_breaks(signal=X)
        else:
            return X

    def _get_series_without_structural_breaks(self, signal: X_TYPE) -> pd.Series:
        change_points = self._get_change_points(y=signal)
        if len(change_points) <= 1:
            return signal
        else:
            change_points = np.concatenate(([0], change_points))
            current_signal = signal[change_points[-2] : change_points[-1]]
            level_current = current_signal.median()
            variability_current = current_signal.std()
            for past, current in zip(change_points[:-1], change_points[1:]):
                variability_past = signal[past:current].std()
                signal[past:current] = self._adjust_level(y=signal[past:current], level_current=level_current)
                signal[past:current] = self._adjust_variability(
                    y=signal[past:current], variability_current=variability_current, variability_past=variability_past
                )
            return signal

    @staticmethod
    def _adjust_variability(y: X_TYPE, variability_current: float, variability_past: float) -> pd.Series:
        if variability_past == 0:
            return y
        else:
            adjust_factor = variability_current / variability_past
            return (y - y.mean()) * adjust_factor + y.mean()

    @staticmethod
    def _adjust_level(y: X_TYPE, level_current: float) -> pd.Series:
        return y + level_current - y.median()

    @staticmethod
    def _get_change_points(y: X_TYPE) -> np.array:
        values = y.values if isinstance(y, pd.Series) else y
        return np.array(rpt.KernelCPD(kernel="rbf", jump=1, min_size=26).fit(values).predict(pen=10))
