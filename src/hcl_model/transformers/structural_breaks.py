from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.base import BaseEstimator, TransformerMixin


class TargetStructuralBreakCorrectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, structural_break_correction: bool = True) -> None:
        self._structural_break_correction = structural_break_correction

    def fit(self, X: pd.Series, y: pd.Series = None) -> TargetStructuralBreakCorrectionTransformer:
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        if self._structural_break_correction:
            return self._get_series_without_structural_breaks(signal=X)
        else:
            return X

    def _get_series_without_structural_breaks(self, signal: pd.Series) -> pd.Series:
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
    def _adjust_variability(y: pd.Series, variability_current: float, variability_past: float) -> pd.Series:
        if variability_past == 0:
            return y
        else:
            adjust_factor = variability_current / variability_past
            return (y - y.mean()) * adjust_factor + y.mean()

    @staticmethod
    def _adjust_level(y: pd.Series, level_current: float) -> pd.Series:
        return y + level_current - y.median()

    @staticmethod
    def _get_change_points(y: pd.Series) -> np.array:
        return np.array(rpt.KernelCPD(kernel="rbf", jump=1, min_size=26).fit(y.values).predict(pen=10))
