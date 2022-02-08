import random

import numpy as np
import pandas as pd
from pandas._testing import assert_series_equal

from hcl_model.transformers.structural_breaks import TargetStructuralBreakCorrectionTransformer


class TestStructuralBreakCorrectionTransformer:
    def test_no_correction_series_with_structural_break(self) -> None:
        series1 = np.ones(100)
        series2 = np.ones(100) + 1
        series = pd.Series(np.append(series1, series2))

        series_corrected = TargetStructuralBreakCorrectionTransformer(structural_break_correction=False).transform(
            X=series
        )

        assert len(series_corrected) == len(series)
        assert_series_equal(series, series_corrected)

    def test_correction_series_with_structural_break(self) -> None:
        series1 = np.ones(100)
        series2 = np.ones(100) + 1
        series = pd.Series(np.append(series1, series2))

        series_corrected = TargetStructuralBreakCorrectionTransformer().transform(X=series)
        series_corrected_expected = pd.Series(np.ones(200) + 1)

        assert len(series_corrected) == len(series)
        assert_series_equal(series_corrected_expected, series_corrected)

    def test_adjusted_variability_still_zero(self) -> None:
        random.seed(101)
        series1 = pd.Series(np.ones(100))
        series2 = pd.Series(np.random.normal(size=100, loc=100))
        series = pd.concat([series1, series2])

        series_corrected = TargetStructuralBreakCorrectionTransformer().transform(X=series)
        series_corrected_first_part = series_corrected[:100]

        assert len(series_corrected_first_part.unique()) == len(series1.unique())

    def test_variability_adjusted(self) -> None:
        random.seed(101)
        series1 = pd.Series(np.random.normal(size=100, loc=1, scale=1))
        series2 = pd.Series(np.random.normal(size=100, loc=100, scale=5))
        series = pd.concat([series1, series2])

        series_corrected = TargetStructuralBreakCorrectionTransformer().transform(X=series)
        series_corrected_first_part = series_corrected[:100]

        assert series_corrected_first_part.std() > series1.std()

    def test_last_part_equal_to_original(self) -> None:
        random.seed(101)
        series1 = pd.Series(np.random.normal(size=100))
        series2 = pd.Series(np.random.normal(size=100, loc=100))
        series = pd.concat([series1, series2])

        series_corrected = TargetStructuralBreakCorrectionTransformer().transform(X=series)
        series_corrected_last_part = series_corrected[100:]

        assert_series_equal(series_corrected_last_part, series2)

    def test_correction_series_without_structural_break(self) -> None:
        series = pd.Series(np.ones(200))

        series_corrected = TargetStructuralBreakCorrectionTransformer().transform(X=series)

        assert len(series_corrected) == len(series)
        assert_series_equal(series, series_corrected)
