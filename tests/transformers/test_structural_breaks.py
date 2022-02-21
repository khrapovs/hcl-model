import random

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from hcl_model.transformers.structural_breaks import TargetStructuralBreakCorrectionTransformer


@pytest.mark.parametrize("y_type", ["series", "ndarray"])
class TestStructuralBreakCorrectionTransformer:
    def test_no_correction_series_with_structural_break(self, y_type: str) -> None:
        series1 = np.ones(100)
        series2 = np.ones(100) + 1
        series = pd.Series(np.append(series1, series2))
        X = series.values if y_type == "ndarray" else series

        transformer = TargetStructuralBreakCorrectionTransformer(structural_break_correction=False)
        series_corrected = transformer.transform(X=X)

        assert len(series_corrected) == len(series)
        if y_type == "ndarray":
            np.array_equal(series.values, series_corrected)
            np.array_equal(transformer.inverse_transform(X=series_corrected), series_corrected)
        else:
            assert_series_equal(series, series_corrected)
            assert_series_equal(transformer.inverse_transform(X=series_corrected), series_corrected)

    def test_correction_series_with_structural_break(self, y_type: str) -> None:
        series1 = np.ones(100)
        series2 = np.ones(100) + 1
        series = pd.Series(np.append(series1, series2))
        X = series.values if y_type == "ndarray" else series

        transformer = TargetStructuralBreakCorrectionTransformer()
        series_corrected = transformer.transform(X=X)
        series_corrected_expected = pd.Series(np.ones(200) + 1)

        assert len(series_corrected) == len(series)
        if y_type == "ndarray":
            np.array_equal(series_corrected_expected.values, series_corrected)
            np.array_equal(transformer.inverse_transform(X=series_corrected), series_corrected)
        else:
            assert_series_equal(series_corrected_expected, series_corrected)
            assert_series_equal(transformer.inverse_transform(X=series_corrected), series_corrected)

    def test_adjusted_variability_still_zero(self, y_type: str) -> None:
        random.seed(101)
        series1 = pd.Series(np.ones(100))
        series2 = pd.Series(np.random.normal(size=100, loc=100))
        series = pd.concat([series1, series2])
        X = series.values if y_type == "ndarray" else series

        series_corrected = TargetStructuralBreakCorrectionTransformer().transform(X=X)
        series_corrected_first_part = series_corrected[:100]

        assert len(np.unique(series_corrected_first_part)) == len(np.unique(series1))

    def test_variability_adjusted(self, y_type: str) -> None:
        random.seed(101)
        series1 = pd.Series(np.random.normal(size=100, loc=1, scale=1))
        series2 = pd.Series(np.random.normal(size=100, loc=100, scale=5))
        series = pd.concat([series1, series2])
        X = series.values if y_type == "ndarray" else series

        series_corrected = TargetStructuralBreakCorrectionTransformer().transform(X=X)
        series_corrected_first_part = series_corrected[:100]

        assert series_corrected_first_part.std() > series1.std()

    def test_last_part_equal_to_original(self, y_type: str) -> None:
        random.seed(101)
        series1 = pd.Series(np.random.normal(size=100))
        series2 = pd.Series(np.random.normal(size=100, loc=100))
        series = pd.concat([series1, series2])
        X = series.values if y_type == "ndarray" else series

        transformer = TargetStructuralBreakCorrectionTransformer()
        series_corrected = transformer.transform(X=X)
        series_corrected_last_part = series_corrected[100:]

        if y_type == "ndarray":
            np.array_equal(series_corrected_last_part, series2)
            np.array_equal(transformer.inverse_transform(X=series_corrected), series_corrected)
        else:
            assert_series_equal(series_corrected_last_part, series2)
            assert_series_equal(transformer.inverse_transform(X=series_corrected), series_corrected)

    def test_correction_series_without_structural_break(self, y_type: str) -> None:
        series = pd.Series(np.ones(200))
        X = series.values if y_type == "ndarray" else series

        transformer = TargetStructuralBreakCorrectionTransformer()
        series_corrected = transformer.transform(X=X)

        assert len(series_corrected) == len(series)
        if y_type == "ndarray":
            np.array_equal(series.values, series_corrected)
            np.array_equal(transformer.inverse_transform(X=series_corrected), series_corrected)
        else:
            assert_series_equal(series, series_corrected)
            assert_series_equal(transformer.inverse_transform(X=series_corrected), series_corrected)
