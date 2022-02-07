import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from hcl_model.transformers.outlier_correction import TargetOutlierCorrectionTransformer, CorrectOutliersMethodNames


class TestOutliers:
    def test_outliers(self) -> None:
        periods = 10
        index = pd.date_range(start="2021-01-01", periods=periods)
        series = pd.Series(0, index=index)
        too_low_idx = [1, 2, 4]
        too_high_idx = [6, 7, 9]
        series.iloc[too_high_idx] = 10
        series.iloc[too_low_idx] = -10

        outliers_detector = TargetOutlierCorrectionTransformer(
            outlier_correction_method=CorrectOutliersMethodNames.trim,
            six_sigma_multiplier=1e-10,
            smoothing_window=periods,
            ewm_alpha=1 / periods,
        )
        too_low_actual = outliers_detector.get_idx_too_low(series=series)
        too_low_expected = pd.Series(False, index=index)
        too_low_expected.iloc[too_low_idx] = True
        too_high_actual = outliers_detector.get_idx_too_high(series=series)
        too_high_expected = pd.Series(False, index=index)
        too_high_expected.iloc[too_high_idx] = True

        pd.testing.assert_series_equal(too_low_actual, too_low_expected)
        pd.testing.assert_series_equal(too_high_actual, too_high_expected)

        outliers_detector = TargetOutlierCorrectionTransformer(
            outlier_correction_method=CorrectOutliersMethodNames.trim,
            six_sigma_multiplier=1e-10,
            smoothing_window=1,
            ewm_alpha=1,
        )
        too_low_actual = outliers_detector.get_idx_too_low(series=series)
        too_low_expected = pd.Series(False, index=index)
        too_high_actual = outliers_detector.get_idx_too_high(series=series)
        too_high_expected = pd.Series(False, index=index)

        pd.testing.assert_series_equal(too_low_actual, too_low_expected)
        pd.testing.assert_series_equal(too_high_actual, too_high_expected)

        outliers_detector = TargetOutlierCorrectionTransformer(
            outlier_correction_method=CorrectOutliersMethodNames.trim,
            six_sigma_multiplier=1e10,
            smoothing_window=periods,
            ewm_alpha=1 / periods,
        )
        too_low_actual = outliers_detector.get_idx_too_low(series=series)
        too_low_expected = pd.Series(False, index=index)
        too_high_actual = outliers_detector.get_idx_too_high(series=series)
        too_high_expected = pd.Series(False, index=index)

        pd.testing.assert_series_equal(too_low_actual, too_low_expected)
        pd.testing.assert_series_equal(too_high_actual, too_high_expected)

    @pytest.mark.parametrize("six_sigma_multiplier", [3, 4, 5])
    def test_get_series_with_corrected_outliers_with_no_correction(self, six_sigma_multiplier: int) -> None:
        date_col = "date"
        value_col = "value"
        size = 300
        values = np.zeros(size)
        values[10] = 40
        values[50] = 80
        values[150] = 60
        df = pd.DataFrame(
            {date_col: pd.date_range(start="2020-12-29", freq="W-SAT", periods=size), value_col: values}
        ).set_index(date_col)
        y = pd.Series(df[value_col], index=df.index, name=value_col)
        outliers_detector = TargetOutlierCorrectionTransformer(
            outlier_correction_method=CorrectOutliersMethodNames.nothing, six_sigma_multiplier=six_sigma_multiplier
        )
        y_corrected = outliers_detector.transform(X=y)
        assert_series_equal(y, y_corrected)

    @pytest.mark.parametrize("six_sigma_multiplier", [3, 4, 5])
    @pytest.mark.parametrize(
        "outlier_correction_method", [CorrectOutliersMethodNames.trim, CorrectOutliersMethodNames.interpolate]
    )
    def test_get_series_with_corrected_outliers(
        self, six_sigma_multiplier: int, outlier_correction_method: CorrectOutliersMethodNames
    ) -> None:
        date_col = "date"
        value_col = "value"
        size = 300
        values = np.zeros(size)
        values[10] = 40
        values[50] = 80
        values[150] = 60
        df = pd.DataFrame(
            {date_col: pd.date_range(start="2020-12-29", freq="W-SAT", periods=size), value_col: values}
        ).set_index(date_col)
        y = pd.Series(df[value_col], index=df.index, name=value_col)
        outliers_detector = TargetOutlierCorrectionTransformer(
            outlier_correction_method=outlier_correction_method, six_sigma_multiplier=six_sigma_multiplier
        )
        y_corrected = outliers_detector.transform(X=y)
        expected_nr_outliers_found = 3
        nr_outliers_found = sum(y != y_corrected)

        assert y.name == y_corrected.name
        assert sum(abs(y_corrected)) <= sum(abs(y))
        assert nr_outliers_found == expected_nr_outliers_found

    @pytest.mark.parametrize("six_sigma_multiplier", [3, 4, 5])
    @pytest.mark.parametrize(
        "outlier_correction_method", [CorrectOutliersMethodNames.trim, CorrectOutliersMethodNames.interpolate]
    )
    def test_get_series_with_corrected_outliers_constant_value_no_outliers(
        self, six_sigma_multiplier: int, outlier_correction_method: CorrectOutliersMethodNames
    ) -> None:
        date_col = "date"
        value_col = "value"
        size = 300
        values = np.zeros(size)
        df = pd.DataFrame(
            {date_col: pd.date_range(start="2020-12-29", freq="W-SAT", periods=size), value_col: values}
        ).set_index(date_col)
        y = pd.Series(df[value_col], index=df.index, name=value_col)
        outliers_detector = TargetOutlierCorrectionTransformer(
            outlier_correction_method=outlier_correction_method, six_sigma_multiplier=six_sigma_multiplier
        )
        y_corrected = outliers_detector.transform(X=y)

        assert_series_equal(y, y_corrected)

    @pytest.mark.parametrize("six_sigma_multiplier", [3, 4, 5])
    @pytest.mark.parametrize(
        "outlier_correction_method", [CorrectOutliersMethodNames.trim, CorrectOutliersMethodNames.interpolate]
    )
    def test_get_series_with_corrected_outliers_no_outliers(
        self, six_sigma_multiplier: int, outlier_correction_method: CorrectOutliersMethodNames
    ) -> None:
        date_col = "date"
        value_col = "value"
        size = 300
        values = np.random.normal(size=size)
        df = pd.DataFrame(
            {date_col: pd.date_range(start="2020-12-29", freq="W-SAT", periods=size), value_col: values}
        ).set_index(date_col)
        y = pd.Series(df[value_col], index=df.index, name=value_col)
        outliers_detector = TargetOutlierCorrectionTransformer(
            outlier_correction_method=outlier_correction_method, six_sigma_multiplier=six_sigma_multiplier
        )
        y_corrected = outliers_detector.transform(X=y)

        assert_series_equal(y, y_corrected)

    def test_get_series_with_outliers_and_with_wrong_method(self) -> None:
        date_col = "date"
        value_col = "value"
        size = 300
        values = np.random.normal(size=size)
        values[10] = 40
        values[50] = 80
        values[150] = 60
        df = pd.DataFrame(
            {date_col: pd.date_range(start="2020-12-29", freq="W-SAT", periods=size), value_col: values}
        ).set_index(date_col)
        y = pd.Series(df[value_col], index=df.index, name=value_col)

        with pytest.raises(NotImplementedError):
            outliers_detector = TargetOutlierCorrectionTransformer(
                outlier_correction_method="magic", six_sigma_multiplier=3
            )
            outliers_detector.transform(X=y)
