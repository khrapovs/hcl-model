import pandas as pd
import pytest

from hcl_model.utils.check_x_y import check_X_y


def test_check_X_y() -> None:
    index = pd.date_range(start="2021-01-01", periods=3, freq="D", name="date")
    X_raw = pd.DataFrame({"a": [1, 2, 4]}, index=index)
    y_raw = pd.Series([3, 5, 7], index=index, name="some name")

    X, y = check_X_y(X=X_raw, y=y_raw)
    pd.testing.assert_frame_equal(X, X_raw)
    pd.testing.assert_series_equal(y, y_raw)

    X, y = check_X_y(X=X_raw, y=y_raw.values)
    pd.testing.assert_frame_equal(X, X_raw)
    pd.testing.assert_series_equal(y, y_raw.rename("value"))

    X, y = check_X_y(X=None, y=y_raw)
    assert X is None
    pd.testing.assert_series_equal(y, y_raw)

    error_msg = "Passing ndarray for y and None for X is not supported"
    with pytest.raises(TypeError, match=error_msg):
        check_X_y(X=None, y=y_raw.values)

    error_msg = r"Length of values \(\d\) does not match length of index \(\d\)"
    with pytest.raises(ValueError, match=error_msg):
        check_X_y(X=X_raw, y=y_raw.values[1:])
