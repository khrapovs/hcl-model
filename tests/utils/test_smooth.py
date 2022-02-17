import numpy as np
import pandas as pd
import pytest

from hcl_model.utils.smooth import smooth_series


def test_smooth_series() -> None:
    y_original = pd.Series(range(50))
    y_smoothed = smooth_series(y=y_original, window=10, quantile=0.5, ewm_alpha=0.1)
    assert len(y_original) == len(y_smoothed)

    y_original = pd.Series(np.zeros(100))
    y_smoothed = smooth_series(y=y_original, window=10, quantile=0.5, ewm_alpha=0.1)
    pd.testing.assert_series_equal(y_original, y_smoothed)

    y_original = pd.Series([0, 1] * 50)
    y_smoothed = smooth_series(y=y_original, window=10, quantile=0.25, ewm_alpha=0.1)
    assert sum(y_smoothed == 0) == len(y_smoothed)
    assert sum(y_smoothed == 1) == 0

    y_smoothed = smooth_series(y=y_original, window=10, quantile=0.5, ewm_alpha=0.1)
    assert sum(y_smoothed < y_original) == sum(y_original == 1)
    assert sum(y_smoothed > y_original) == sum(y_original == 0)

    y_smoothed = smooth_series(y=y_original, window=10, quantile=0.75, ewm_alpha=0.1)
    assert sum(y_smoothed == 1) == len(y_smoothed)
    assert sum(y_smoothed == 0) == 0

    with pytest.raises(ValueError):
        smooth_series(y=y_original, window=-10, quantile=0.5, ewm_alpha=0.1)

    with pytest.raises(ValueError):
        smooth_series(y=y_original, window=10, quantile=-0.5, ewm_alpha=0.1)

    with pytest.raises(ValueError):
        smooth_series(y=y_original, window=10, quantile=0.5, ewm_alpha=-0.1)
