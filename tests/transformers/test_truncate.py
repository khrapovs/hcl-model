import pandas as pd
import pytest

from hcl_model.transformers.truncate import TruncateTransformer


@pytest.mark.parametrize("full_weight_weeks", [1, 2, 31, 32])
def test_truncate(full_weight_weeks: int) -> None:
    lbl_date = "date"
    lbl_value = "value"
    df = pd.DataFrame({lbl_date: pd.date_range(start="2021-01-01", end="2021-02-01"), lbl_value: range(32)})
    s = df.set_index(lbl_date)[lbl_value]

    s_actual = TruncateTransformer(full_weight_weeks=full_weight_weeks).transform(X=s)
    s_expected = s[s.index > s.index.max() - pd.DateOffset(weeks=full_weight_weeks, days=1)]
    pd.testing.assert_series_equal(s_actual, s_expected)

    s_actual = TruncateTransformer(full_weight_weeks=len(s) + 1).transform(X=s)
    pd.testing.assert_series_equal(s_actual, s)

    s_actual = TruncateTransformer(full_weight_weeks=full_weight_weeks, truncate=False).transform(X=s)
    pd.testing.assert_series_equal(s_actual, s)

    data = df.set_index(lbl_date)
    df_actual = TruncateTransformer(full_weight_weeks=full_weight_weeks).transform(X=data)
    df_expected = data[data.index > data.index.max() - pd.DateOffset(weeks=full_weight_weeks, days=1)]
    pd.testing.assert_frame_equal(df_actual, df_expected)

    df_actual = TruncateTransformer(full_weight_weeks=data.shape[0] + 1).transform(X=data)
    pd.testing.assert_frame_equal(df_actual, data)

    df_actual = TruncateTransformer(full_weight_weeks=full_weight_weeks, truncate=False).transform(X=data)
    pd.testing.assert_frame_equal(df_actual, data)
