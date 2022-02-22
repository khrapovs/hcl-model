import pandas as pd
import pytest
from pandas._testing import assert_series_equal, assert_frame_equal

from hcl_model.transformers.truncate import TruncateTransformer


@pytest.mark.parametrize("full_weight_weeks", [1, 2, 31, 32])
def test_truncate(full_weight_weeks: int) -> None:
    lbl_date = "date"
    lbl_value = "value"
    df = pd.DataFrame({lbl_date: pd.date_range(start="2021-01-01", end="2021-02-01"), lbl_value: range(32)})
    s = df.set_index(lbl_date)[lbl_value]

    s_expected = s[s.index > s.index.max() - pd.DateOffset(weeks=full_weight_weeks, days=1)]
    full_weight_obs = len(s_expected)
    transformer = TruncateTransformer(full_weight_obs=full_weight_obs)
    s_actual = transformer.transform(X=s)
    assert_series_equal(s_actual, s_expected)
    assert_series_equal(transformer.inverse_transform(X=s_actual), s_actual)

    transformer = TruncateTransformer(full_weight_obs=len(s) + 1)
    s_actual = transformer.transform(X=s)
    assert_series_equal(s_actual, s)
    assert_series_equal(transformer.inverse_transform(X=s_actual), s_actual)

    transformer = TruncateTransformer(full_weight_obs=full_weight_weeks, truncate=False)
    s_actual = transformer.transform(X=s)
    assert_series_equal(s_actual, s)
    assert_series_equal(transformer.inverse_transform(X=s_actual), s_actual)

    data = df.set_index(lbl_date)
    df_expected = data[data.index > data.index.max() - pd.DateOffset(weeks=full_weight_weeks, days=1)]
    full_weight_obs = len(df_expected)
    transformer = TruncateTransformer(full_weight_obs=full_weight_obs)
    df_actual = transformer.transform(X=data)
    assert_frame_equal(df_actual, df_expected)
    assert_frame_equal(transformer.inverse_transform(X=df_actual), df_actual)

    transformer = TruncateTransformer(full_weight_obs=data.shape[0] + 1)
    df_actual = transformer.transform(X=data)
    assert_frame_equal(df_actual, data)
    assert_frame_equal(transformer.inverse_transform(X=df_actual), df_actual)

    transformer = TruncateTransformer(full_weight_obs=full_weight_weeks, truncate=False)
    df_actual = transformer.transform(X=data)
    assert_frame_equal(df_actual, data)
    assert_frame_equal(transformer.inverse_transform(X=df_actual), df_actual)
