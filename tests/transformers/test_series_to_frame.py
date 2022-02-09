import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from hcl_model.transformers.series_to_frame import SeriesToFrameTransformer


def test_series_to_frame() -> None:
    df = SeriesToFrameTransformer().transform(X=pd.Series(dtype=float))
    assert_frame_equal(df, pd.DataFrame(columns=[0], dtype=np.float64))
    df = SeriesToFrameTransformer().transform(X=pd.Series(np.arange(5), name="value"))
    assert_frame_equal(df, pd.DataFrame({"value": np.arange(5)}))
    index = pd.date_range(start="2022-01-01", periods=5, name="date")
    df = SeriesToFrameTransformer().transform(X=pd.Series(np.arange(5), index=index, name="value"))
    assert_frame_equal(df, pd.DataFrame({"value": np.arange(5)}, index=index))
