from typing import Union

import numpy as np
import pandas as pd


def smooth_series(y: Union[pd.Series, np.ndarray], window: int, quantile: float, ewm_alpha: float) -> pd.Series:
    return (
        (pd.Series(y.flatten()) if isinstance(y, np.ndarray) else y)
        .rolling(window=window)
        .quantile(quantile=quantile)
        .fillna(method="bfill")
        .ewm(alpha=ewm_alpha)
        .mean()
    )
