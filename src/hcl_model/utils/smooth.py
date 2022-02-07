import pandas as pd


def smooth_series(y: pd.Series, window: int, quantile: float, ewm_alpha: float) -> pd.Series:
    return y.rolling(window=window).quantile(quantile=quantile).fillna(method="bfill").ewm(alpha=ewm_alpha).mean()
