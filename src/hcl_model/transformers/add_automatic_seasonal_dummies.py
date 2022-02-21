from __future__ import annotations

import pandas as pd
from scipy.stats import median_abs_deviation


class AddAutomaticSeasonalDummies:
    lbl_auto_dummy = "automatic_dummy_{}"

    def __init__(self, var_name: str, lim_num_dummies: int = 5, threshold: float = 3) -> None:
        self.var_name = var_name
        self.lim_num_dummies = lim_num_dummies
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> AddAutomaticSeasonalDummies:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add automatic seasonal dummies.

        Outliers among weekly percentage changes are detected by normalizing and comparing with a certain threshold.
        Only weekly frequency is supported.

        :param df: original data
        :param var_name: the name of the variable to model and forecast
        :param lim_num_dummies: limit on the number of seasonal dummies
        :param threshold: quantile cutoff for "irregular" time series changes
        :return: original data with new columns corresponding to seasonal dummies
        """
        freq = pd.infer_freq(X.index)
        if freq[0] != "W":
            raise RuntimeError("Only weekly data is supported. Frequency detected: {}".format(freq))

        lbl_diff = "diff"
        lbl_week_number = "week_number"
        data = X.copy()
        data[lbl_week_number] = data.index.map(lambda x: x.isocalendar()[1])
        data[lbl_diff] = data[self.var_name] - data[self.var_name].ewm(com=10).mean()
        data[lbl_diff] = (data[lbl_diff] / data[lbl_diff].std()).abs()
        mean_abs_diff = data.iloc[10:].groupby(lbl_week_number)[lbl_diff].mean().dropna()
        normalized = (
            (mean_abs_diff - mean_abs_diff.median()).abs() / median_abs_deviation(mean_abs_diff, scale=1 / 1.4826**2)
        ).sort_values(ascending=False)
        weeks = normalized.loc[normalized > self.threshold].index[: self.lim_num_dummies]
        for week in weeks:
            data[self.lbl_auto_dummy.format(week)] = 0.0
            data.loc[data.index.map(lambda x: x.isocalendar()[1]) == week, self.lbl_auto_dummy.format(week)] = 1
        return data.drop([lbl_week_number, lbl_diff], axis=1)
