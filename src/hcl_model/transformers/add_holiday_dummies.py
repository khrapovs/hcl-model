from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from hcl_model.labels import LabelsCommon, LabelsExog
from hcl_model.utils.calendar_reader import CalendarReader


class AddHolidayDummies(BaseEstimator, TransformerMixin):
    lbl = LabelsCommon()
    lbl_exog = LabelsExog()

    lbl_weekly_freq = "7 D"

    def __init__(
        self,
        holiday_name: str,
        country_code: str,
        dummy_name: str = None,
        add_td: str = None,
        lags: Iterable[int] = (-2, -1, 0, 1, 2),
        by: str = None,
    ) -> None:
        self.holiday_name = holiday_name
        self.country_code = country_code
        self.dummy_name = dummy_name
        self.add_td = add_td
        self.lags = lags
        self.by = by
        self._cal_reader = CalendarReader()

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> AddHolidayDummies:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds calendar effects to dataframe df with either 0 or 1 in each entry. 1 indicates that the corresponding
        date is close to the holiday that is considered.

        Inputs:
            df: dataframe with datetime index.
            holiday_name: name of the holiday. See CalendarReader.
            country_code: calendar's country code. See CalendarReader.
            dummy_name: optional name for renaming the dummy columns.
            add_td: adds to each holiday a fixed time delta. See add_cc_dummies as a typical example.
            lags: list of integers; add forward and backward looking lags of the dummy.
            by: name of group column if df should be grouped by that column when the merge_asof is applied.
        Output:
            returns df extended by dummy columns.

        Typical output:

        .. code-block:: python

            df = pd.DataFrame({'x1': np.random.normal(0, 1, 53 * 3)},
                               index=pd.date_range(start='2015-01-01', freq='7 D', periods=53 * 3))

            cal = CalendarTransformer()
            cal.add_holiday_dummies(df, 'Chinese New Year', 'TW').iloc[7:12, :]

                              x1  is_holiday  ...  is_holiday_lag_1  is_holiday_lag_2
            2015-02-19 -1.127138         0.0  ...               0.0               0.0
            2015-02-26  0.303294         1.0  ...               0.0               0.0
            2015-03-05 -0.398241         0.0  ...               1.0               0.0
            2015-03-12  1.281731         0.0  ...               0.0               1.0
            2015-03-19  2.303174         0.0  ...               0.0               0.0
        """
        dummy = self._get_holiday_dummy()

        if self.dummy_name is not None:
            dummy = dummy.rename(columns={self.lbl_exog.lbl_is_holiday: self.dummy_name})

        return self._add_dummy_lags_and_merge(df=X, dummy=dummy)

    def _get_holiday_dummy(self) -> pd.DataFrame:
        """
        Any holiday included in the input dataframe can be translated to dummy column.
        Use e.g. `holiday_name` = 'Chinese New Year' and `country_code` = 'CN'.
        """
        dates = self._read_holiday_dates().assign(**{self.lbl_exog.lbl_is_holiday: 1})
        if self.add_td is not None:
            dates[self.lbl.date] = dates[self.lbl.date] + pd.Timedelta(self.add_td)

        # TODO: this is a very dirty quick fix!!!
        dates[self.lbl.date] -= pd.Timedelta("7 D")

        return pd.merge_asof(
            pd.DataFrame({self.lbl.date: self._get_weekly_date_range()}),
            dates,
            on=self.lbl.date,
            allow_exact_matches=False,
            tolerance=pd.Timedelta(self.lbl_weekly_freq),
        ).fillna(0)

    def _add_dummy_lags_and_merge(self, df: pd.DataFrame, dummy: pd.DataFrame) -> pd.DataFrame:
        """
        Add optional forward and backward looking lags and merge to dataframe.
        """
        dummy_name = dummy.columns[-1]
        for lag in self.lags:
            lag_name = "{}_lag_{}".format(dummy_name, str(lag).replace("-", "neg"))
            dummy[lag_name] = dummy[dummy_name].shift(lag)

        # Add 0 to lags to preserve original dummy and rename:
        dummy = dummy.drop(dummy_name, axis=1).rename(columns={"{}_lag_0".format(dummy_name): dummy_name})

        return pd.merge_asof(
            df,
            dummy.set_index(self.lbl.date),
            left_index=True,
            right_index=True,
            by=self.by,
            tolerance=pd.Timedelta(self.lbl_weekly_freq),
        )

    def _get_weekly_date_range(self, start: str = "2005-01-03", periods: int = 53 * 25) -> pd.DatetimeIndex:
        return pd.date_range(start=start, periods=periods, freq=self.lbl_weekly_freq)

    def _read_holiday_dates(self, from_year: int = 2000, to_year: int = 2030) -> pd.DataFrame:
        return (
            self._cal_reader.get_holidays(
                holiday_name=self.holiday_name, country_code=self.country_code, from_year=from_year, to_year=to_year
            )
            .reset_index()
            .filter(items=[self.lbl.date])
        )
