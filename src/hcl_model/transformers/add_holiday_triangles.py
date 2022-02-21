from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from hcl_model.labels import LabelsCommon
from hcl_model.utils.calendar_reader import CalendarReader


class AddHolidayTriangles(BaseEstimator, TransformerMixin):
    lbl = LabelsCommon()
    lbl_weekly_freq = "7 D"

    def __init__(
        self,
        holiday_name: str,
        country_code: str,
        triangle_name: str = "",
        lh_factors: Iterable[float] = (1.0, 2.0, 3.0),
        rh_factors: Iterable[float] = (2.0, 1.0),
        by: str = None,
        merge_factors: bool = False,
    ) -> None:
        self.holiday_name = holiday_name
        self.country_code = country_code
        self.triangle_name = triangle_name
        self.lh_factors = lh_factors
        self.rh_factors = rh_factors
        self.by = by
        self.merge_factors = merge_factors
        self._cal_reader = CalendarReader()

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> AddHolidayTriangles:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds triangular calendar regressors to the dataframe df. Index of df must be sorted and of date type.
        If there are additional grouping columns, then this can be set with the `by` argument (see pd.merge_asof).

        The user must pass two list, specifying the factors values that will be used.
        lh_factors = [..., factor one week before the holiday, factor on the week of the holiday]
        rh_factors = [factor on the week of the holiday, factor one week after the holiday, ...]

        If `merge_factors` = False, then a single numeric column of holiday scores is added. If not, two columns
        are added, representing effects before (including at) and after the holiday, respectively.

        Typical output:

        .. code-block:: python

            df = pd.DataFrame({'x1': np.random.normal(0, 1, 53 * 3)},
                               index=pd.date_range(start='2015-01-01', freq='7 D', periods=53 * 3))

            cal = CalendarTransformer()
            cal.add_holiday_triangles(df, 'New year', 'DE').head()

                              x1  _lh_factor  _rh_factor
            2015-01-01 -0.301179         3.0         2.0
            2015-01-08  0.301794         0.0         1.0
            2015-01-15 -0.431848         0.0         0.0
            2015-01-22  2.669445         0.0         0.0
            2015-01-29  0.220345         0.0         0.0
        """
        lbl_lh = "{}_lh_factor".format(self.triangle_name)
        lbl_rh = "{}_rh_factor".format(self.triangle_name)

        dates = self._read_holiday_dates(self.holiday_name, self.country_code)

        lh_df = list()
        for i, factor in enumerate(reversed([0.0] + list(self.lh_factors))):

            if self.merge_factors and i == 0:
                # if the user required the lh and rh factors to be merged in one, take only half value for the
                # day shared between lh and rh
                factor = factor * 0.5

            lh_df.append(
                pd.DataFrame(
                    {
                        self.lbl.date: dates[self.lbl.date] - pd.Timedelta("{} D".format(7 * i)),
                        lbl_lh: [factor] * len(dates),
                    }
                )
            )
        lh_df = pd.concat(lh_df).sort_values([self.lbl.date], ascending=True)

        # - build right-hand holiday factors
        rh_df = list()
        for i, factor in enumerate(list(self.rh_factors) + [0.0]):

            if self.merge_factors and i == 0:
                # if the user required the lh and rh factors to be merged in one, take only half value for the
                # day shared between lh and rh
                factor = factor * 0.5

            rh_df.append(
                pd.DataFrame(
                    {
                        self.lbl.date: dates[self.lbl.date] + pd.Timedelta("{} D".format(7 * i)),
                        lbl_rh: [factor] * len(dates),
                    }
                )
            )
        rh_df = pd.concat(rh_df).sort_values([self.lbl.date], ascending=True)

        # - generate new DataFrame with all holidays factors
        edf = pd.merge(left=lh_df, right=rh_df, how="outer", on=self.lbl.date).fillna(0).reset_index(drop=True)

        # - if the user required the lh and rh factors to be merged in one, merge them by summing the two columns
        # NOTE: the lh and rh factors in the case of merged factor are built to make the sum of the two columns
        # producing the merged factor
        if self.merge_factors:
            edf["{}_factor".format(self.triangle_name)] = edf[lbl_lh] + edf[lbl_rh]
            edf = edf.drop(columns=[lbl_lh, lbl_rh])

        # - insert all other missing days in the calendar, with factor 0
        edf = (
            edf.set_index(self.lbl.date)
            .sort_index()
            .reindex(
                pd.date_range(
                    start="{}-01-01".format(dates[self.lbl.date].dt.year.min()),
                    end="{}-12-31".format(dates[self.lbl.date].dt.year.max()),
                    freq="D",
                ),
                method="nearest",
            )
            .reset_index()
            .rename(columns={"index": self.lbl.date})
        )

        # - remove the time from the date column
        edf[self.lbl.date] = pd.to_datetime(edf[self.lbl.date].dt.date)

        return pd.merge_asof(
            X,
            edf.set_index(self.lbl.date),
            left_index=True,
            right_index=True,
            by=self.by,
            tolerance=pd.Timedelta(self.lbl_weekly_freq),
        )

    def _read_holiday_dates(
        self,
        holiday_name: str,
        country_code: str,
        from_year: int = 2000,
        to_year: int = 2030,
    ) -> pd.DataFrame:
        return (
            self._cal_reader.get_holidays(holiday_name, country_code, from_year, to_year)
            .reset_index()
            .filter(items=[self.lbl.date])
        )
