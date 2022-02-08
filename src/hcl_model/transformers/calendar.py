from typing import Iterable

import pandas as pd
from patsy import dmatrix
from scipy.stats import median_abs_deviation

from hcl_model.labels import LabelsCommon, LabelsExog
from hcl_model.utils.calendar_reader import CalendarReader


class CalendarTransformer:
    """
    Allows to easily add a variety of calendar-based columns to the exogenous regressor matrix.
    """

    lbl = LabelsCommon()
    lbl_exog = LabelsExog()

    lbl_weekly_freq = "7 D"
    lbl_auto_dummy = "automatic_dummy_{}"

    def __init__(self):
        self._cal_reader = CalendarReader()

    def add_periodic_splines(
        self,
        df: pd.DataFrame,
        input_level: str = "isoweek",
        degrees_of_freedom: int = 4,
    ) -> pd.DataFrame:
        """
        Add periodic spline scores to the dataframe df.

        df must have a datetime column as index. This method adds periodic splines
        scores to the dataframe by using its index as input.

        Typical output:

        .. code-block:: python

            df = pd.DataFrame({'x1': np.random.normal(0, 1, 53 * 3)},
                               index=pd.date_range(start='2015-01-01', freq='7 D', periods=53 * 3))

            cal = CalendarTransformer()
            cal.add_periodic_splines(df).head()

                              x1  spline_dim_1  spline_dim_2  spline_dim_3  spline_dim_4
            2015-01-01 -0.301179     -0.439735     -0.438071     -0.438801     -0.437531
            2015-01-08  0.301794     -0.343647     -0.460494     -0.409252     -0.498250
            2015-01-15 -0.431848     -0.218113     -0.474556     -0.374233     -0.526180
            2015-01-22  2.669445     -0.071386     -0.479289     -0.335690     -0.526172
            2015-01-29  0.220345      0.088282     -0.473725     -0.295568     -0.503074

        """
        data = df.copy(deep=True)
        data[self.lbl_exog.lbl_calendar_numeric] = self._get_numeric_periodicity(
            data.index, input_level=input_level
        ).astype(float)

        # the 'cc' part in the string of the dmatrix call is responsible for periodic splines.
        formula = f"cc({self.lbl_exog.lbl_calendar_numeric}, df={degrees_of_freedom}, constraints='center') - 1"
        calendar_matrix = dmatrix(formula, data=data, return_type="dataframe")

        # rename columns for convenience
        cols = calendar_matrix.columns
        calendar_matrix = calendar_matrix.rename(
            columns={col: f"{self.lbl_exog.lbl_spline_dim}_{num + 1}" for num, col in enumerate(cols)}
        )
        data[calendar_matrix.columns] = calendar_matrix

        return data.drop(self.lbl_exog.lbl_calendar_numeric, axis=1)

    def _get_numeric_periodicity(self, dates: pd.DatetimeIndex, input_level: str = "isoweek") -> pd.Series:
        """
        Add a column of numeric scores between 0 and 1 that represent annual periodicity.
        Input levels 'isoweek' and 'month' supported currently. For any other input we fall
        back to day of the year. The effect of using week or month is that all dates within one
        week or month get the same score while fall-back routine assigns each day of year same score.

        dates must have datetime data type.
        """
        if input_level == self.lbl.isoweek:
            return dates.isocalendar().week / 53
        elif input_level == self.lbl.month:
            return dates.month / 12
        elif input_level == self.lbl.day:
            return dates.isocalendar().day / 366
        else:
            raise ValueError(f"Unknown input_level argument: {input_level}")

    def add_holiday_triangles(
        self,
        df: pd.DataFrame,
        holiday_name: str,
        country_code: str,
        triangle_name: str = "",
        lh_factors: Iterable[float] = (1.0, 2.0, 3.0),
        rh_factors: Iterable[float] = (2.0, 1.0),
        by: str = None,
        merge_factors: bool = False,
    ) -> pd.DataFrame:
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
        lbl_lh = "{}_lh_factor".format(triangle_name)
        lbl_rh = "{}_rh_factor".format(triangle_name)

        dates = self._read_holiday_dates(holiday_name, country_code)

        lh_df = list()
        for i, factor in enumerate(reversed([0.0] + list(lh_factors))):

            if merge_factors and i == 0:
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
        for i, factor in enumerate(list(rh_factors) + [0.0]):

            if merge_factors and i == 0:
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
        if merge_factors:
            edf["{}_factor".format(triangle_name)] = edf[lbl_lh] + edf[lbl_rh]
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
            df,
            edf.set_index(self.lbl.date),
            left_index=True,
            right_index=True,
            by=by,
            tolerance=pd.Timedelta(self.lbl_weekly_freq),
        )

    def add_holiday_dummies(
        self,
        df: pd.DataFrame,
        holiday_name: str,
        country_code: str,
        dummy_name: str = None,
        add_td: str = None,
        lags: Iterable[int] = (-2, -1, 0, 1, 2),
        by: str = None,
    ) -> pd.DataFrame:
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
        dummy = self._get_holiday_dummy(holiday_name, country_code, add_td)

        if dummy_name is not None:
            dummy = dummy.rename(columns={self.lbl_exog.lbl_is_holiday: dummy_name})

        return self._add_dummy_lags_and_merge(df, dummy, lags, by)

    def _get_holiday_dummy(self, holiday_name: str, country_code: str, add_td: str = None) -> pd.DataFrame:
        """
        Any holiday included in the input dataframe can be translated to dummy column.
        Use e.g. `holiday_name` = 'Chinese New Year' and `country_code` = 'CN'.
        """
        dates = self._read_holiday_dates(holiday_name, country_code).assign(**{self.lbl_exog.lbl_is_holiday: 1})
        if add_td is not None:
            dates[self.lbl.date] = dates[self.lbl.date] + pd.Timedelta(add_td)

        # TODO: this is a very dirty quick fix!!!
        dates[self.lbl.date] -= pd.Timedelta("7 D")

        return pd.merge_asof(
            pd.DataFrame({self.lbl.date: self._get_weekly_date_range()}),
            dates,
            on=self.lbl.date,
            allow_exact_matches=False,
            tolerance=pd.Timedelta(self.lbl_weekly_freq),
        ).fillna(0)

    def _add_dummy_lags_and_merge(self, df: pd.DataFrame, dummy, lags: Iterable[int], by: str) -> pd.DataFrame:
        """
        Add optional forward and backward looking lags and merge to dataframe.
        """
        dummy_name = dummy.columns[-1]
        for lag in lags:
            lag_name = "{}_lag_{}".format(dummy_name, str(lag).replace("-", "neg"))
            dummy[lag_name] = dummy[dummy_name].shift(lag)

        # Add 0 to lags to preserve original dummy and rename:
        dummy = dummy.drop(dummy_name, axis=1).rename(columns={"{}_lag_0".format(dummy_name): dummy_name})

        return pd.merge_asof(
            df,
            dummy.set_index(self.lbl.date),
            left_index=True,
            right_index=True,
            by=by,
            tolerance=pd.Timedelta(self.lbl_weekly_freq),
        )

    def _get_weekly_date_range(self, start: str = "2005-01-03", periods: int = 53 * 25) -> pd.DatetimeIndex:
        return pd.date_range(start=start, periods=periods, freq=self.lbl_weekly_freq)

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

    @classmethod
    def add_automatic_seasonal_dummies(
        cls,
        df: pd.DataFrame,
        var_name: str,
        lim_num_dummies: int = 5,
        threshold: float = 3,
    ) -> pd.DataFrame:
        """Add automatic seasonal dummies.

        Outliers among weekly percentage changes are detected by normalizing and comparing with a certain threshold.
        Only weekly frequency is supported.

        :param df: original data
        :param var_name: the name of the variable to model and forecast
        :param lim_num_dummies: limit on the number of seasonal dummies
        :param threshold: quantile cutoff for "irregular" time series changes
        :return: original data with new columns corresponding to seasonal dummies
        """
        freq = pd.infer_freq(df.index)
        if freq[0] != "W":
            raise RuntimeError("Only weekly data is supported. Frequency detected: {}".format(freq))

        lbl_diff = "diff"
        lbl_week_number = "week_number"
        data = df.copy()
        data[lbl_week_number] = data.index.map(lambda x: x.isocalendar()[1])
        data[lbl_diff] = data[var_name] - data[var_name].ewm(com=10).mean()
        data[lbl_diff] = (data[lbl_diff] / data[lbl_diff].std()).abs()
        mean_abs_diff = data.iloc[10:].groupby(lbl_week_number)[lbl_diff].mean().dropna()
        normalized = (
            (mean_abs_diff - mean_abs_diff.median()).abs() / median_abs_deviation(mean_abs_diff, scale=1 / 1.4826 ** 2)
        ).sort_values(ascending=False)
        weeks = normalized.loc[normalized > threshold].index[:lim_num_dummies]
        for week in weeks:
            data[cls.lbl_auto_dummy.format(week)] = 0.0
            data.loc[
                data.index.map(lambda x: x.isocalendar()[1]) == week,
                cls.lbl_auto_dummy.format(week),
            ] = 1
        return data.drop([lbl_week_number, lbl_diff], axis=1)
