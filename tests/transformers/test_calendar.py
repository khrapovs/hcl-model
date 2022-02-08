import numpy as np
import pandas as pd
import pytest

from hcl_model.transformers.calendar import CalendarTransformer


class TestCalendarTransformer:
    # countries
    lbl_country_code_tw = "TW"
    lbl_country_code_de = "DE"

    # holidays
    lbl_cny = "Chinese New Year"
    lbl_easter_monday = "Easter Monday"
    lbl_new_year = "New year"
    lbl_cgw_tw = "National Day/Double Tenth Day"

    @staticmethod
    def get_xreg_example() -> pd.DataFrame:
        return pd.DataFrame(
            {"x1": np.random.normal(0, 1, 53 * 3)},
            index=pd.date_range(start="2015-01-01", freq="7 D", periods=53 * 3),
        )

    def test_add_periodic_splines(self):
        xreg = self.get_xreg_example()
        cal_transformer = CalendarTransformer()

        degrees_of_freedom = 3
        df_default = cal_transformer.add_periodic_splines(df=xreg, degrees_of_freedom=degrees_of_freedom)

        assert df_default.shape == (xreg.shape[0], degrees_of_freedom + xreg.shape[1])
        assert np.sum(df_default.isnull()).sum() == 0

        df = cal_transformer.add_periodic_splines(df=xreg, degrees_of_freedom=degrees_of_freedom, input_level="isoweek")

        assert df.shape == (xreg.shape[0], degrees_of_freedom + xreg.shape[1])
        assert np.sum(df.isnull()).sum() == 0
        pd.testing.assert_frame_equal(df, df_default)

        df = cal_transformer.add_periodic_splines(df=xreg, degrees_of_freedom=degrees_of_freedom, input_level="month")

        assert df.shape == (xreg.shape[0], degrees_of_freedom + xreg.shape[1])
        assert np.sum(df.isnull()).sum() == 0

        with pytest.raises(ValueError):
            cal_transformer.add_periodic_splines(df=xreg, degrees_of_freedom=degrees_of_freedom, input_level="day")

        with pytest.raises(ValueError) as error:
            wrong_input_level = "wrong_input_level"
            cal_transformer.add_periodic_splines(
                df=xreg,
                degrees_of_freedom=degrees_of_freedom,
                input_level=wrong_input_level,
            )

        assert wrong_input_level in str(error)

    def test_add_holiday_triangles(self):
        xreg = self.get_xreg_example()
        cal_transformer = CalendarTransformer()

        lh_factors = [1.0, 2.0]
        rh_factors = [7.0]

        tasks = [
            (self.lbl_cny, self.lbl_country_code_tw),
            (self.lbl_new_year, self.lbl_country_code_de),
            (self.lbl_easter_monday, self.lbl_country_code_de),
            (self.lbl_cgw_tw, self.lbl_country_code_tw),
        ]

        for holiday, country in tasks:
            df1 = cal_transformer.add_holiday_triangles(
                xreg, holiday, country, lh_factors=lh_factors, rh_factors=rh_factors
            )
            df2 = cal_transformer.add_holiday_triangles(xreg, holiday, country, merge_factors=True)

            assert df1.shape == (xreg.shape[0], xreg.shape[1] + 2)
            assert df2.shape == (xreg.shape[0], xreg.shape[1] + 1)

            assert set(df1.iloc[:, -2:].values.ravel()) <= set([0.0] + lh_factors + rh_factors)

    def test_add_holiday_dummies(self):
        xreg = self.get_xreg_example()
        cal_transformer = CalendarTransformer()

        xreg_cny = (
            cal_transformer.add_holiday_dummies(xreg, self.lbl_cny, self.lbl_country_code_tw, lags=[1]),
            [1],
        )
        xreg_ny = (
            cal_transformer.add_holiday_dummies(xreg, self.lbl_new_year, self.lbl_country_code_de, lags=[-1, 0, 2]),
            [-1, 0, 2],
        )
        xreg_cc = (
            cal_transformer.add_holiday_dummies(
                xreg,
                self.lbl_easter_monday,
                self.lbl_country_code_de,
                add_td="59 D",
                lags=[-1, -2],
            ),
            [-1, -2],
        )
        xreg_cgw = (
            cal_transformer.add_holiday_dummies(xreg, self.lbl_cgw_tw, self.lbl_country_code_tw),
            None,
        )

        for df, lags in [xreg_cny, xreg_cc, xreg_ny, xreg_cgw]:

            if lags is None:
                lags_len = 5  # default is lags=(-1,-2,0,1,2)
            else:
                lags_len = len(lags)

            assert np.sum(df.isnull()).sum() == 0
            assert df.shape[1] == lags_len + 1
            assert df.shape[0] == xreg.shape[0]
            assert set(np.ravel(df.iloc[:, 1:].values)) == {0.0, 1.0}


class TestAddAutomaticSeasonalDummies:
    def test_add_automatic_seasonal_dummies(self):
        np.random.seed(42)
        nobs = 200
        lbl_endog = "endog"
        lbl_date = "date"
        lbl_dummy = CalendarTransformer.lbl_auto_dummy
        df = pd.DataFrame(
            {lbl_endog: np.random.random(nobs) + 100},
            index=pd.DatetimeIndex(pd.date_range("2017-01-01", periods=nobs, freq="W-FRI"), name=lbl_date),
        )
        weeks = {2: 10, 5: 5, 10: 20, 50: 7}
        for week, change in weeks.items():
            df.loc[df.index.map(lambda x: x.isocalendar()[1]) == week, lbl_endog] += change
            df[lbl_dummy.format(week)] = 0.0
            df.loc[
                df.index.map(lambda x: x.isocalendar()[1]) == week,
                lbl_dummy.format(week),
            ] = 1

        df_result = CalendarTransformer.add_automatic_seasonal_dummies(
            df=df[[lbl_endog]],
            var_name=lbl_endog,
            threshold=3,
            lim_num_dummies=len(weeks) * 2,
        )

        pd.testing.assert_frame_equal(df.sort_index(axis=1), df_result.sort_index(axis=1))

        df_result = CalendarTransformer.add_automatic_seasonal_dummies(
            df=df[[lbl_endog]], var_name=lbl_endog, threshold=3, lim_num_dummies=2
        )

        correct_cols = [lbl_endog, lbl_dummy.format(10), lbl_dummy.format(2)]

        pd.testing.assert_frame_equal(
            df.filter(items=correct_cols).sort_index(axis=1),
            df_result.sort_index(axis=1),
        )

        df_result = CalendarTransformer.add_automatic_seasonal_dummies(
            df=df[[lbl_endog]], var_name=lbl_endog, threshold=3, lim_num_dummies=4
        )
        correct_cols = [
            lbl_endog,
            lbl_dummy.format(10),
            lbl_dummy.format(50),
            lbl_dummy.format(2),
            lbl_dummy.format(5),
        ]

        pd.testing.assert_frame_equal(
            df.filter(items=correct_cols).sort_index(axis=1),
            df_result.sort_index(axis=1),
        )

        df_result = CalendarTransformer.add_automatic_seasonal_dummies(
            df=df[[lbl_endog]], var_name=lbl_endog, threshold=3, lim_num_dummies=6
        )
        correct_cols = [
            lbl_endog,
            lbl_dummy.format(10),
            lbl_dummy.format(2),
            lbl_dummy.format(5),
            lbl_dummy.format(50),
        ]

        pd.testing.assert_frame_equal(
            df.filter(items=correct_cols).sort_index(axis=1),
            df_result.sort_index(axis=1),
        )

        num_dummies = 0
        for quantile in np.linspace(0.99, 0.01, num=10):
            df_result = CalendarTransformer.add_automatic_seasonal_dummies(
                df=df[[lbl_endog]],
                var_name=lbl_endog,
                threshold=quantile,
                lim_num_dummies=len(weeks) * 2,
            )
            num_dummies_new = len(df_result.columns) - 1
            assert num_dummies_new >= num_dummies
            num_dummies = num_dummies_new

        assert num_dummies == len(weeks) * 2

    def test_add_automatic_seasonal_dummies_raises(self):
        nobs = 200
        lbl_endog = "endog"
        lbl_date = "date"
        freq = "M"
        df = pd.DataFrame(
            {lbl_endog: np.random.random(nobs)},
            index=pd.DatetimeIndex(pd.date_range("2017-01-01", periods=nobs, freq=freq), name=lbl_date),
        )

        with pytest.raises(RuntimeError) as e:
            CalendarTransformer.add_automatic_seasonal_dummies(
                df=df[[lbl_endog]], var_name=lbl_endog, threshold=0.5, lim_num_dummies=8
            )

            assert freq in str(e)

    def test_add_automatic_seasonal_dummies_empty(self):
        nobs = 200
        lbl_endog = "endog"
        lbl_date = "date"
        df = pd.DataFrame(
            {lbl_endog: np.ones(nobs)},
            index=pd.DatetimeIndex(pd.date_range("2017-01-01", periods=nobs, freq="W-FRI"), name=lbl_date),
        )

        df_result = CalendarTransformer.add_automatic_seasonal_dummies(
            df=df[[lbl_endog]], var_name=lbl_endog, threshold=3, lim_num_dummies=8
        )

        pd.testing.assert_frame_equal(df[[lbl_endog]], df_result)
