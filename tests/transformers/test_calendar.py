import numpy as np
import pandas as pd
import pytest

from hcl_model.transformers.calendar import CalendarTransformer


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
