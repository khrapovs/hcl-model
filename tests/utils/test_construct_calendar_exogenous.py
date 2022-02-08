import numpy as np
import pandas as pd

from hcl_model.transformers.calendar import CalendarTransformer
from hcl_model.utils.construct_calendar_exogenous import construct_calendar_exogenous


class TestConstructCalendarExogenous:

    # countries
    lbl_country_code_tw = "TW"
    lbl_country_code_de = "DE"

    # holidays
    lbl_cny = "Chinese New Year"
    lbl_easter_monday = "Easter Monday"
    lbl_new_year = "New year"
    lbl_cgw_tw = "National Day/Double Tenth Day"

    lbl_holiday_name = "holiday_name"
    lbl_country_code = "country_code"
    lbl_lags = "lags"
    lbl_add_td = "add_td"

    def test_construct_calendar_exogenous(self):
        np.random.seed(42)
        nobs, num_steps = 100, 26
        endog = pd.Series(
            np.random.random(nobs) + 100,
            index=pd.date_range("2017-02-01", periods=nobs, freq="W-FRI"),
        )
        lbl_endog = "endog"
        df = pd.DataFrame({lbl_endog: endog})
        lbl_dummy = CalendarTransformer.lbl_auto_dummy

        weeks = [1, 5, 10, 50]
        for week in weeks:
            df.loc[df.index.map(lambda x: x.isocalendar()[1]) == week, lbl_endog] += 100
            df[lbl_dummy.format(week)] = 0.0
            df[lbl_dummy.format(week + 1)] = 0.0
            df.loc[
                df.index.map(lambda x: x.isocalendar()[1]) == week,
                lbl_dummy.format(week),
            ] = 1
            df.loc[
                df.index.map(lambda x: x.isocalendar()[1]) == week + 1,
                lbl_dummy.format(week + 1),
            ] = 1

        endog = df[lbl_endog]

        exog = construct_calendar_exogenous(endog=endog, num_steps=num_steps)
        assert exog.shape == (nobs + num_steps, 1)

        for trend in ["c", "t", "ct", "ctt"]:
            exog = construct_calendar_exogenous(endog=endog, num_steps=num_steps, trend=trend)
            assert exog.shape == (nobs + num_steps, len(trend))

        for num in [1, 2, 3]:
            exog = construct_calendar_exogenous(endog=endog, num_steps=num_steps, splines_df=num)
            assert exog.shape == (nobs + num_steps, num + 1)

        holidays = [
            {
                self.lbl_holiday_name: self.lbl_cny,
                self.lbl_country_code: self.lbl_country_code_tw,
                self.lbl_lags: [1],
            },
            {
                self.lbl_holiday_name: self.lbl_new_year,
                self.lbl_country_code: self.lbl_country_code_de,
                self.lbl_lags: [-1, 0, 2],
            },
            {
                self.lbl_holiday_name: self.lbl_easter_monday,
                self.lbl_country_code: self.lbl_country_code_de,
                self.lbl_lags: [-1, -2],
                self.lbl_add_td: "59 D",
            },
            {
                self.lbl_holiday_name: self.lbl_cgw_tw,
                self.lbl_country_code: self.lbl_country_code_tw,
            },
        ]

        for holiday in holidays:
            exog = construct_calendar_exogenous(endog=endog, num_steps=num_steps, holidays=[holiday])
            if self.lbl_lags in holiday.keys():
                ncols = len(holiday[self.lbl_lags])
            else:
                ncols = 5
            assert exog.shape == (nobs + num_steps, ncols + 1)

        exog = construct_calendar_exogenous(
            endog=endog,
            num_steps=num_steps,
            auto_dummy_max_number=len(weeks) * 2,
            auto_dummy_threshold=3,
        )

        assert exog.shape == (nobs + num_steps, len(weeks) + 1)

        holiday = {
            self.lbl_holiday_name: self.lbl_new_year,
            self.lbl_country_code: self.lbl_country_code_de,
            self.lbl_lags: [-1, 0, 1],
        }

        exog = construct_calendar_exogenous(
            endog=endog,
            num_steps=num_steps,
            holidays=[holiday],
            auto_dummy_max_number=len(weeks) * 2,
            auto_dummy_threshold=3,
        )

        # test that one column is dropped since automatic dummy coincides with NY
        assert len(holiday[self.lbl_lags]) + len(weeks) == len(exog.columns)
