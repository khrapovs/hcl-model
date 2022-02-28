import numpy as np
import pandas as pd

from hcl_model.transformers.add_holiday_dummies import AddHolidayDummies


class TestAddHolidayDummies:
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

    def test_add_holiday_dummies(self) -> None:
        xreg = self.get_xreg_example()

        xreg_cny = (
            AddHolidayDummies(holiday_name=self.lbl_cny, country_code=self.lbl_country_code_tw, lags=[1]).transform(
                X=xreg
            ),
            [1],
        )
        xreg_ny = (
            AddHolidayDummies(
                holiday_name=self.lbl_new_year, country_code=self.lbl_country_code_de, lags=[-1, 0, 2]
            ).transform(X=xreg),
            [-1, 0, 2],
        )
        xreg_cc = (
            AddHolidayDummies(
                holiday_name=self.lbl_easter_monday, country_code=self.lbl_country_code_de, add_td="59 D", lags=[-1, -2]
            ).transform(X=xreg),
            [-1, -2],
        )
        xreg_cgw = (
            AddHolidayDummies(holiday_name=self.lbl_cgw_tw, country_code=self.lbl_country_code_tw).transform(X=xreg),
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
