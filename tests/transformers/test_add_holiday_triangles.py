import numpy as np
import pandas as pd

from hcl_model.transformers.add_holiday_triangles import AddHolidayTriangles


class TestAddHolidayTriangles:
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

    def test_add_holiday_triangles(self) -> None:
        xreg = self.get_xreg_example()

        lh_factors = [1.0, 2.0]
        rh_factors = [7.0]

        tasks = [
            (self.lbl_cny, self.lbl_country_code_tw),
            (self.lbl_new_year, self.lbl_country_code_de),
            (self.lbl_easter_monday, self.lbl_country_code_de),
            (self.lbl_cgw_tw, self.lbl_country_code_tw),
        ]

        for holiday, country in tasks:
            df1 = AddHolidayTriangles(
                holiday_name=holiday, country_code=country, lh_factors=lh_factors, rh_factors=rh_factors
            ).transform(X=xreg)
            df2 = AddHolidayTriangles(holiday_name=holiday, country_code=country, merge_factors=True).transform(X=xreg)

            assert df1.shape == (xreg.shape[0], xreg.shape[1] + 2)
            assert df2.shape == (xreg.shape[0], xreg.shape[1] + 1)

            assert set(df1.iloc[:, -2:].values.ravel()) <= set([0.0] + lh_factors + rh_factors)
