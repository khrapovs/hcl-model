import numpy as np
import pandas as pd
import pytest

from hcl_model.transformers.add_periodic_splines import AddPeriodicSplines


class TestCalendarTransformer:
    @staticmethod
    def get_xreg_example() -> pd.DataFrame:
        return pd.DataFrame(
            {"x1": np.random.normal(0, 1, 53 * 3)},
            index=pd.date_range(start="2015-01-01", freq="7 D", periods=53 * 3),
        )

    def test_add_periodic_splines(self):
        xreg = self.get_xreg_example()

        degrees_of_freedom = 3
        cal_transformer = AddPeriodicSplines(degrees_of_freedom=degrees_of_freedom)
        df_default = cal_transformer.transform(X=xreg)

        assert df_default.shape == (xreg.shape[0], degrees_of_freedom + xreg.shape[1])
        assert np.sum(df_default.isnull()).sum() == 0

        cal_transformer = AddPeriodicSplines(degrees_of_freedom=degrees_of_freedom, input_level="isoweek")
        df = cal_transformer.transform(X=xreg)

        assert df.shape == (xreg.shape[0], degrees_of_freedom + xreg.shape[1])
        assert np.sum(df.isnull()).sum() == 0
        pd.testing.assert_frame_equal(df, df_default)

        cal_transformer = AddPeriodicSplines(degrees_of_freedom=degrees_of_freedom, input_level="month")
        df = cal_transformer.transform(X=xreg)

        assert df.shape == (xreg.shape[0], degrees_of_freedom + xreg.shape[1])
        assert np.sum(df.isnull()).sum() == 0

        cal_transformer = AddPeriodicSplines(degrees_of_freedom=degrees_of_freedom, input_level="day")
        with pytest.raises(ValueError):
            cal_transformer.transform(X=xreg)

        wrong_input_level = "wrong_input_level"
        cal_transformer = AddPeriodicSplines(degrees_of_freedom=degrees_of_freedom, input_level=wrong_input_level)
        with pytest.raises(ValueError) as error:
            cal_transformer.transform(X=xreg)

        assert wrong_input_level in str(error)
