from __future__ import annotations

import pandas as pd
from patsy import dmatrix

from hcl_model.labels import LabelsCommon, LabelsExog
from hcl_model.utils.calendar_reader import CalendarReader


class AddPeriodicSplines:
    lbl = LabelsCommon()
    lbl_exog = LabelsExog()

    def __init__(self, input_level: str = "isoweek", degrees_of_freedom: int = 4) -> None:
        self.input_level = input_level
        self.degrees_of_freedom = degrees_of_freedom
        self._cal_reader = CalendarReader()

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> AddPeriodicSplines:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        data = X.copy(deep=True)
        data[self.lbl_exog.lbl_calendar_numeric] = self._get_numeric_periodicity(
            data.index, input_level=self.input_level
        ).astype(float)

        # the 'cc' part in the string of the dmatrix call is responsible for periodic splines.
        formula = f"cc({self.lbl_exog.lbl_calendar_numeric}, df={self.degrees_of_freedom}, constraints='center') - 1"
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
