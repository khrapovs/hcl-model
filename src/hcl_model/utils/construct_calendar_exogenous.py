from typing import List, Union

import pandas as pd
from statsmodels.tsa.tsatools import add_trend

from hcl_model.transformers.calendar import CalendarTransformer
from hcl_model.utils.get_duplicate_columns import get_duplicate_columns


def construct_calendar_exogenous(
    endog: pd.Series,
    num_steps: int = 52,
    trend: str = "c",
    splines_df: int = None,
    holidays: List[dict] = None,
    auto_dummy_max_number: int = None,
    auto_dummy_threshold: float = 2,
) -> Union[None, pd.DataFrame]:
    """Construct deterministic exogenous variables.

    :param endog: time series of endogenous regressor
    :param num_steps: number of periods for forecasting.
        Output DataFrame will be longer than `endog` by this number of rows.
    :param trend: follows `statsmodels.tsa.tsatools.add_trend`
    :param splines_df: number of degrees of freedom for splines. 1 or more
    :param holidays: list of dicts with each dict representing one holiday.
        Each value of this series is a string representation of a dictionary.
        This dictionary is consumed by `utils.calendar_transformer.CalendarTransformer.add_holiday_dummies`.
    :param auto_dummy_max_number: limit on the number of automatic seasonal dummies
    :param auto_dummy_threshold: cutoff for "irregular" time series changes

    :return: DataFrame with exogenous regressors.
    """
    if endog.name is None:
        endog.name = "endog"

    extended = endog.reindex(
        pd.date_range(
            start=endog.index[0],
            periods=endog.shape[0] + num_steps,
            freq=pd.infer_freq(endog.index),
        )
    )

    df = add_trend(extended, trend=trend)
    cal_transformer = CalendarTransformer()

    if splines_df is not None:
        df = cal_transformer.add_periodic_splines(df, degrees_of_freedom=int(splines_df))

    if holidays is not None:
        for i, holiday in enumerate(holidays):
            if holiday is not None:
                df = cal_transformer.add_holiday_dummies(df, **holiday, dummy_name="holiday_{}".format(i + 1))

    if auto_dummy_max_number is not None:
        df = cal_transformer.add_automatic_seasonal_dummies(
            df=df,
            var_name=endog.name,
            threshold=auto_dummy_threshold,
            lim_num_dummies=auto_dummy_max_number,
        )

    return df.drop(columns=get_duplicate_columns(df)).iloc[:, 1:]
