from typing import List, Union

import numpy as np
import pandas as pd
from statsmodels.tsa import tsatools as tst

from hcl_model.calendar_transformer import CalendarTransformer


def decayed_weights(endog: pd.Series, full_weight_obs: int = 52, downweight_order: int = 2) -> np.ndarray:
    """Construct weights starting to decay at `start_from` with polynomial order.

    :param endog: pd.Series with index of datetime type in consecutive order.
    :param full_weight_obs: number of observations from the end that have full weight (1).
    :param downweight_order: polynomial order of decrease in weights.
    :return: numpy 1d array of floats between 0 and 1 with same length as endog.
    """
    len_weights = endog.size

    if len_weights <= full_weight_obs:
        return np.ones(len_weights)
    else:
        downweighted = np.power(np.linspace(start=0, stop=1, num=len_weights - full_weight_obs), downweight_order)
        return np.concatenate((downweighted, np.ones(full_weight_obs)), axis=None)


def _get_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """Get a list of duplicate columns.

    https://thispointer.com/how-to-find-drop-duplicate-columns-in-a-dataframe-python-pandas/

    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.

    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    """
    duplicate_column_names = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            other_col = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(other_col):
                duplicate_column_names.add(df.columns.values[y])

    return list(duplicate_column_names)


def construct_calendar_exogenous(endog: pd.Series,
                                 num_steps: int = 52,
                                 trend: str = 'c',
                                 splines_df: int = None,
                                 holidays: List[dict] = None,
                                 auto_dummy_max_number: int = None,
                                 auto_dummy_threshold: float = 2) -> Union[None, pd.DataFrame]:
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
        endog.name = 'endog'

    extended = endog.reindex(pd.date_range(start=endog.index[0],
                                           periods=endog.shape[0] + num_steps,
                                           freq=pd.infer_freq(endog.index)))

    df = tst.add_trend(extended, trend=trend)
    cal_transformer = CalendarTransformer()

    if splines_df is not None:
        df = cal_transformer.add_periodic_splines(df, degrees_of_freedom=int(splines_df))

    if holidays is not None:
        for i, holiday in enumerate(holidays):
            if holiday is not None:
                df = cal_transformer.add_holiday_dummies(df, **holiday, dummy_name='holiday_{}'.format(i + 1))

    if auto_dummy_max_number is not None:
        df = cal_transformer.add_automatic_seasonal_dummies(df=df,
                                                            var_name=endog.name,
                                                            threshold=auto_dummy_threshold,
                                                            lim_num_dummies=auto_dummy_max_number)

    return df.drop(columns=_get_duplicate_columns(df)).iloc[:, 1:]
