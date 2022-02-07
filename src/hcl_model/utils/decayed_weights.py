import numpy as np
import pandas as pd


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
        downweighted = np.power(
            np.linspace(start=0, stop=1, num=len_weights - full_weight_obs),
            downweight_order,
        )
        return np.concatenate((downweighted, np.ones(full_weight_obs)), axis=None)
