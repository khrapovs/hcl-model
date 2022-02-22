from typing import Optional, Union

import numpy as np
import pandas as pd


def decayed_weights(
    y: Union[np.ndarray, pd.Series], full_weight_obs: Optional[int] = None, downweight_order: Optional[int] = None
) -> np.ndarray:
    """Construct weights starting to decay at `start_from` with polynomial order.

    :param y: array-like
    :param full_weight_obs: optional int
        Number of observations from the end that have full weight (1).
        If `None`, then the weights returned are all ones.
    :param downweight_order: optional int
        Polynomial order of decrease in weights.
        If `None`, then all weights prior to `full_weight_obs` are equal to zero.
    :return: numpy 1d array of floats between 0 and 1 with same length as endog.
    """
    len_weights = y.size

    if (full_weight_obs is None) or (len_weights <= full_weight_obs):
        return np.ones(len_weights)
    else:
        if downweight_order is None:
            downweighted = np.zeros(len_weights - full_weight_obs)
        else:
            downweighted = np.power(np.linspace(start=0, stop=1, num=len_weights - full_weight_obs), downweight_order)
        return np.concatenate((downweighted, np.ones(full_weight_obs)), axis=None)
