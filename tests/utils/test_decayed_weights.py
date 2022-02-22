from typing import Optional

import numpy as np
import pandas as pd
import pytest

from hcl_model.utils.decayed_weights import decayed_weights


@pytest.mark.parametrize("nobs", [5, 10, 99, 100, 101])
@pytest.mark.parametrize("full_weight_obs", [None, 15, 1000])
@pytest.mark.parametrize("downweight_order", [None, 0, 2, 10])
def test_decayed_weights(nobs: int, full_weight_obs: Optional[int], downweight_order: Optional[int]) -> None:
    endog = pd.Series(np.random.normal(size=nobs), index=pd.date_range("2018-01-01", periods=nobs, freq="W-FRI"))
    weights = decayed_weights(y=endog, full_weight_obs=full_weight_obs, downweight_order=downweight_order)
    assert isinstance(weights, np.ndarray)
    assert weights.shape[0] == endog.shape[0]
    assert (weights >= 0).all()
    assert (weights <= 1).all()

    if full_weight_obs is None:
        assert (weights == 1).all()
    else:
        assert (weights[-full_weight_obs:] == 1).all()

        if downweight_order is None:
            assert (weights[:-full_weight_obs] == 0).all()
        elif downweight_order == 0:
            assert (weights[:-full_weight_obs] == 1).all()
        else:
            assert (weights[1:-full_weight_obs] > 0).all()
            assert (np.diff(weights[:-full_weight_obs]) > 0).all()
