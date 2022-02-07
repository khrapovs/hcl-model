import numpy as np
import pandas as pd

from hcl_model.utils.decayed_weights import decayed_weights


def test_decayed_weights() -> None:
    for nobs in [5, 10, 99, 100, 101]:
        endog = pd.Series(
            np.random.normal(size=nobs),
            index=pd.date_range("2018-01-01", periods=nobs, freq="W-FRI"),
        )

        for weights in [
            decayed_weights(endog=endog),
            decayed_weights(endog=endog, full_weight_obs=10),
            decayed_weights(endog=endog, full_weight_obs=nobs + 10),
            decayed_weights(endog=endog, full_weight_obs=10, downweight_order=0),
            decayed_weights(endog=endog, full_weight_obs=10, downweight_order=1),
            decayed_weights(endog=endog, full_weight_obs=10, downweight_order=2),
            decayed_weights(endog=endog, full_weight_obs=10, downweight_order=3),
        ]:
            assert isinstance(weights, np.ndarray)
            assert weights.shape[0] == endog.shape[0]
            assert (weights >= 0).all()
            assert (weights <= 1).all()

        weights = decayed_weights(endog=endog, full_weight_obs=nobs + 10)

        assert (weights == 1).all()

        weights = decayed_weights(endog=endog, full_weight_obs=10, downweight_order=0)

        assert (weights == 1).all()
