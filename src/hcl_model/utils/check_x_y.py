from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def check_X_y(X: Optional[pd.DataFrame], y: Union[pd.Series, np.ndarray]) -> Tuple[Optional[pd.DataFrame], pd.Series]:
    X = X.copy() if X is not None else None
    if isinstance(y, np.ndarray) & (X is not None):
        y = pd.Series(y, index=X.index, name="value")
    elif isinstance(y, np.ndarray):
        raise TypeError(
            "Passing ndarray for y and None for X is not supported. "
            "Please, consider converting y to Series with DatetimeIndex."
        )
    else:
        y = y.copy()
    return X, y
