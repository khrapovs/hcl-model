from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def check_X_y(X: Optional[pd.DataFrame], y: Union[pd.Series, np.ndarray]) -> Tuple[Optional[pd.DataFrame], pd.Series]:
    X = X.copy() if X is not None else None
    y = pd.Series(y, index=X.index, name="value") if isinstance(y, np.ndarray) else y.copy()
    return X, y
