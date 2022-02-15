from __future__ import annotations

from typing import List, Union, Sequence, Dict, Callable, Optional

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import WLS

from hcl_model.model_base import ModelBase


class HandCraftedLinearModel(ModelBase):
    r"""Hand Crafted Linear Model

    :param endog_transform: transformation functions of the endogenous variable
    :param exog_transform: transformation functions of the exogenous variables

    Notes
    -----
    This class implements a general HCL model described `here <../hcl_model_math.html>`_.

    **Example of an endog_transform input:**

    .. code-block:: python

        f = {'lag1': lambda y: y.shift(1),
             'local_mean': lambda y: y.shift(1).ewm(span=5).mean()}

    Note that names for f must be chosen. These names will be used to call columns of the explanatory data.

    **Example of a exog_transform input:**

    Say, exog has column names 'x1', 'x2' and 'x3' but you only want to transform
    the first two. Then use

    .. code-block:: python

        g1 = lambda x: x**x
        g2 = lambda x: x+1
        g = {'x1': g1, 'x2': g2}

    This will result in 2 dimensional regressor matrix exog_transformed with just
    the transforms and without `'x3'`. If `g_transform=None`, then original exog is used.

    .. code-block:: python

        model = HandCraftedLinearModel(endog_transform=f, exog_transform=g)

    This function should run only if `endog_transform` is not `None`.

    **Example of weights:**

    .. code-block:: python

        weights = np.power(np.arange(start=0, stop=1, step=1/len(endog)), 0.5)

    Such a weighting scheme will put more focus on recent observations, with a rather slow decrease
    to 0 with increasing distance.

    """

    lbl_original_endog = "original_endog"
    lbl_original_exog = "original_exog"

    def __init__(self, endog_transform: Dict[str, Callable] = None, exog_transform: Dict[str, Callable] = None) -> None:
        super().__init__()
        self._endog_transform = self._init_transform(name=self.lbl_original_endog, transform=endog_transform)
        self._exog_transform = self._init_transform(name=self.lbl_original_exog, transform=exog_transform)

    def _fit(self, weights: Union[Sequence, float] = 1.0) -> None:
        transformed = self._transform_all_data(endog=self._y_train, exog=self._x_train)
        rhs_vars = self._convert_transformed_dict_to_frame(transformed=transformed)
        self._fit_results = WLS(endog=self._y_train, exog=rhs_vars, weights=weights, missing="drop").fit()

    def _predict(
        self,
        num_steps: int,
        X: pd.DataFrame = None,
        weights: Union[Sequence, float] = 1.0,
        quantile_levels: List[float] = None,
        num_simulations: int = None,
    ) -> pd.DataFrame:
        endog_updated = pd.concat(
            [self._y_train, pd.Series(np.empty(num_steps), name=self._get_endog_name(), index=X.index)]
        )
        x_train_and_test = pd.concat([self._x_train, X])
        for j in range(num_steps):
            transformed = self._transform_all_data(
                endog=endog_updated[: self._nobs + j + 1], exog=x_train_and_test.iloc[: self._nobs + j + 1]
            )
            rhs_vars = self._convert_transformed_dict_to_frame(transformed=transformed).iloc[-1, :]
            endog_updated.iloc[self._nobs + j] = np.dot(rhs_vars, self.get_parameters())

        return endog_updated.iloc[self._nobs :].to_frame().rename_axis(index=self._y_train.index.name)

    def _simulate(self, num_steps: int, num_simulations: int, X: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        num_params = self.get_parameters().shape[0]
        simulation = np.empty((num_steps, num_simulations))

        # simulate num_simulations parameters from its multivariate normal distribution
        # Cholesky decomposition returns only the lower triangular part,
        # so it has to be transposed before multiplication
        beta_simulated = (
            np.dot(
                np.random.normal(loc=0, scale=1, size=(num_simulations, num_params)),
                np.linalg.cholesky(self._fit_results.cov_params()).T,
            )
            + self.get_parameters().values
        )
        beta_simulated = pd.DataFrame(beta_simulated, columns=self._fit_results.params.index)
        # simulate innovations for the right hand side of the model
        innovation = np.random.normal(
            loc=0, scale=self._fit_results.mse_resid ** 0.5, size=(num_steps, num_simulations)
        )

        # stack observed endogenous series and empty container for future simulations
        # Series of length (nobs + num_steps)
        endog_updated = pd.concat([self._y_train, pd.Series(np.empty(num_steps))], axis=0, ignore_index=True)
        # DataFrame (nobs + num_steps) x num_simulations
        endog_updated = pd.concat([endog_updated] * num_simulations, axis=1)

        # loop over horizon
        for j in range(num_steps):
            transformed_endog = self._transform_all_data(endog=endog_updated.iloc[: self._nobs + j + 1])
            # apply model recursion plus innovation
            temp = 0
            for key, val in transformed_endog.items():
                temp += val.iloc[-1, :] * beta_simulated[key]

            transformed_exog = self._transform_all_data(exog=pd.concat([self._x_train, X]).iloc[: self._nobs + j + 1])
            exog_df = self._convert_transformed_dict_to_frame(transformed=transformed_exog)
            for key in exog_df.columns:
                temp += exog_df[key].iloc[-1] * beta_simulated[key]

            simulation[j] = temp + innovation[j]
            # update out-of-sample endogenous series with simulated value
            endog_updated.loc[self._nobs + j] = simulation[j]

        return pd.DataFrame(simulation, index=X.index)

    def _compute_prediction_quantiles(
        self, num_steps: int, num_simulations: int = None, quantile_levels: List[float] = None, X: pd.DataFrame = None
    ) -> pd.DataFrame:
        simulations = self.simulate(num_steps=num_steps, num_simulations=num_simulations, X=X)
        quantiles = simulations.quantile(np.array(quantile_levels) / 100, axis=1).T
        quantiles.columns = self.get_quantile_names(quantile_levels)
        return quantiles

    def _get_rsquared(self) -> float:
        return self._fit_results.rsquared

    def _get_aic(self) -> float:
        return self._fit_results.aic

    def _get_fitted_values(self) -> pd.Series:
        return self._fit_results.fittedvalues

    def _get_residuals(self) -> pd.Series:
        return self._fit_results.resid

    @staticmethod
    def _transform_data(
        transform: Dict[str, Callable], data: Union[pd.Series, pd.DataFrame] = None
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Transform original data for the use as right-hand-side variables.

        :param data: original data
        :param transform: dictionary with transformation functions
        :return: dictionary with the same keys as in transform dictionary
        """
        return {key: fun(data) for key, fun in transform.items() if data is not None}

    def _transform_all_data(
        self, endog: Union[pd.Series, pd.DataFrame] = None, exog: pd.DataFrame = None
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Transform data according to the model.

        :param endog: endogenous time series or frame (when doing Monte Carlo)
        :param exog: frame of exogenous variables
        :return: dictionary {f(Y_{t,L}), g(X_t)} of transformed endogenous and exogenous.
        Original endogenous variable is dropped from the dictionary to avoid using it as a right-hand-side variable.
        """
        transformed_endog = self._transform_data(data=endog, transform=self._endog_transform)
        transformed_exog = self._transform_data(data=exog, transform=self._exog_transform)
        transformed = {**transformed_endog, **transformed_exog}
        transformed.pop(self.lbl_original_endog, None)
        return transformed

    @staticmethod
    def _convert_transformed_dict_to_frame(transformed: Dict[str, Union[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
        """Convert the dictionary of data into one DataFrame.

        :param transformed: dictionary of transformed data
        :return: DataFrame with transformed data.
        Column names in the resulting frame will be concatenated from the dictionary keys
        and column names of the inputs.
        """
        out = list()
        for key, frame in transformed.items():
            if isinstance(frame, pd.Series):
                out.append(pd.DataFrame({key: frame}))
            elif isinstance(frame, pd.DataFrame):
                out.append(frame.rename(columns={col: "{} {}".format(key, col) for col in frame.columns}))
        if len(out) > 0:
            return pd.concat(out, axis=1)
        else:
            return pd.DataFrame()

    @staticmethod
    def _init_transform(name: str, transform: Dict[str, Callable] = None) -> Optional[Dict[str, Callable]]:
        return {name: lambda x: x} if transform is None else transform

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
