from typing import List, Union, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from hcl_model.time_series_model_archetype import TimeSeriesModelArchetype


class HandCraftedLinearModel(TimeSeriesModelArchetype):
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

    def __init__(self, endog_transform: dict = None, exog_transform: dict = None):
        super().__init__()
        self._endog_transform = endog_transform
        self._exog_transform = exog_transform
        # statsmodels object with model fit results
        self._fit_results = None
        # empty Series
        self._endog = pd.Series()
        self._exog = pd.DataFrame()

    def fit(self, endog: pd.Series, exog: pd.DataFrame = None, weights: Union[Sequence, float] = 1.0, **kwargs):

        # this method updates class variables: `_endog` and `_exog`
        self._endog, self._exog = self._prepare_data(endog=endog, exog=exog)

        # fit the parameters using OLS
        self._fit_results = sm.WLS(endog=self._endog, missing='drop',
                                   exog=self._transform_data(endog=self._endog,
                                                             exog=self._get_in_sample_exog(self._endog)),
                                   weights=weights
                                   ).fit()

    def predict(self, num_steps: int, endog: pd.Series = None, exog: pd.DataFrame = None,
                weights: Union[Sequence, float] = 1.0,
                quantile_levels: List[float] = None, num_simulations: int = None, **kwargs) -> pd.DataFrame:

        # this method updates class variables: `_endog` and `_exog`
        self._endog, self._exog = self._prepare_data(endog=endog, exog=exog)

        # get number of endogenous observations
        nobs = self._get_num_observations(self._endog)
        # check that provided exogenous data cover prediction horizon
        self._check_exogenous(exog=exog, nobs=nobs, num_steps=num_steps)
        # make sure that the model is estimated
        if self._fit_results is None:
            self.fit(endog=self._endog, exog=self._get_in_sample_exog(self._endog), weights=weights)

        # stack observed endogenous series and empty container for future predictions
        endog_updated = self._endog.append(pd.Series(np.empty(num_steps), name=self._get_endog_name()),
                                           ignore_index=True)

        # loop over horizon
        for j in range(num_steps):
            # take the last observation of the transformed data
            rhs_vars = self._transform_data(endog=endog_updated[:nobs + j + 1],
                                            exog=self._exog.iloc[:nobs + j + 1]).iloc[-1, :]
            # update out-of-sample endogenous series with predicted value
            endog_updated.iloc[nobs + j] = np.dot(rhs_vars, self._get_parameters())

        # cut off forecasts from past observations
        predictions = endog_updated.iloc[nobs:].to_frame()

        if quantile_levels is not None:
            quantiles = self._compute_prediction_quantiles(num_steps=num_steps, num_simulations=num_simulations,
                                                           quantile_levels=quantile_levels)
            predictions = pd.concat([predictions.reset_index(drop=True), quantiles], axis=1)

        return predictions

    def simulate(self, num_steps: int, num_simulations: int, endog: pd.Series = None, exog: pd.DataFrame = None,
                 weights: Union[Sequence, float] = 1.0, **kwargs) -> pd.DataFrame:

        # this method updates class variables: `_endog` and `_exog`
        self._endog, self._exog = self._prepare_data(endog=endog, exog=exog)

        # get number of endogenous observations
        nobs = self._get_num_observations(self._endog)
        # check that provided exogenous data cover prediction horizon
        self._check_exogenous(exog=self._exog, nobs=nobs, num_steps=num_steps)
        # make sure that the model is estimated
        if self._fit_results is None:
            self.fit(endog=self._endog, exog=self._get_in_sample_exog(self._endog), weights=weights)
        # get number of parameters
        num_params = self._get_parameters().shape[0]

        # initialize empty container for simulations
        simulation = np.empty((num_steps, num_simulations))

        # simulate num_simulations parameters from its multivariate normal distribution
        # Cholesky decomposition returns only the lower triangular part,
        # so it has to be transposed before multiplication
        beta_simulated = (np.dot(np.random.normal(loc=0, scale=1, size=(num_simulations, num_params)),
                                 np.linalg.cholesky(self._fit_results.cov_params()).T)
                          + self._get_parameters().values)

        # simulate innovations for the right hand side of the model
        innovation = np.random.normal(loc=0, scale=self._fit_results.mse_resid ** .5, size=(num_steps, num_simulations))

        # TODO: the outer loop should be removed!
        # loop over simulations
        for i in range(num_simulations):

            # stack observed endogenous series and empty container for future simulations
            endog_updated = self._endog.append(pd.Series(np.empty(num_steps)), ignore_index=True)

            # loop over horizon
            for j in range(num_steps):
                # take the last observation of the transformed data
                rhs_vars = self._transform_data(endog=endog_updated.iloc[:nobs + j + 1],
                                                exog=self._exog.iloc[:nobs + j + 1]).iloc[-1, :]
                # apply model recursion plus innovation
                simulation[j, i] = np.dot(rhs_vars, beta_simulated[i]) + innovation[j, i]
                # update out-of-sample endogenous series with simulated value
                endog_updated[nobs + j] = simulation[j, i]

        return pd.DataFrame(simulation)

    def _get_rsquared(self) -> float:
        return self._fit_results.rsquared

    def _get_aic(self) -> float:
        return self._fit_results.aic

    def _get_fitted_values(self) -> pd.Series:
        return self._fit_results.fittedvalues

    def _get_residuals(self) -> pd.Series:
        return self._fit_results.resid

    def _transform_endog(self, endog: pd.Series = None) -> pd.DataFrame:
        """Transform series of endogenous values to use as regressors."""
        if self._endog_transform is not None:
            endog_transformed = pd.DataFrame()
            for col_name, single_transform in self._endog_transform.items():
                endog_transformed[col_name] = endog.transform(single_transform)
            return endog_transformed
        else:
            return pd.DataFrame()

    def _transform_exog(self, exog: pd.DataFrame = None) -> pd.DataFrame:
        """Transform exogenous variables to use as regressors."""
        if self._exog_transform is not None:
            return exog.transform(self._exog_transform)
        else:
            return exog

    def _transform_data(self, endog: pd.Series = None, exog: pd.DataFrame = None) -> pd.DataFrame:
        """Transform data according to the model.

        Return [f(Y_{t,L}), g(X_t)]. This is used in OLS estimation.

        """
        endog_transformed = self._transform_endog(endog=endog)
        if endog_transformed.shape[0] == 0:
            return self._transform_exog(exog=exog)
        else:
            return pd.concat([endog_transformed, self._transform_exog(exog=exog)], axis=1)
