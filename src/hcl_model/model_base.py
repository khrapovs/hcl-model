from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from hcl_model.utils.check_x_y import check_X_y


class ModelBase(ABC):
    """Time Series Model base class."""

    lbl_r2 = "rsquared"
    lbl_aic = "aic"
    lbl_mape = "mape"
    lbl_resid_mean = "error_mean"
    lbl_resid_std = "error_std"
    lbl_resid_skewness = "error_skewness"
    lbl_resid_kurtosis = "error_kurtosis"
    lbl_params = "parameters"
    fit_results_: Any

    def fit(self, X: Optional[pd.DataFrame], y: Union[pd.Series, np.ndarray]) -> ModelBase:
        """
        Fit the model using some provided training data.

        :param X: optional exogenous explanatory variables
        :param y: pd.Series or np.ndarray, endogenous variable
            In case of np.ndarray its index is inherited from `X`.
            In case of np.ndarray and if `X` is `None`, then `TypeError` is risen.
        """
        self.x_train_, self.y_train_ = check_X_y(X=X, y=y)
        self._fit()
        return self

    @abstractmethod
    def _fit(self) -> None:
        """Core fit method."""

    def predict(
        self,
        X: Optional[pd.DataFrame],
        num_steps: int = None,
        quantile_levels: List[float] = None,
        num_simulations: int = None,
    ) -> pd.DataFrame:
        """
        Forecast the values and prediction intervals

        :param X: optional pd.DataFrame
            exogenous variables should cover the whole prediction horizon
        :param num_steps: optional int
            Number of point in the future that we want to forecast.
            If None, and X is not None, it will be computed as the number of observations in X
        :param quantile_levels: list of desired prediction interval levels between 0 and 100 (in percentages).
            If not provided, no confidence interval will be given as output
        :param num_simulations: number of simulations for simulation-based prediction intervals
        :returns: A DataFrame containing values and prediction intervals.

        Example of output from `num_steps=2` and `quantile_levels=[5, 95]`:

        .. code-block:: python

                        rate      q5     q95
            2019-06-07   102      75     127
            2019-06-14   305     206     278
        """
        if num_steps is None:
            if X is not None:
                num_steps = X.shape[0]
            else:
                raise ValueError("Either `num_steps` or `X` should be provided")
        self._check_exogenous(exog=X, nobs=self._nobs, num_steps=num_steps)
        predictions = self._predict(num_steps=num_steps, X=X, quantile_levels=quantile_levels)
        if quantile_levels is not None:
            quantiles = self._compute_prediction_quantiles(
                num_steps=num_steps, quantile_levels=quantile_levels, X=X, num_simulations=num_simulations
            )
            predictions = pd.concat([predictions, quantiles], axis=1)
        return self._add_trend(df=predictions)

    @abstractmethod
    def _predict(self, num_steps: int, X: pd.DataFrame = None, quantile_levels: List[float] = None) -> pd.DataFrame:
        """Core predict method."""

    def simulate(self, num_steps: int, num_simulations: int, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        Simulate `num_simulations` realizations of the next `num_steps` values

        :param num_steps: number of points in the future that we want to simulate
        :param num_simulations: number of independent simulations
        :param X: exogenous variables
        :return: A DataFrame containing simulations
        """
        self._check_exogenous(exog=X, nobs=self._nobs, num_steps=num_steps)
        return self._simulate(num_steps=num_steps, num_simulations=num_simulations, X=X)

    @abstractmethod
    def _simulate(self, num_steps: int, num_simulations: int, X: pd.DataFrame = None) -> pd.DataFrame:
        """Core simulate method."""

    @abstractmethod
    def _compute_prediction_quantiles(
        self, num_steps: int, num_simulations: int = None, quantile_levels: List[float] = None, X: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Compute prediction quantiles."""

    def summary(self) -> pd.Series:
        """A summary of in-sample model performance KPIs.

        :return: A series of model fit KPIs.
        """
        return pd.Series(
            {
                self.lbl_r2: self._get_rsquared(),
                self.lbl_aic: self._get_aic(),
                self.lbl_mape: self._get_mape(),
                self.lbl_resid_mean: self._get_residual_mean(),
                self.lbl_resid_std: self._get_residual_std(),
                self.lbl_resid_skewness: self._get_residual_skewness(),
                self.lbl_resid_kurtosis: self._get_residual_kurtosis(),
                self.lbl_params: self.get_parameters().to_dict(),
            }
        )

    @abstractmethod
    def _get_aic(self) -> float:
        """Akaike Information Criterion of a model fit.

        :return: AIC statistic as a float.
        """

    @abstractmethod
    def _get_fitted_values(self) -> pd.Series:
        """get fitted values

        :return: One point ahead forecasts on the in-sample period which are the "fitted values" in time series context.
        """

    @abstractmethod
    def _get_residuals(self) -> pd.Series:
        """Get residuals

        :return: Residuals of one point ahead forecasts on the in-sample period.
        """

    def _get_mape(self) -> float:
        """Mean absolute percentage error on in-sample.

        :return: Error as a float in percent.
        """
        return ((self.y_train_ - self._get_fitted_values()) / self.y_train_).abs().mean() * 100

    def _get_rsquared(self) -> float:
        """Mean absolute percentage error on in-sample.

        :return: Error as a float in percent.
        """
        return np.corrcoef(self._get_fitted_values(), self.y_train_.values)[0, 1] ** 2

    def _get_residual_moment(self, degree: int = 1, center_first: bool = False) -> float:
        """Get residual moment.

        :return: (Centered) moments of model fit residuals. Note that residuals can be biased on average.
        """
        if center_first:
            resid = self._get_residuals() - self._get_residuals().mean()
        else:
            resid = self._get_residuals()
        return resid.pow(degree).mean()

    def _get_residual_mean(self) -> float:
        return self._get_residual_moment(degree=1, center_first=False)

    def _get_residual_std(self) -> float:
        return self._get_residuals().std()

    def _get_residual_skewness(self) -> float:
        return self._get_residual_moment(degree=3, center_first=True) / (self._get_residual_std() ** 3)

    def _get_residual_kurtosis(self) -> float:
        return self._get_residual_moment(degree=4, center_first=True) / (self._get_residual_std() ** 4)

    def _check_exogenous(self, nobs: int, num_steps: int, exog: pd.DataFrame = None) -> None:
        """Check that provided exogenous data cover prediction horizon.

        :param nobs: the number of observations
        :param num_steps: the number of steps in the future we want to make forecast of
        :param exog: exogenous data
        :raise RuntimeError:
        """
        x_train_and_test = pd.concat([self.x_train_, exog]) if exog is not None else self.x_train_
        if x_train_and_test is not None and x_train_and_test.shape[0] < nobs + num_steps:
            raise RuntimeError("Provided exogenous data does not cover the whole prediction horizon!")

    def _get_endog_name(self) -> str:
        return self.y_train_.name

    def _get_index_name(self) -> str:
        return self.y_train_.index.name

    @property
    def _nobs(self) -> int:
        return self.y_train_.shape[0]

    def get_parameters(self) -> pd.Series:
        """Get model parameters.

        :return: A series of model parameters.
        """
        return self.fit_results_.params

    @staticmethod
    def get_quantile_names(quantile_levels: List[float]) -> List[str]:
        return ["prediction_quantile{}".format(x) for x in quantile_levels]

    @abstractmethod
    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend."""
