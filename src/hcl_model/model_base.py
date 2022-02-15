from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


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

    def __init__(self) -> None:
        self._fit_results = None
        self._y_train = None
        self._x_train = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None, **kwargs):
        """
        Fit the model using some provided training data.

        :param y: endogenous variable
        :param X: exogenous explanatory variables
        """
        self._y_train = y.copy()
        self._x_train = X.copy() if X is not None else None
        self._fit(**kwargs)
        return self

    @abstractmethod
    def _fit(self, **kwargs) -> None:
        pass

    def predict(
        self,
        num_steps: int,
        X: pd.DataFrame = None,
        quantile_levels: List[float] = None,
        num_simulations: int = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Forecast the values and prediction intervals

        :param num_steps: number of point in the future that we want to forecast
        :param X: exogenous variables should cover the whole prediction horizon
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
        self._check_exogenous(exog=X, nobs=self._nobs, num_steps=num_steps)
        predictions = self._predict(num_steps=num_steps, X=X, quantile_levels=quantile_levels, **kwargs)
        if quantile_levels is not None:
            quantiles = self._compute_prediction_quantiles(
                num_steps=num_steps, quantile_levels=quantile_levels, X=X, num_simulations=num_simulations
            )
            predictions = pd.concat([predictions, quantiles], axis=1)
        return self._add_trend(df=predictions)

    @abstractmethod
    def _predict(
        self, num_steps: int, X: pd.DataFrame = None, quantile_levels: List[float] = None, **kwargs
    ) -> pd.DataFrame:
        pass

    def simulate(self, num_steps: int, num_simulations: int, X: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Simulate `num_simulations` realizations of the next `num_steps` values

        :param num_steps: number of points in the future that we want to simulate
        :param num_simulations: number of independent simulations
        :param X: exogenous variables
        :return: A DataFrame containing simulations
        """
        self._check_exogenous(exog=X, nobs=self._nobs, num_steps=num_steps)
        return self._simulate(num_steps=num_steps, num_simulations=num_simulations, X=X, **kwargs)

    @abstractmethod
    def _simulate(self, num_steps: int, num_simulations: int, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def _compute_prediction_quantiles(
        self, num_steps: int, num_simulations: int = None, quantile_levels: List[float] = None, X: pd.DataFrame = None
    ) -> pd.DataFrame:
        pass

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
        return ((self._y_train - self._get_fitted_values()) / self._y_train).abs().mean() * 100

    def _get_rsquared(self) -> float:
        """Mean absolute percentage error on in-sample.

        :return: Error as a float in percent.
        """
        return np.corrcoef(self._get_fitted_values(), self._y_train.values)[0, 1] ** 2

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
        x_train_and_test = pd.concat([self._x_train, exog]) if exog is not None else self._x_train
        if x_train_and_test is not None and x_train_and_test.shape[0] < nobs + num_steps:
            raise RuntimeError("Provided exogenous data does not cover the whole prediction horizon!")

    def _get_endog_name(self) -> str:
        return self._y_train.name

    def _get_index_name(self) -> str:
        return self._y_train.index.name

    @property
    def _nobs(self) -> int:
        return self._y_train.shape[0]

    def get_parameters(self) -> pd.Series:
        """Get model parameters.

        :return: A series of model parameters.
        """
        return self._fit_results.params

    @staticmethod
    def get_quantile_names(quantile_levels: List[float]) -> List[str]:
        return ["prediction_quantile{}".format(x) for x in quantile_levels]

    @abstractmethod
    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
