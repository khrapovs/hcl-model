from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd


class TimeSeriesModelArchetype(ABC):
    """Time Series Model Archetype."""

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
        self._fit(y=y, X=X, **kwargs)
        return self

    @abstractmethod
    def _fit(self, y: pd.Series, X: pd.DataFrame = None, **kwargs) -> None:
        pass

    def predict(
        self, num_steps: int, X: pd.DataFrame = None, quantile_levels: List[float] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Forecast the values and prediction intervals

        :param num_steps: number of point in the future that we want to forecast
        :param X: exogenous variables should cover the whole prediction horizon
        :param quantile_levels: list of desired prediction interval levels between 0 and 100 (in percentages).
            If not provided, no confidence interval will be given as output
        :returns: A DataFrame containing values and prediction intervals.

        Example of output from `num_steps=2` and `quantile_levels=[5, 95]`:

        .. code-block:: python

                        rate      q5     q95
            2019-06-07   102      75     127
            2019-06-14   305     206     278
        """
        if X is not None:
            self._x_train = pd.concat([self._x_train, X])
        return self._predict(num_steps=num_steps, X=X, quantile_levels=quantile_levels, **kwargs)

    @abstractmethod
    def _predict(
        self, num_steps: int, X: pd.DataFrame = None, quantile_levels: List[float] = None, **kwargs
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def simulate(
        self, num_steps: int, num_simulations: int, y: pd.Series = None, X: pd.DataFrame = None, **kwargs
    ) -> pd.DataFrame:
        """
        Simulate `num_simulations` realizations of the next `num_steps` values

        :param num_steps: number of points in the future that we want to simulate
        :param num_simulations: number of independent simulations
        :param y: endogenous variables, if not provided the model should use the data provided into the fit() method
        :param X: exogenous variables
        :return: A DataFrame containing simulations
        """

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
                self.lbl_params: self._get_parameters().to_dict(),
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

    def _compute_prediction_quantiles(
        self, num_steps: int, num_simulations: int, quantile_levels: List[float] = None
    ) -> pd.DataFrame:
        """Compute prediction percentiles from simulations.

        :param num_steps: number of points in the future that we want to simulate
        :param num_simulations: number of independent simulations
        :param quantile_levels: quantile levels
        :return: quantiles
        """
        simulations = self.simulate(num_steps=num_steps, num_simulations=num_simulations)
        quantiles = simulations.quantile(np.array(quantile_levels) / 100, axis=1).T
        quantiles.columns = self.get_quantile_names(quantile_levels)
        return quantiles

    @staticmethod
    def _check_exogenous(exog: pd.DataFrame, nobs: int, num_steps: int) -> None:
        """Check that provided exogenous data cover prediction horizon.

        :param nobs: the number of observations
        :param num_steps: the number of steps in the future we want to make forecast of
        :raise RuntimeError:
        """
        if exog is not None and exog.shape[0] < nobs + num_steps:
            raise RuntimeError("Provided exogenous data does not cover the whole prediction horizon!")

    def _get_endog_name(self) -> str:
        return self._y_train.name

    def _get_index_name(self) -> str:
        return self._y_train.index.name

    @staticmethod
    def _get_num_observations(endog: pd.Series = None) -> int:
        return endog.shape[0]

    def _get_out_sample_exog(self, num_steps: int = None) -> Union[pd.DataFrame, None]:
        if self._x_train is not None:
            idx = slice(
                self._get_num_observations(self._y_train), self._get_num_observations(self._y_train) + num_steps
            )
            return self._x_train.iloc[idx]
        else:
            return None

    def _get_in_sample_data(self) -> pd.DataFrame:
        return pd.concat([self._y_train, self._x_train], axis=1)

    def _get_parameters(self) -> pd.Series:
        return self._fit_results.params

    @staticmethod
    def get_quantile_names(quantile_levels: List[float]) -> List[str]:
        return ["prediction_quantile{}".format(x) for x in quantile_levels]
