from __future__ import annotations

from typing import List

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tsatools import add_trend

from hcl_model.time_series_model_archetype import TimeSeriesModelArchetype


class SARIMAXModel(TimeSeriesModelArchetype):
    """
    SARIMAX model

    Wrapper around `statsmodels implementation
    <http://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX>`_.

    See documentation of `statsmodels.tsa.statespace.sarimax.SARIMAX`
    """

    def __init__(
        self,
        order: tuple = (0, 0, 0),
        seasonal_order: tuple = (0, 0, 0, 0),
        trend: str = "c",
        enforce_stationarity: bool = True,
    ) -> None:
        super().__init__()
        self._order = order
        self._seasonal_order = seasonal_order
        self._trend = trend
        self._enforce_stationarity = enforce_stationarity
        self._trend_fit = None

    def _fit(self, y: pd.Series, X: pd.DataFrame = None, **kwargs) -> None:
        self._y_train = self._remove_trend(self._y_train)
        self._fit_results = SARIMAX(
            self._y_train,
            exog=self._x_train,
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=self._enforce_stationarity,
        ).fit(disp=False)

    def _predict(
        self, num_steps: int, X: pd.DataFrame = None, quantile_levels: List[float] = None, **kwargs
    ) -> pd.DataFrame:
        nobs = self._get_num_observations(self._y_train)
        self._check_exogenous(exog=self._x_train, nobs=nobs, num_steps=num_steps)
        forecast = self._fit_results.get_forecast(steps=num_steps, exog=self._get_out_sample_exog(num_steps=num_steps))
        predictions = pd.DataFrame(forecast.predicted_mean.rename(self._get_endog_name())).rename_axis(
            index=self._y_train.index.name
        )

        if quantile_levels is not None:
            quantiles = self._compute_prediction_quantiles_exact(num_steps=num_steps, quantile_levels=quantile_levels)
            predictions = pd.concat([predictions, quantiles], axis=1)

        return self._add_trend(df=predictions)

    def simulate(
        self, num_steps: int, num_simulations: int, y: pd.Series = None, X: pd.DataFrame = None, **kwargs
    ) -> pd.DataFrame:
        if X is not None:
            self._x_train = pd.concat([self._x_train, X])
        self._y_train = self._remove_trend(self._y_train)
        nobs = self._get_num_observations(self._y_train)
        self._check_exogenous(exog=self._x_train, nobs=nobs, num_steps=num_steps)
        if self._fit_results is None:
            self.fit(y=self._y_train, exog=self._x_train)

        idx = slice(
            self._get_num_observations(self._y_train),
            self._get_num_observations(self._y_train) + num_steps,
        )
        sim_model = SARIMAX(
            pd.Series(index=self._x_train.iloc[idx].index, dtype=float),
            exog=self._x_train.iloc[idx],
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=self._enforce_stationarity,
        )
        sim_model = sim_model.filter(self._fit_results.params)

        # TODO: check simulation output for different model. I am not sure it is correct without initial_sate.
        simulation = {i: sim_model.simulate(num_steps) for i in range(num_simulations)}
        return pd.DataFrame(simulation)

    def _get_aic(self) -> float:
        return self._fit_results.aic

    def _get_fitted_values(self) -> pd.Series:
        return self._fit_results.fittedvalues

    def _get_residuals(self) -> pd.Series:
        return self._fit_results.resid

    def _compute_prediction_quantiles_exact(self, num_steps: int, quantile_levels: List[float] = None) -> pd.DataFrame:
        forecast = self._fit_results.get_forecast(steps=num_steps, exog=self._get_out_sample_exog(num_steps=num_steps))
        out = dict()
        for alpha, q_name in zip(quantile_levels, self.get_quantile_names(quantile_levels)):
            if alpha < 50:
                out[q_name] = forecast.conf_int(alpha=2 * alpha / 100).iloc[:, 0]
            else:
                out[q_name] = forecast.conf_int(alpha=2 * (100 - alpha) / 100).iloc[:, 1]
        return pd.DataFrame(out).rename_axis(index=self._y_train.index.name)

    def _remove_trend(self, endog: pd.Series) -> pd.Series:
        if self._trend == "n":
            return endog
        else:
            name = endog.name
            trend = add_trend(self._y_train, trend=self._trend, prepend=False)
            self._trend_fit = OLS(endog, trend.iloc[:, 1:]).fit()
            endog -= self._trend_fit.fittedvalues
            return endog.rename(name)

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._trend == "n":
            return df
        else:
            exog = add_trend(
                pd.concat([self._y_train, df[self._y_train.name]]),
                trend=self._trend,
                prepend=False,
                has_constant="add",
            )
            return df.apply(lambda x: x + self._trend_fit.predict(exog.iloc[-df.shape[0] :, 1:]))
