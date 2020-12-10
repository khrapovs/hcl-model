from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    ):
        super().__init__()
        # model settings
        self._order = order
        self._seasonal_order = seasonal_order
        self._trend = trend
        self._enforce_stationarity = enforce_stationarity
        # statsmodels object with model fit results
        self._fit_results = None
        # empty Series
        self._endog = pd.Series(dtype=float)
        self._exog = None
        self._trend_fit = None

    def fit(self, endog: pd.Series, exog: pd.DataFrame = None, **kwargs):

        # this method updates class variables: `_endog` and `_exog`
        self._endog, self._exog = self._prepare_data(endog=endog, exog=exog)

        self._endog = self._remove_trend(self._endog)

        # fit the parameters using OLS
        self._fit_results = sm.tsa.SARIMAX(
            self._endog,
            exog=self._get_in_sample_exog(self._endog),
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=self._enforce_stationarity,
        ).fit(disp=False)

    def predict(
        self,
        num_steps: int,
        endog: pd.Series = None,
        exog: pd.DataFrame = None,
        quantile_levels: List[float] = None,
        **kwargs
    ) -> pd.DataFrame:

        # this method updates class variables: `_endog` and `_exog`
        self._endog, self._exog = self._prepare_data(endog=endog, exog=exog)

        # get number of endogenous observations
        nobs = self._get_num_observations(self._endog)
        # check that provided exogenous data cover prediction horizon
        self._check_exogenous(exog=self._exog, nobs=nobs, num_steps=num_steps)
        # make sure that the model is estimated
        if self._fit_results is None:
            self.fit(endog=self._endog, exog=self._get_in_sample_exog(self._endog))

        forecast = self._fit_results.get_forecast(
            steps=num_steps, exog=self._get_out_sample_exog(num_steps=num_steps)
        )
        predictions = pd.DataFrame({self._get_endog_name(): forecast.predicted_mean})

        if quantile_levels is not None:
            quantiles = self._compute_prediction_quantiles_exact(
                num_steps=num_steps, quantile_levels=quantile_levels
            )
            predictions = pd.concat([predictions, quantiles], axis=1)

        return self._add_trend(predictions, nobs=nobs)

    def simulate(
        self,
        num_steps: int,
        num_simulations: int,
        endog: pd.Series = None,
        exog: pd.DataFrame = None,
        **kwargs
    ) -> pd.DataFrame:

        # this method updates class variables: `_endog` and `_exog`
        self._endog, self._exog = self._prepare_data(endog=endog, exog=exog)

        self._endog = self._remove_trend(self._endog)

        # get number of endogenous observations
        nobs = self._get_num_observations(self._endog)
        # check that provided exogenous data cover prediction horizon
        self._check_exogenous(exog=self._exog, nobs=nobs, num_steps=num_steps)
        # make sure that the model is estimated
        if self._fit_results is None:
            self.fit(endog=self._endog, exog=self._get_in_sample_exog(self._endog))

        # index for out-of-sample values
        idx = slice(
            self._get_num_observations(self._endog),
            self._get_num_observations(self._endog) + num_steps,
        )
        # initialize out-of-sample model
        sim_model = sm.tsa.SARIMAX(
            pd.Series(index=self._exog.iloc[idx].index, dtype=float),
            exog=self._exog.iloc[idx],
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=self._enforce_stationarity,
        )
        # pass fitted parameters to the model
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

    def _compute_prediction_quantiles_exact(
        self, num_steps: int, quantile_levels: List[float] = None
    ) -> pd.DataFrame:
        """Compute exact prediction percentiles."""
        forecast = self._fit_results.get_forecast(
            steps=num_steps, exog=self._get_out_sample_exog(num_steps=num_steps)
        )
        out = dict()
        for alpha, q_name in zip(
            quantile_levels, self.get_quantile_names(quantile_levels)
        ):
            if alpha < 50:
                out[q_name] = forecast.conf_int(alpha=2 * alpha / 100).iloc[:, 0]
            else:
                out[q_name] = forecast.conf_int(alpha=2 * (100 - alpha) / 100).iloc[
                    :, 1
                ]
        return pd.DataFrame(out)

    def _remove_trend(self, endog: pd.Series) -> pd.Series:
        if self._trend == "n":
            return endog
        else:
            name = endog.name
            trend = sm.tsa.tsatools.add_trend(
                self._endog, trend=self._trend, prepend=False
            )
            self._trend_fit = sm.OLS(endog, trend.iloc[:, 1:]).fit()
            endog -= self._trend_fit.fittedvalues
            return endog.rename_axis(name)

    def _add_trend(self, df: pd.DataFrame, nobs: int) -> pd.DataFrame:
        if self._trend == "n":
            return df
        else:
            exog = sm.tsa.tsatools.add_trend(
                pd.Series(np.ones(nobs + df.shape[0])),
                trend=self._trend,
                prepend=False,
                has_constant="add",
            )
            return df.apply(
                lambda x: x + self._trend_fit.predict(exog.iloc[-df.shape[0] :, 1:])
            )
