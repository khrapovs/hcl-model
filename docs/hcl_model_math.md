# Hand Crafted Linear Model

## Motivation

In this document we present a flexible class of autoregressive time series models with exogenous variables. There are several facts that motivate the new model:

1. **SARIMAX models are very slow to fit.** In order to fit parameters of a single SARIMAX model, one needs to find an optimum of a highly non-linear objective function (likelihood, or something even more complicated in the case of Kalman filter based algorithm). For producing one forecast for one time series this is definitely not an issue. But when it becomes necessary to redo the fitting hundreds or even thousands of times (e.g. cross-validation for multiple time series), it becomes a heavy burden on computation time. Finally, exploration of alternative model specifications becomes very costly in a fast development environment.

1. **SARIMAX models are not flexible enough.** The definition of $AR(p)$ model allows inclusion of only last $p$ lags in the model recursion. It does not allow for a combination of, say, lags 1, 4, 26, and 52. $SARIMA(p,d,q)\times(P,D,Q)[s]$ implicitly includes lags $s$ and some more. This may be sufficient to cover cover dependence on, for example, first $p$ lags and more distant lag $s$, but still nothing in between.

1. **SARIMAX may be an overkill in forecasting.** The optimal forecast derived from $SARIMAX$ model includes only lags of the dependent variable and exogenous variables, but not moving average component since its expectation is zero anyway. The moving average component affects the point forecast only indirectly by changing the parameters of the model. The main purpose of the MA component is to address autocorrelation in the residuals and, therefore, properly model their covariance matrix.

## Model Definition

The model in its most general formulation is simply a multivariate linear regression:
$$
Y_{t+m}=f\left(Y_{t,L}\right)\gamma_m^\prime+g\left(X_{t+m}\right)\beta_m^\prime+\varepsilon_{t+m},\quad\varepsilon_{t+m}\sim IID(0,\sigma_m),\quad m\in\{1,2,\ldots\},
$$
where $Y_{t,L}$ and $X_t$ are random variables observed at time $t$ and $t+m$, respectively. Functions $f$ and $g$ are vector-valued deterministic functions.

The purpose of the subindex $m$ is to provide a separate model for distant observations ($m>1$). This could also be achieved by an autoregressive process with $p>m$ lags and first $m$ autoregressive coefficients restricted to zero. But this would complicate estimation and implementation of the model. In our set up parameters are unrestricted.

We can state immediately, that parameters of this model can be estimated using lightning fast OLS versus slow non-linear optimization techniques like MLE or Kalman filter used for $SARIMAX$. Here the vector $X_t$ has $k$ elements. Vector $Y_{t,L}$ is the arbitrary collection of lags of the dependent variable $Y_t$. The lags can be specified by a $l$-vector $L=\left[s_1,\ldots,s_p\right]$ for $s_i\in\left\{0,1,2,\ldots\right\}$. Given this vector of lags we can define
$$
Y_{t,L}=\left[Y_{t-s_1},\ldots,Y_{t-s_p}\right].
$$

In particular, for $h=1$, $f(x)=x$, $g(x)=0$,
- if $L=\left[0,1,\ldots,p-1\right]$, then we reduce the model to $ARX(p)$.
- if $L=\left[0,p-1,p\right]$, then we reduce the model to $SARIMAX(1,0,0)\times(1,0,0)[p]$.

Note that the dependent variable can be $h$ periods away from the last observable explanatory data. This allows to have several independent models for each specific horizon $h$.

The idea behind deterministic transformations $f$ and $g$ is to allow, for example, for exponential moving average of $Y_t$ as an explanatory variable.

Here and below we assume that we have the data $\left\{Y_t,X_t\right\}_{t=1}^T$, where the first component is a scalar variable we aim to forecast, and the second is a vector of exogenous variables potentially relevant in forecasting $Y_t$.

## Model Estimation

Model parameters, $\beta_m$ are $\gamma_m$ estimated using standard OLS. Variance parameter $\sigma_m$ can be estimated using sample standard deviation of model residuals. The corresponding estimates are denoted by $\hat{\beta}_m$, $\hat{\gamma}_m$, and $\hat{\sigma}_m$.

## Forecasting

In general, the forecast is computed from the following recursion:

$$
Y^f_{T+h}=f\left(Y^f_{T-m+h,L}\right)\hat{\gamma}_m^\prime+g\left(X^f_{T+h}\right)\hat{\beta}_m^\prime,\quad m,h\geq1.
$$

Note that $Y^f_{T-m+h,L}=\left[Y^f_{T-m+h-s_1},\ldots,Y^f_{T-m+h-s_p}\right]$ which means $Y^f_{T-m+h-s_i}=Y_{T-m+h-s_i}$ is observed if $h\leq m+s_i$, otherwise it has to be computed using the forecasting recursion. Exogenous forecast $X^f_{T+h}$ has to be obtained from a separate model.

## Prediction Intervals

The goal is, given $\alpha\in[0,1]$, to provide an interval $C(h,m,\alpha)=\left[c_d,c_u\right]$ such that $P\left[Y_{T+h}^f\in C(h,m,\alpha)\right]=\alpha$. We propose to use parametric bootstrap.

1. Simulate $\left\{\varepsilon_{T+i}^{(s)}\right\}_{s=1,i=1}^{S,h}$ from $N(0,\hat{\sigma}_m)$, where $S$ is the number of simulations. For each $s\in S$ simulate the vector of parameters $\left\{\left[\beta^{(s)}_m,\gamma^{(s)}_m\right]\right\}_{s=1}^S$ from estimated asymptotic distribution $N\left(\left[\hat{\beta}_m,\hat{\gamma}_m\right],\hat{V}\left[\hat{\beta}_m,\hat{\gamma}_m\right]\right)$.

2. Given parameter and innovations simulations, produce $S$ samples of $h$ horizon realizations $\left\{Y_{T+h}^{f(s)}\right\}_{s=1}^S$ iteratively:

    $$
    Y_{T+h}^{(s)}=f\left(Y_{T-m+h,L}^{(s)}\right)\hat{\gamma}_m^{(s)\prime}+g\left(X^{(s)}_{T+h}\right)\hat{\beta}_m^{(s)\prime}+\varepsilon^{(s)}_{T+h}.
    $$

4. Compute empirical quantiles $\hat{c}_d$ and $\hat{c}_u$ of $\left\{Y_{T+h}^{f(s)}\right\}_{s=1}^S$.

## Model Mixing

Suppose we have two models. One model is designed to produce the best forecasts for short term horizon, and the second for the long term horizon. The problem is that the forecast line may not be "connected" at the point where short term "end" and the long term "begins". The proposed solution is to mix forecasts using a smooth transition function. In general, the mixed forecast is
$$
Y_{t+h}^f=\alpha_hY_{t+h}^{f1}+(1-\alpha_h)Y_{t+h}^{f2},
$$
where $f1$ marks short term forecast and $f2$ marks long term forecast. $\alpha_h$ is a time dependent weighting function. It is natural to require that $\alpha_1=1$, and $\alpha_H=0$, where 1 and $H$ are extreme short and long term, respectively. In between, we may use, for example, a logistic function:

$$
\alpha_h=1 - \frac{1}{1+\exp\left\{-a(h-H/2)\right\}},
$$

where $a$ is a nuisance parameter that stretches the logistic function appropriately. Ideally, this parameter should be chosen to minimize cross-validation MAPE.

## Model Components

In this section we will describe model components defined by a specific choice of functions $f$ and $g$, as well as vectors $Y_{t,L}$ and $X_t$.

### Linear Trend

One of the features of a non-stationary time series is a trend. Here we outline only one possible choice for the trend function: linear trend.

The easiest way to introduce a linear trend in our framework is to add two columns to the $T\times k$ matrix $X=\left[X_1^\prime,\ldots,X_T^\prime\right]^\prime$: constant one and time. In particular, let $X_{1,t}=1$ and $X_{2,t}=t$. The function $g$ is simply $g(x)=x$.

### Seasonality

A seasonal regressor is a deterministic vector valued function of time, such that $g(t)=g(t+p)$ for all $t$, and $p>1$ is a given period length. For example, it can be that $p=52$ for weekly data or $p=12$ for monthly data.

Below we introduce several options to model seasonal effects.

#### Seasonality with Dummies

The vector of exogenous regressors is
$$
X_t=\left[\mathbf{1}(t\in A_1),\ldots,\mathbf{1}(t\in A_p)\right],
\quad A_i=\{i,i+p,i+2p,\ldots\},
$$
and $g(x)=x$.

If a constant is one of the additional model components, then one of the indicators in $X_t$ should be dropped to avoid multicollinearity in OLS estimation.

#### Seasonality with Splines

The vector of exogenous regressors is
$$
X_t=\left[s_1(t/p),\ldots,s_d(t/p)\right],
$$
such that the restricted weighted sum of these components is periodic. The dimension $d$ (selected by a statistician) controls the degree of freedom. The so-called basis $s_j$ is a deterministic function and consists of cubic splines. Also, $g(x)=x$.

The idea is that the spline is sufficiently flexible so that one needs only a few dimensions to adequately fit any random periodic process. For an alternative dummy representation of seasonal effect this approach is preferred due to a small number of parameters to estimate.

### Weighted Average (Smoothing) Components

In many examples the assumption of a constant or constantly changing mean (linear trend) is too restrictive. Here we propose to use a non-periodic smoother as a component of vector-valued function $f$. The idea is to add an explanatory regressor that changes with the dependent variable, but much slower, and represents its "local" level over the not-so-distant past.

The scalar deterministic smoother function $f$ depends on $n$ past observations of the dependent variable and a nuisance parameter $\alpha$ (possibly a vector):
$$
f\left(Y_{t-1},\ldots,Y_{t-n},\alpha\right).
$$
A trivial choice of such a function $f$ is a weighted moving average:
$$
f = \frac{\sum_{i=0}^{n-1}w(i,\alpha)Y_{t-i}}{\sum_{i=0}^{n-1}w(i,\alpha)}
=\sum_{i=0}^{n-1}\tilde{w}(i,\alpha)Y_{t-i},
\quad w(i,\alpha)\geq0.
$$

### Structural Changes

Suppose we have determined that the time series is structurally unstable. Suppose also that we were able to find a set of structural breakpoints $B=\left\{t_1,\ldots,t_b\right\}$, where $t_i\in(1,T)$ for each $i$. If we believe that the trend was one part of structural instability, then we may fit a piece-wise trend instead of a trend common for the whole observable sample. In particular, we consider $b+1$ intervals between the breakpoints and fit a trend model for each interval separately:
$$
Y_t=D_{t,i}+\varepsilon_t,\quad\varepsilon_t\sim IID(0,\sigma),\quad t\in\left[t_{i-1},t_i\right],\quad i=1,\ldots,b,\quad t_0=1.
$$
Out-of-sample forecast is then a simple extension of the last available trend line:
$$
Y^f_{T+h}=D^f_{T+h,b}.
$$
