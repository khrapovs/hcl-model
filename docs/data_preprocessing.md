# Data preprocessing methods

In the following, we briefly describe different implemented methods for data preprocessing.

Basic notation: Let $Y=(Y_t)_{1,\ldots,t}$ be the time series and $Y_t$ the observation at time point t.

## Outlier identification and correction

For identification of outliers we do the following. 

1. Smooth the time series $Y$ by applying a rolling median with a fixed window size and then directly apply a exponential weighted moving average. Let $\hat{Y}$ be the smoothed time series with values $\hat{Y}_t$ at time point t.
2. Compute the residuals $R_t = Y_t - \hat{Y}_t$ for all $t\in\{1,\ldots,T\}$. Let $R = (R_t)_{1,\ldots,T}$ be the residual time series.
3. Compute the inter quantile range $iqr = q_{0.75}(R) - q_{0.25}(R)$, which is the difference between the $0.75$ quantile and the $0.25$ quantile of the residual series.
4. Compute the upper limit and lower limit time series, namely $UL = (UL_t)_{1,\ldots,T}$ with $UL_t = \hat{Y}_t + S \cdot iqr$ and $LL = (LL_t)_{1,\ldots,T}$ with $LL_t = \hat{Y}_t - S \cdot iqr$. Here, $S > 0$ is a fixed value, the so called six sigma multiplier.
5. The identified potential outliers are all values $(Y_t)_{t \in I_{outlier}}$ with $I_{outlier}=\{i \in \{1, \ldots, T\}| Y_t > UL_t\text{ or }Y_t < LL_t\}$

If we identify any observation as potential outlier, then we correct the outliers by one of the following methods.

1. Trim method: 
  - For all values $Y_t$ with $Y_t > UL_t$, set $Y_t = UL_t$ 
  - For all values $Y_t$ with $Y_t < LL_t$, set $Y_t = LL_t$ 

2. Interpolation method: 
  - For all values $Y_t$ with $t \in I_{outlier}$, set $Y_t = NA$
  - Apply a linear interpolation in order to impute the NA values
