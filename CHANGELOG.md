## Changelog

### 0.3.5 (2020-12-10)
- Use `median_abs_deviation` instead of deprecated `median_absolute_deviation`. This requires `scipy>=1.5`.
- Use `dates.isocalendar().week` instead of deprecated `dates.week`. This requires `pandas>=1.1`.
- Silence warning about default `dtype` in empty `Series`. Now `dtype=float`.

### 0.3.4 (2020-10-21)
- Change the way seasonal outliers are detected. Instead of looking at percentage changes, the focus now is at the absolute deviation from the exponentially weighted moving average.

### 0.3.3 (2020-07-24)
- Limit the version of `skyfield` package to 1.22 due to an error `AttributeError: module 'datetime' has no attribute 'combine'`.

### 0.3.2 (2020-06-05)
- Require `workalendar` version to be greater than 10.0.0 due to a rename of `IsoRegistry.get_calendar_class()` into `IsoRegistry.get()`.
 
### 0.3.1 (2020-01-27)
- Fix the bug in automatic dummy detection. Instead of taking weeks from sorted normalized series of changes, they were taken from some other intermediate result. 

### 0.3 (2020-01-10)
- Add several utility functions that help construct exogenous regressors based on a calendar. These functions are accessible through `construct_calendar_exogenous`. 
- Add utility function to create decayed weights for weighted OLS estimation.

### 0.2.1 (2020-01-21)

- Add parameter dictionary to model summary output.

### 0.2 (2020-01-10)

- Speed up HCL simulation and, as a consequence, prediction interval computation by Monte Carlo loop.
 
### 0.1 (2019-11-28)

- Move HCL model class from popeye repository