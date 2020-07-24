## Changelog

#### 0.3.3

- Limit the version of `skyfield` package to 1.22 due to an error `AttributeError: module 'datetime' has no attribute 'combine'`.

#### 0.3.2

- Require `workalendar` version to be greater than 10.0.0 due to a rename of `IsoRegistry.get_calendar_class()` into `IsoRegistry.get()`.
 
#### 0.3.1

- Fix the bug in automatic dummy detection. Instead of taking weeks from sorted normalized series of changes, they were taken from some other intermediate result. 

### 0.3

- Add several utility functions that help construct exogenous regressors based on a calendar. These functions are accessible through `construct_calendar_exogenous`. 
- Add utility function to create decayed weights for weighted OLS estimation.

#### 0.2.1

- Add parameter dictionary to model summary output.

### 0.2

- Speed up HCL simulation and, as a consequence, prediction interval computation by Monte Carlo loop.
 
### 0.1

- Move HCL model class from popeye repository