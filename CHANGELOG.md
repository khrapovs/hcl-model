# Changelog

## 0.5.1 (2022-02-15)
### Changed
- Convert previously private method to public: `hcl_model.model_base.ModelBase.get_parameters`.

## 0.5.0 (2022-02-08)
### Added
- `sklearn` compliant transformer `TargetOutlierCorrectionTransformer` in `hcl_model.transformers.outlier_correction`. Read documentation in `docs/data_preprocessing.md`.
- `sklearn` compliant transformer `TargetStructuralBreakCorrectionTransformer` in `hcl_model.transformers.structural_breaks`. It uses [`ruptures` package](https://github.com/deepcharles/ruptures/) to correct structural breaks in `X` data.
- `sklearn` compliant transformer `EstimatorToTransformer` in `hcl_model.transformers.estimator_to_transformer`. Effectively, it replaces `predict` method with `transform` so that the object is usable as an intermediate step in `Pipeline`.
- `sklearn` compliant transformer `TruncateTransformer` in `hcl_model.transformers.truncate`. Leave only a specific number of past weeks in the data.
- `sklearn` compliant transformer `SeriesToFrameTransformer` in `hcl_model.transformers.series_to_frame`. It simply applies `.to_frame()` to a `pd.Series`.
- A replacement for `sklearn.pipeline.FeatureUnion`: `DatetimeIndexedFeatureUnion` in `hcl_model.transformers.feature_union`.
- A replacement for `sklearn.pipeline.make_union`: `make_union_of_datetime_indexed_features` in `hcl_model.transformers.feature_union`.
- Move utility functions from `utils.py` to a separate module `utils` with `.py` file per function.
#### Changed
- Move `CalendarTransformer` to `hcl_model.transformers.calendar`.
- Move `CalendarReader` to `hcl_model.utils.calendar_reader.CalendarReader`.

## 0.4.0 (2022-01-26)
- Rename arguments: `exog` into `X`, and `endog` into `y`. This is done to comply with [`sklearn`](https://scikit-learn.org/stable/developers/develop.html) general interface.
- Now `.fit` should be fed only with past data, and `.predict` only with future data.

## 0.3.9 (2021-05-19)
- Allow passing arbitrary transformations to `HandCraftedLinearModel` that return multi-column dataframes.

## 0.3.8 (2021-05-17)
- Use datetime index of exogenous and endogenous variables. 

## 0.3.7 (2021-03-09)
- Remove upper constraint on `pandas` version.

## 0.3.6 (2021-01-04)
- Remove version restriction on `skyfield`.
- Limit `pandas` version by 1.1.5 pending incompatibility fix `patsy` 0.5.1.

## 0.3.5 (2020-12-10)
- Use `median_abs_deviation` instead of deprecated `median_absolute_deviation`. This requires `scipy>=1.5`.
- Use `dates.isocalendar().week` instead of deprecated `dates.week`. This requires `pandas>=1.1`.
- Silence warning about default `dtype` in empty `Series`. Now `dtype=float`.

## 0.3.4 (2020-10-21)
- Change the way seasonal outliers are detected. Instead of looking at percentage changes, the focus now is at the absolute deviation from the exponentially weighted moving average.

## 0.3.3 (2020-07-24)
- Limit the version of `skyfield` package to 1.22 due to an error `AttributeError: module 'datetime' has no attribute 'combine'`.

## 0.3.2 (2020-06-05)
- Require `workalendar` version to be greater than 10.0.0 due to a rename of `IsoRegistry.get_calendar_class()` into `IsoRegistry.get()`.
 
## 0.3.1 (2020-01-27)
- Fix the bug in automatic dummy detection. Instead of taking weeks from sorted normalized series of changes, they were taken from some other intermediate result. 

## 0.3 (2020-01-10)
- Add several utility functions that help construct exogenous regressors based on a calendar. These functions are accessible through `construct_calendar_exogenous`. 
- Add utility function to create decayed weights for weighted OLS estimation.

## 0.2.1 (2020-01-21)
- Add parameter dictionary to model summary output.

## 0.2 (2020-01-10)
- Speed up HCL simulation and, as a consequence, prediction interval computation by Monte Carlo loop.
 
## 0.1 (2019-11-28)
- Move HCL model class from popeye repository