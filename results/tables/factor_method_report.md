# Factor Methods Comparison

## Reconstruction Error Metric

- MSE = mean((y_true - y_hat)^2) over all timestamps and tenors in each split.
- MAE = mean(|y_true - y_hat|) over all timestamps and tenors in each split.
- R2 = 1 - sum((y_true - y_hat)^2) / sum((y_true - mean(y_true))^2).
- Ranking priority in this report: lower MSE first, then lower MAE.

## Datasets

| dataset | n_samples | n_tenors | tenor_span_months | source_period | dy_period | train/val/test_periods |
|---|---:|---:|---|---|---|---|
| liu_wu1 | 9775 | 360 | 1-360 | 1985-11-25..2024-12-31 | 1985-11-26..2024-12-31 | train:1985-11-26..2015-12-31 (7523); val:2016-01-04..2018-12-31 (749); test:2019-01-02..2024-12-31 (1503) |

## liu_wu1

### train

| rank | method | mse | mae | r2 |
|---:|---|---:|---:|---:|
| 1 | PCA | 0.05748950 | 0.13455978 | 0.94250286 |
| 2 | MFA | 0.05748950 | 0.13455978 | 0.94250286 |
| 3 | AE | 0.07708546 | 0.14756084 | 0.92290430 |
| 4 | NMF | 0.15041026 | 0.23558012 | 0.84956975 |

### val

| rank | method | mse | mae | r2 |
|---:|---|---:|---:|---:|
| 1 | PCA | 0.00850197 | 0.06137091 | 0.97867835 |
| 2 | MFA | 0.00850197 | 0.06137091 | 0.97867835 |
| 3 | AE | 0.01001971 | 0.06410095 | 0.97487211 |
| 4 | NMF | 0.07170744 | 0.16961063 | 0.82016875 |

### test

| rank | method | mse | mae | r2 |
|---:|---|---:|---:|---:|
| 1 | PCA | 0.04237360 | 0.10799276 | 0.95588710 |
| 2 | MFA | 0.04237360 | 0.10799276 | 0.95588710 |
| 3 | AE | 0.07035418 | 0.12848184 | 0.92675801 |
| 4 | NMF | 0.14307215 | 0.22434922 | 0.85105521 |
