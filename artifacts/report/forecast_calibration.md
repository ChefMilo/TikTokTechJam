# Forecast calibration

Pairs each HYPOTHESIS's `expected_gain` (logged before evaluation)
against the realized delta from the matching DECISION event, by node.

| node | target_slot | expected_gain | realized_delta | abs_error |
|---|---|---|---|---|
| 1 | model | +0.0000 | +0.0000 | 0.0000 |
| 2 | weighting | +0.0080 | +0.0003 | 0.0077 |
| 3 | data_view | +0.0050 | -0.0090 | 0.0140 |
| 5 | model | +0.0010 | +0.0000 | 0.0010 |
| 8 | objective | +0.0120 | -0.0033 | 0.0153 |
| 10 | model | +0.0020 | +0.0003 | 0.0017 |
| 11 | ensemble | +0.0010 | +0.0008 | 0.0002 |

Mean absolute error: 0.0057
