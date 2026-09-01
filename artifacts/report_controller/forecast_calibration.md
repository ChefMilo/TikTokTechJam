# Forecast calibration

Pairs each HYPOTHESIS's `expected_gain` (logged before evaluation)
against the realized delta from the matching DECISION event, by node.

| node | target_slot | expected_gain | realized_delta | abs_error |
|---|---|---|---|---|
| 2 | model | +0.0000 | +0.0000 | 0.0000 |
| 4 | weighting | +0.0080 | +0.0003 | 0.0077 |
| 5 | model | +0.0010 | +0.0000 | 0.0010 |
| 6 | objective | +0.0120 | -0.0033 | 0.0153 |

Mean absolute error: 0.0060
