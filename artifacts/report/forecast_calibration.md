# Forecast calibration

Pairs each HYPOTHESIS's `expected_gain` (logged before evaluation)
against the realized delta from the matching DECISION event, by node.
Only hypotheses that reached a real DECISION are scored here — see
'Unmeasured hypotheses' below for the rest.

| node | target_slot | expected_gain | realized_delta | abs_error |
|---|---|---|---|---|
| 1 | model | +0.0000 | +0.0000 | 0.0000 |
| 2 | weighting | +0.0080 | +0.0003 | 0.0077 |
| 3 | data_view | +0.0050 | -0.0090 | 0.0140 |
| 5 | model | +0.0010 | +0.0000 | 0.0010 |
| 8 | objective | +0.0120 | -0.0033 | 0.0153 |
| 10 | model | +0.0020 | +0.0003 | 0.0017 |
| 11 | ensemble | +0.0010 | +0.0008 | 0.0002 |

Mean absolute error (measured hypotheses only, n=7): 0.0057

**Largest same-direction miss**: node 2 (weighting) forecast +0.0080, realized +0.0003 — 29x smaller than predicted. Right direction, magnitude badly overestimated.

## Unmeasured hypotheses

Never reached a DECISION, so there is no realized delta to score against — excluded from the table and the mean absolute error above rather than counted as zero error.

| node | target_slot | expected_gain | why unmeasured |
|---|---|---|---|
| 4 | objective | +0.0080 | error: contract |
| 6 | calibration | +0.0100 | error: contract |
| 7 | model | +0.0060 | error: contract |
| 9 | calibration | +0.0040 | error: contract |
