# Results

| Metric | Value |
|---|---|
| Validation-best GAUC | n/a |
| Validation-best nDCG@5 | n/a |
| Validation-best primary | 0.6014 |
| Delta vs official baseline GAUC (0.6674) | n/a |
| Delta vs official baseline nDCG@5 (0.5357) | n/a |
| Delta vs official baseline primary (0.6016) | -0.0002 |
| Iterations used | 1 / 50 |
| Total agent wall-clock (s) | 40.7 |
| Total GPU-seconds | 0.0 |
| Total tokens | 0 |
| Manual interventions | 0 |
| Total training wall-clock (s), measured | 69.6 |

_"Total agent wall-clock" above sums each EVAL_RESULT's own wall_seconds — it covers only the time this render's run spent re-evaluating, which is ~0s whenever candidates are rebuilt from cache rather than trained. Actual training wall-clock is reported in the row above, passed in separately by the caller._

## Convergence

- Candidates decided: 1
- Baseline adopted (not an improvement): 1
- Accepted as improvements: 0
- Cleared epsilon=0.002: 0
- Consequence: No candidate was accepted, so the epsilon question does not arise.

2 candidate(s) failed (error classes: contract, generator_exhausted); the run continued past all 2 of them and reached FINALIZE.
