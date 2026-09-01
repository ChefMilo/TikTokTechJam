# Results

| Metric | Value |
|---|---|
| Validation-best GAUC | 0.6683 |
| Validation-best nDCG@5 | 0.5363 |
| Validation-best primary | 0.6023 |
| Delta vs official baseline GAUC (0.6674) | +0.0009 |
| Delta vs official baseline nDCG@5 (0.5357) | +0.0006 |
| Delta vs official baseline primary (0.6016) | +0.0007 |
| Iterations used | 2 / 50 |
| Total agent wall-clock (s) | 21.7 |
| Total tokens | 0 (no LLM in the loop yet) |
| Manual interventions | 0 |
| Total training wall-clock (s), measured | 1935.8 |

_"Total agent wall-clock" above sums each EVAL_RESULT's own wall_seconds — it covers only the time this render's run spent re-evaluating, which is ~0s whenever candidates are rebuilt from cache rather than trained. Actual training wall-clock is reported in the row above, passed in separately by the caller._

## Convergence

- Candidates decided: 7
- Baseline adopted (not an improvement): 1
- Accepted as improvements: 1
- Cleared epsilon=0.002: 0
- Consequence: 1 candidate(s) were accepted as statistically real improvements, but none cleared epsilon=0.002. Under the organizers' N=3 no-improvement rule, this run would still be judged as stalled despite the real gain(s).

4 candidate(s) failed (error classes: contract); the run continued past all 4 of them and reached FINALIZE.
