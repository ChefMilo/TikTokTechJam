"""The MANUAL CEILING: a hand-built, best-effort pipeline whose only job
is to measure how much headroom exists above the organizers' published FM
baseline (validation primary ~0.6016).

WHY IT EXISTS
-------------
An autonomous agent's result is uninterpretable on its own. "+0.0008 on
validation" is either an excellent result or a rounding error depending
entirely on how much was there to win, and nothing else in this repo
answers that. This package is the reference point: a human, given the
same data, the same harness and no budget limit, gets THIS far. Whatever
the agent finds is then reported as a fraction of a measured ceiling
rather than as a bare number.

WHAT IT IS NOT
--------------
It is NOT part of the agent. It never goes through the Controller, it
proposes nothing, it has no ports, and no journal. It is a standalone
experiment we run once and quote.

THE IMPORT BOUNDARY, WHICH IS ENFORCED BY A TEST
------------------------------------------------
This package imports ONLY:

  - harness/       (W1: data, backtest, metrics, gate, cache)
  - contracts.py   (the shared value types)
  - the vendored starter kit, loaded by file path (see manual/_vendor.py)

It must NEVER import executor/ (W3). tests/test_manual.py parses every
module here and fails on an `executor` import, so the boundary cannot rot
quietly. The reason is not tidiness: executor/realize.py is another
workstream's file, under active development, and its `realize()` dispatch
raises NotImplementedError for exactly the additive moves this ceiling
exists to try. Riding it would mean either editing W3's file or being
blocked by it. Standing beside it costs one ~50-line training loop
(manual/run.py) and buys total independence.

Deliberate consequence: manual/run.py's training loop and
executor/realize.py's `_realize_fm` are near-identical, and that
duplication is the price of the boundary, paid knowingly. Both are
transcriptions of the same read-only vendor `run_fm`; the vendor's
version is the shared source of truth, not either copy.

WHAT IS HERE (unit 1)
---------------------
  manual/_vendor.py  loads the read-only vendored baseline module
  manual/encode.py   a field-parameterized re-implementation of the
                     vendor's encode(), proven identical to it by test
  manual/run.py      the standalone per-seed train/predict/score loop
  manual/report.py   human-readable output, plus the gate wiring

Crosses and the calibration blend are later units. Multi-task is out of
scope: its auxiliary labels (is_click and friends) are not reachable
through harness.data's 7-tuple, and reading the raw CSVs here would
bypass the split-leak guard that data.load's PermissionError exists to
enforce.
"""
