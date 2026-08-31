"""Rolling-origin temporal backtest, carved entirely out of the TRAINING
window: fit on an earlier sub-window, score on a later one.

Its purpose is to reproduce, inside the training data alone, the same
forward-in-time gap that exists between validation and the hidden test
set — so that a change which only helps by overfitting the validation
set gets caught here before it ever reaches validation.

Both sub-windows come from harness.data.load("train") only. This module
must never call data.load("val") or data.load("test"): the whole point is
to measure the forward-gap effect using data a candidate is already
allowed to see, not to sneak a second look at validation or test.

BOUNDARY CHOICE — volume shape, not just date arithmetic: daily row
counts show train is heavily front-loaded (peak 278,835 rows on
20220411, decaying to a ~20-24k/day plateau by 20220418-21), while
validation is flat throughout at 14-27k/day — i.e. validation resembles
the PLATEAU, not the burst. Splitting at BT_FIT_END=20220414 (the
original boundary) put nearly all of the burst in fit and the
burst-to-plateau TRANSITION in score, simulating a different regime
shift than the real train->val one (mixed burst+plateau -> pure
plateau). Moving the boundary to BT_FIT_END=20220417 puts the whole
burst plus the early plateau days in fit — matching the real train
split's own mixed composition — and leaves score
(20220418-20220421) as pure plateau, matching real validation's shape.
A change that only looks good against the old boundary's transition
period, rather than against real steady-state behaviour, is exactly the
kind of overfitting this backtest exists to catch.
"""

from __future__ import annotations

from harness import data

BT_FIT_END = 20220417
BT_SCORE_START = 20220418
BT_SCORE_END = 20220421

# Row counts of both sub-windows, recorded on first split() call. There is
# no organizer-published figure for this internal split to check against
# — ours becomes the reference for anyone who needs to sanity-check a
# future change to the boundaries above.
_ROW_COUNTS: dict[str, int] | None = None


def split() -> tuple[list[tuple], list[tuple]]:
    """Returns (fit_rows, score_rows), both drawn from data.load("train").

    fit:   date <= BT_FIT_END
    score: BT_SCORE_START <= date <= BT_SCORE_END

    BT_SCORE_START is BT_FIT_END + 1 and BT_SCORE_END equals data.TRAIN_END,
    so the two windows partition the entire train split with no gap and no
    overlap.
    """
    global _ROW_COUNTS
    train_rows = data.load("train")
    fit_rows = [row for row in train_rows if row[0] <= BT_FIT_END]
    score_rows = [row for row in train_rows if BT_SCORE_START <= row[0] <= BT_SCORE_END]

    if _ROW_COUNTS is None:
        _ROW_COUNTS = {"fit": len(fit_rows), "score": len(score_rows)}
        print(
            f"harness.backtest.split(): fit={_ROW_COUNTS['fit']:,} rows "
            f"(date <= {BT_FIT_END}), score={_ROW_COUNTS['score']:,} rows "
            f"({BT_SCORE_START} <= date <= {BT_SCORE_END}) - recorded as reference"
        )

    return fit_rows, score_rows
