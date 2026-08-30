"""Produces the final hidden-test submission from the validation-best
configuration: a 3-seed rank-average ensemble of move 1 (baseline FM),
which reached val primary 0.6022/0.6029/0.6018 per ensemble seed versus
0.6011-0.6020 for individual baseline seeds (scripts/ensemble_probe.py,
scripts/ensemble_candidate.py).

Step 1 trains fresh on the FULL train split (not the backtest's fit
window) for seeds 0,1,2, scoring on TEST. This is the one sanctioned use
of test-split structure — see harness/validate.py's module docstring:
we read test ROW STRUCTURE (user_id, video_id, ordering) to produce
aligned predictions, never test LABELS. Model selection (which training
epoch's weights survive) is decided entirely from the real validation
split, before test rows are ever touched — see
executor.realize.realize_for_submission's docstring for exactly how
that's enforced. harness.data.load("test") remains correctly forbidden;
test rows come from harness.validate.load_split_rows("test"), the same
vendor-loader path harness/validate.py itself uses to check submissions.

Usage: python scripts/make_submission.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from executor.realize import build_config, realize_for_submission  # noqa: E402
from harness import data, validate  # noqa: E402
from methods.scripted import ScriptedGenerator  # noqa: E402

SEEDS = (0, 1, 2)
OUTPUT_PATH = REPO_ROOT / "artifacts" / "submission_test.csv"


def main() -> None:
    start = time.perf_counter()

    generator = ScriptedGenerator()
    fragment, payload = generator.propose(state=None)  # move 1: baseline_reproduce
    target_slot = payload["target_slot"]
    print(f"move 1: target_slot={target_slot!r} impl={fragment.impl!r} params={fragment.params}")

    train_rows = data.load("train")
    val_rows = data.load("val")
    # The one sanctioned way to get test row structure — see module
    # docstring. Never harness.data.load("test"), which refuses by design.
    test_rows = validate.load_split_rows("test")
    print(f"train_rows={len(train_rows):,}  val_rows={len(val_rows):,}  test_rows={len(test_rows):,}")

    reference_user_ids = None
    seed_scores = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        config = build_config(fragment, target_slot, seed)
        user_ids, scores = realize_for_submission(config, train_rows, val_rows, test_rows, seed)
        elapsed = time.perf_counter() - t0
        print(f"  seed {seed}: trained on full train, model-selected on val, scored on test in {elapsed:.1f}s")

        if reference_user_ids is None:
            reference_user_ids = user_ids
        elif not np.array_equal(user_ids, reference_user_ids):
            raise RuntimeError(
                f"seed {seed}'s test user_ids do not match seed {SEEDS[0]}'s — misaligned rows"
            )

        seed_scores.append(scores)

    # Step 2: rank-average the three seeds' test scores.
    ranks = [rankdata(s) for s in seed_scores]
    rank_avg_scores = np.mean(ranks, axis=0)
    print(f"\nrank-averaged {len(SEEDS)} seeds' test scores")

    # Step 3: write via harness.validate so formatting matches the
    # vendor's exactly.
    validate.write_submission(str(OUTPUT_PATH), test_rows, rank_avg_scores)
    print(f"wrote {OUTPUT_PATH}")

    # Step 4: validate. If this fails, report the specific check that
    # failed rather than working around it — see harness/SCHEMA_NOTES.md
    # submit.py Q17 for the eight checks read_submission performs.
    try:
        validate.validate_submission(str(OUTPUT_PATH), "test")
        print("validate_submission(path, 'test'): PASSED (all eight vendor checks)")
    except ValueError as exc:
        print(f"validate_submission(path, 'test') FAILED: {exc}")
        raise

    # Step 5: row count + first five lines.
    with open(OUTPUT_PATH, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    print(f"\nrow count (including header): {len(lines):,}")
    print("first five lines:")
    for line in lines[:5]:
        print(f"  {line.rstrip()}")

    elapsed_total = time.perf_counter() - start
    print(f"\ntotal elapsed: {elapsed_total / 60:.1f} min")


if __name__ == "__main__":
    main()
