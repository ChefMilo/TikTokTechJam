"""Exploratory ensemble probe: does combining move 1's own seed
predictions (rank-averaged) beat any individual seed, and does WIDER
beat narrower? No training involved — reads harness.cache's already-
cached validation predictions for the baseline FM (config_id
bce19171850a) at seeds 0-8. This is a measurement only; it does not
change scripts/make_submission.py's current 3-seed submission.

Rank-averaging (scipy.stats.rankdata per seed, then averaged) rather
than averaging fitted weights: only relative order matters for GAUC and
nDCG@5, and rank-averaging is robust to scale differences between seeds
that plain score-averaging is not.

Usage: python scripts/ensemble_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from harness import cache, metrics  # noqa: E402

CONFIG_ID = "bce19171850a"
ALL_SEEDS = tuple(range(9))
WIDTHS = (3, 5, 7, 9)


def _print_metrics(label: str, user_ids, labels, scores):
    m = metrics.evaluate(user_ids, labels, scores)
    print(
        f"{label:22s} | GAUC {m.values['GAUC']:.4f} | nDCG@5 {m.values['nDCG@5']:.4f} "
        f"| primary {m.primary:.4f}"
    )
    return m


def main() -> None:
    predictions = {}
    for seed in ALL_SEEDS:
        predictions[seed] = cache.load_predictions(CONFIG_ID, seed, "val")

    reference_seed = ALL_SEEDS[0]
    reference_user_ids, reference_labels, _ = predictions[reference_seed]
    for seed in ALL_SEEDS[1:]:
        user_ids, labels, _ = predictions[seed]
        if not np.array_equal(user_ids, reference_user_ids) or not np.array_equal(labels, reference_labels):
            print(
                f"STOP: seed {seed}'s (user_ids, labels) differ from seed "
                f"{reference_seed}'s — rows are not aligned across seeds, "
                "cannot ensemble."
            )
            raise SystemExit(1)
    print(f"confirmed: user_ids and labels are identical across all {len(ALL_SEEDS)} seeds\n")

    user_ids, labels = reference_user_ids, reference_labels

    print("--- individual seeds ---")
    individual_primaries = {}
    for seed in ALL_SEEDS:
        _, _, scores = predictions[seed]
        m = _print_metrics(f"seed {seed}", user_ids, labels, scores)
        individual_primaries[seed] = m.primary
    best_seed = max(individual_primaries, key=individual_primaries.get)
    print(f"\nbest single seed (reference): seed {best_seed}, primary {individual_primaries[best_seed]:.4f}")

    print("\n--- ensemble width sweep: rank-average of seeds 0..N-1 ---")
    width_primaries = {}
    for width in WIDTHS:
        seeds = ALL_SEEDS[:width]
        ranks = [rankdata(predictions[s][2]) for s in seeds]
        rank_avg = np.mean(ranks, axis=0)
        m = _print_metrics(f"rank-avg width={width}", user_ids, labels, rank_avg)
        width_primaries[width] = m.primary

    print()
    diff = width_primaries[9] - width_primaries[3]
    if diff > 0:
        print(
            f"9-seed rank-average BEATS 3-seed on validation: "
            f"primary {width_primaries[9]:.6f} (width=9) vs {width_primaries[3]:.6f} (width=3), "
            f"+{diff:.6f}."
        )
    elif diff < 0:
        print(
            f"9-seed rank-average is WORSE than 3-seed on validation: "
            f"primary {width_primaries[9]:.6f} (width=9) vs {width_primaries[3]:.6f} (width=3), "
            f"{diff:+.6f}."
        )
    else:
        print(f"9-seed and 3-seed rank-average are TIED on validation at primary {width_primaries[9]:.6f}.")


if __name__ == "__main__":
    main()
