"""Exploratory ensemble probe: does combining move 1's own seed
predictions (rank-averaged or mean) beat any individual seed? No
training involved — reads harness.cache's already-cached validation
predictions for the baseline FM (config_id bce19171850a) at seeds 0,1,2.

Rank-averaging (scipy.stats.rankdata per seed, then averaged) rather
than averaging fitted weights: only relative order matters for GAUC and
nDCG@5, and rank-averaging is robust to scale differences between seeds
that plain score-averaging is not.

Usage: python scripts/ensemble_probe.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from harness import cache, metrics  # noqa: E402

CONFIG_ID = "bce19171850a"
SEEDS = (0, 1, 2)


def _print_metrics(label: str, user_ids, labels, scores) -> None:
    m = metrics.evaluate(user_ids, labels, scores)
    print(
        f"{label:20s} | GAUC {m.values['GAUC']:.4f} | nDCG@5 {m.values['nDCG@5']:.4f} "
        f"| primary {m.primary:.4f}"
    )


def main() -> None:
    predictions = {}
    for seed in SEEDS:
        predictions[seed] = cache.load_predictions(CONFIG_ID, seed, "val")

    reference_seed = SEEDS[0]
    reference_user_ids, reference_labels, _ = predictions[reference_seed]
    for seed in SEEDS[1:]:
        user_ids, labels, _ = predictions[seed]
        if not np.array_equal(user_ids, reference_user_ids) or not np.array_equal(labels, reference_labels):
            print(
                f"STOP: seed {seed}'s (user_ids, labels) differ from seed "
                f"{reference_seed}'s — rows are not aligned across seeds, "
                "cannot ensemble."
            )
            raise SystemExit(1)
    print("confirmed: user_ids and labels are identical across all three seeds\n")

    user_ids, labels = reference_user_ids, reference_labels

    print("--- individual seeds ---")
    for seed in SEEDS:
        _, _, scores = predictions[seed]
        _print_metrics(f"seed {seed}", user_ids, labels, scores)

    print("\n--- ensembles across all three seeds ---")
    all_scores = [predictions[s][2] for s in SEEDS]
    ranks = [rankdata(s) for s in all_scores]
    rank_avg_all = np.mean(ranks, axis=0)
    _print_metrics("rank-avg (0,1,2)", user_ids, labels, rank_avg_all)

    mean_raw_all = np.mean(all_scores, axis=0)
    _print_metrics("mean raw (0,1,2)", user_ids, labels, mean_raw_all)

    print("\n--- pairwise rank-averages ---")
    for s1, s2 in combinations(SEEDS, 2):
        r1 = rankdata(predictions[s1][2])
        r2 = rankdata(predictions[s2][2])
        rank_avg_pair = (r1 + r2) / 2.0
        _print_metrics(f"rank-avg ({s1},{s2})", user_ids, labels, rank_avg_pair)


if __name__ == "__main__":
    main()
