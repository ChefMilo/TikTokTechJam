"""Turns the rank-average ensemble probe (scripts/ensemble_probe.py) into
a properly gated candidate — validation AND backtest.

Design: an "ensemble seed" is a disjoint group of 3 base seeds, rank-
averaged. Disjoint groups matter — overlapping groups would share
training runs and the resulting per-seed results would be correlated,
which breaks the paired comparison harness.gate does.

    ensemble seed 0 <- base seeds 0,1,2
    ensemble seed 1 <- base seeds 3,4,5
    ensemble seed 2 <- base seeds 6,7,8

Requires every base seed's predictions to be cached for BOTH split="val"
and split="backtest" (see scripts/populate_move1_backtest.py, which
back-fills this for base seeds that predate executor/run.py caching the
backtest split at all). If either split is missing for any base seed,
this raises rather than silently building a candidate on partial
evidence.

Usage: python scripts/ensemble_candidate.py
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

from contracts import CandidateResult, Status, Verdict  # noqa: E402
from executor.journal import Journal  # noqa: E402
from executor.realize import build_config  # noqa: E402
from harness import cache, gate, metrics  # noqa: E402
from methods.scripted import ScriptedGenerator  # noqa: E402

BASE_SEEDS = tuple(range(9))
ENSEMBLE_GROUPS = {0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8)}
ENSEMBLE_CONFIG_ID = "ens_rank3"
SPLITS = ("val", "backtest")


def _print_verdict(label: str, verdict: Verdict) -> None:
    print(f"\n{label}:")
    print(f"  accept          = {verdict.accept}")
    print(f"  delta           = {verdict.delta:+.5f}")
    print(f"  ci95            = ({verdict.ci95[0]:+.5f}, {verdict.ci95[1]:+.5f})")
    print(f"  n_seeds         = {verdict.n_seeds}")
    print(f"  backtest_delta  = {verdict.backtest_delta}")
    print(f"  reason          = {verdict.reason}")


def _rank_average_group(config_id: str, seeds: tuple, split: str):
    predictions = [cache.load_predictions(config_id, s, split) for s in seeds]
    reference_user_ids, reference_labels, _ = predictions[0]
    for user_ids, labels, _ in predictions[1:]:
        if not np.array_equal(user_ids, reference_user_ids) or not np.array_equal(labels, reference_labels):
            raise ValueError(
                f"seeds {seeds} do not share identical (user_ids, labels) for "
                f"split {split!r} — cannot rank-average"
            )
    ranks = [rankdata(scores) for _, _, scores in predictions]
    rank_avg = np.mean(ranks, axis=0)
    return reference_user_ids, reference_labels, rank_avg


def main() -> None:
    start = time.perf_counter()

    generator = ScriptedGenerator()
    fragment, payload = generator.propose(state=None)  # move 1: baseline_reproduce
    target_slot = payload["target_slot"]
    config_id = build_config(fragment, target_slot, seed=0).config_id
    print(f"move 1 config_id={config_id}")

    journal_path = REPO_ROOT / "artifacts" / "journal_ensemble_candidate.jsonl"
    journal = Journal(str(journal_path), run_id="ensemble_candidate")
    journal.log_run_start(
        base_seeds=list(BASE_SEEDS),
        ensemble_groups={str(k): list(v) for k, v in ENSEMBLE_GROUPS.items()},
        splits=list(SPLITS),
    )

    # --- Preflight: every base seed must have both splits cached ---
    missing = [
        (s, split) for s in BASE_SEEDS for split in SPLITS if not cache.exists(config_id, s, split)
    ]
    if missing:
        raise RuntimeError(
            f"missing cached predictions for (seed, split) = {missing}; "
            "run scripts/populate_move1_backtest.py first"
        )
    print(f"confirmed: all {len(BASE_SEEDS)} base seeds have both {SPLITS} cached")

    # --- Rank-average each ensemble group, per split ---
    ensemble_metrics = {split: {} for split in SPLITS}
    print("\n--- ensemble seeds (rank-averaged) ---")
    for split in SPLITS:
        for g, base_seeds in ENSEMBLE_GROUPS.items():
            user_ids, labels, rank_avg_scores = _rank_average_group(config_id, base_seeds, split)
            cache.save_predictions(ENSEMBLE_CONFIG_ID, g, split, user_ids, labels, rank_avg_scores)
            m = metrics.evaluate(user_ids, labels, rank_avg_scores)
            ensemble_metrics[split][g] = m
            print(
                f"  [{split:8s}] ensemble seed {g} <- base {base_seeds} | GAUC {m.values['GAUC']:.4f} "
                f"| nDCG@5 {m.values['nDCG@5']:.4f} | primary {m.primary:.4f}"
            )

    # --- Baseline incumbent over base seeds 0,1,2, both splits ---
    baseline_metrics = {split: {} for split in SPLITS}
    for split in SPLITS:
        for s in ENSEMBLE_GROUPS[0]:
            user_ids, labels, scores = cache.load_predictions(config_id, s, split)
            baseline_metrics[split][s] = metrics.evaluate(user_ids, labels, scores)

    baseline = CandidateResult(
        config_id=config_id,
        status=Status.OK,
        val=baseline_metrics["val"],
        backtest=baseline_metrics["backtest"],
        val_pred_path=f"artifacts/preds/{config_id}__<seed>__val.npz",
        wall_seconds=0.0,
    )
    ensemble = CandidateResult(
        config_id=ENSEMBLE_CONFIG_ID,
        status=Status.OK,
        val=ensemble_metrics["val"],
        backtest=ensemble_metrics["backtest"],
        val_pred_path=f"artifacts/preds/{ENSEMBLE_CONFIG_ID}__<seed>__val.npz",
        wall_seconds=time.perf_counter() - start,
    )

    journal.log_intervention(
        "ensemble_candidate.py",
        "rank_average_ensemble",
        "3-seed rank-average ensemble (config_id=ens_rank3) built from 9 "
        "disjoint-grouped base seeds of move 1, both val and backtest, "
        "to test whether ensembling clears the noise gate against the "
        "single-model baseline.",
    )
    journal.log_eval_result(
        ENSEMBLE_CONFIG_ID,
        ensemble_metrics["val"],
        ensemble.wall_seconds,
        backtest_per_seed_metrics=ensemble_metrics["backtest"],
        target_slot=None,
        fragment_impl="rank_avg_ensemble",
        fragment_params={"groups": {str(k): list(v) for k, v in ENSEMBLE_GROUPS.items()}},
    )

    # --- gate.compare, with full val + backtest evidence this time ---
    verdict = gate.compare(ensemble, baseline)
    _print_verdict("gate.compare(ensemble=ens_rank3, baseline)", verdict)
    journal.log_decision(verdict)
    journal.log_finalize(stop_reason="ensemble_candidate_script_complete")

    elapsed_total = time.perf_counter() - start
    print(f"\ntotal elapsed: {elapsed_total / 60:.1f} min")
    print(f"journal written to {journal_path}")


if __name__ == "__main__":
    main()
