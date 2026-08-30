"""Re-runs move 1 (baseline_reproduce) for base seeds 0..8 under the new
backtest-caching code path in executor/run.py.

Needed because run_candidate does not train val and backtest
independently — seeds 3-8 already had val cached (from
scripts/ensemble_candidate.py's step 1) but not backtest, and seeds 0,1,2
had neither cached under the new code path (their original backtest pass,
from the very first move-1 run, predates cache.save_predictions being
called for split="backtest" at all and was discarded). All nine need a
full pass to get both splits cached consistently.

Usage: python scripts/populate_move1_backtest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts import Citation, Status  # noqa: E402  (path bootstrap must run first)
from executor.journal import Journal  # noqa: E402
from executor.realize import build_config  # noqa: E402
from executor.run import run_candidate  # noqa: E402
from harness import cache  # noqa: E402
from methods.scripted import ScriptedGenerator  # noqa: E402

BASE_SEEDS = tuple(range(9))


def main() -> None:
    start = time.perf_counter()

    generator = ScriptedGenerator()
    fragment, payload = generator.propose(state=None)  # move 1: baseline_reproduce
    target_slot = payload["target_slot"]
    config_id = build_config(fragment, target_slot, seed=0).config_id
    print(f"move 1 config_id={config_id}, seeds={BASE_SEEDS}")

    journal_path = REPO_ROOT / "artifacts" / "journal_populate_move1_backtest.jsonl"
    journal = Journal(str(journal_path), run_id="populate_move1_backtest")
    journal.log_run_start(base_seeds=list(BASE_SEEDS), purpose="populate split=backtest cache for move 1")
    journal.log_hypothesis(
        target_slot,
        payload["rationale"],
        Citation(**payload["citation"]),
        payload["expected_gain"],
        payload["expected_cost_s"],
        tuple(payload["predecessor_evidence"]),
    )

    result = run_candidate(fragment, target_slot, seeds=BASE_SEEDS, journal=journal)
    elapsed = time.perf_counter() - start
    print(f"status={result.status}  elapsed={elapsed:.1f}s")

    if result.status is not Status.OK:
        print(f"FAILED: {result.error_excerpt}")
        journal.log_finalize(stop_reason="populate_move1_backtest_failed")
        raise SystemExit(1)

    for seed in BASE_SEEDS:
        print(
            f"  seed {seed} | val primary {result.val[seed].primary:.4f} "
            f"| backtest primary {result.backtest[seed].primary:.4f}"
        )

    print("\ncache.exists check, both splits, all nine seeds:")
    all_present = True
    for seed in BASE_SEEDS:
        val_ok = cache.exists(config_id, seed, "val")
        bt_ok = cache.exists(config_id, seed, "backtest")
        all_present = all_present and val_ok and bt_ok
        print(f"  seed {seed} | val={val_ok} | backtest={bt_ok}")
    print(f"\nall nine seeds fully cached (val + backtest): {all_present}")

    journal.log_finalize(stop_reason="populate_move1_backtest_complete")
    print(f"\ntotal elapsed: {elapsed / 60:.1f} min")
    print(f"journal written to {journal_path}")


if __name__ == "__main__":
    main()
