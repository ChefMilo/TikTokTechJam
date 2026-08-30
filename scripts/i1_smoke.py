"""I1 checkpoint: one real candidate, end to end, with zero LLM variance.

Runs methods.scripted's move 1 (baseline_reproduce) through
executor.run_candidate on 3 seeds, confirms predictions actually reached
the cache, confirms our own harness reproduces the published FM baseline
within tolerance, and — the real proof this is wired correctly — runs
harness.gate.compare on the result against itself and confirms the
REAL per-user bootstrap ran (not the coarse fallback).

Usage: python scripts/i1_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts import Status  # noqa: E402  (path bootstrap must run first)
from executor.run import run_candidate  # noqa: E402
from harness import cache, gate  # noqa: E402
from methods.scripted import ScriptedGenerator  # noqa: E402

SEEDS = (0, 1, 2)
PUBLISHED_FM_VALID_PRIMARY = 0.6015
PRIMARY_TOLERANCE = 0.005


def main() -> None:
    start = time.perf_counter()

    generator = ScriptedGenerator()
    fragment, payload = generator.propose(state=None)
    target_slot = payload["target_slot"]
    print(f"move 1: target_slot={target_slot!r} impl={fragment.impl!r} params={fragment.params}")

    result = run_candidate(fragment, target_slot, seeds=SEEDS)
    print(f"config_id={result.config_id}  status={result.status}")

    if result.status is not Status.OK:
        print(f"FAILED: {result.error_excerpt}")
        raise SystemExit(1)

    for seed in SEEDS:
        val_primary = result.val[seed].primary
        backtest_primary = result.backtest[seed].primary
        print(f"  seed {seed} | val primary {val_primary:.4f} | backtest primary {backtest_primary:.4f}")

    for seed in SEEDS:
        assert cache.exists(result.config_id, seed, "val"), (
            f"cache.exists is False for config_id={result.config_id} seed={seed} "
            "— predictions were not actually cached"
        )
    print(f"cache.exists confirmed for all {len(SEEDS)} seeds")

    mean_val_primary = sum(result.val[s].primary for s in SEEDS) / len(SEEDS)
    diff = abs(mean_val_primary - PUBLISHED_FM_VALID_PRIMARY)
    print(
        f"mean val primary = {mean_val_primary:.4f} "
        f"(published: {PUBLISHED_FM_VALID_PRIMARY}, diff: {diff:.4f}, tolerance: {PRIMARY_TOLERANCE})"
    )
    assert diff <= PRIMARY_TOLERANCE, (
        f"mean val primary {mean_val_primary:.4f} misses published "
        f"{PUBLISHED_FM_VALID_PRIMARY} by more than {PRIMARY_TOLERANCE}"
    )

    # The real proof: compare the candidate against itself. Identical
    # predictions on both sides should reject (delta ~0), but the REASON
    # is what matters here — if the user-level bootstrap is wired
    # correctly, it never falls back to the coarse per-seed one.
    verdict = gate.compare(result, result)
    print(f"gate.compare(result, result) -> {verdict}")
    assert "coarse_ci_seed_bootstrap" not in verdict.reason, (
        "gate fell back to the coarse seed-level bootstrap — the real "
        "user-level bootstrap did not run, so harness and executor are "
        "NOT correctly wired together"
    )
    print("confirmed: user-level bootstrap is live (no coarse_ci_seed_bootstrap fallback)")

    elapsed = time.perf_counter() - start
    print(f"\nI1 smoke test PASSED in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
