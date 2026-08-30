"""Runs a configurable subset of methods.scripted's moves through
executor.run_candidate on seeds (0,1,2), journaling as it goes, then
gate.compares every non-baseline move against move 1.

MOVE_INDICES controls which moves actually run — the generator always
emits moves in fixed order, so reaching move N still calls propose() N
times, but moves not in MOVE_INDICES are discarded immediately (never
run through run_candidate) rather than wasting a training pass. Move 1
is always required: everything else is compared against it.

This run: MOVE_INDICES = (1, 8). Moves 2 and 3 were already measured in
an earlier run (recency_weight_exp: +0.00028, noise; recency_window:
-0.00904, decisively worse) and are not worth repeating here — move 8
(pairwise_loss) is the new one being tested.

Usage: python scripts/compare_moves.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts import Citation, Status, Verdict  # noqa: E402  (path bootstrap must run first)
from executor.journal import Journal  # noqa: E402
from executor.run import run_candidate  # noqa: E402
from harness import gate  # noqa: E402
from methods.scripted import ScriptedGenerator  # noqa: E402

SEEDS = (0, 1, 2)
MOVE_INDICES = (1, 8)


def _print_verdict(label: str, verdict: Verdict) -> None:
    print(f"\n{label}:")
    print(f"  accept          = {verdict.accept}")
    print(f"  delta           = {verdict.delta:+.5f}")
    print(f"  ci95            = ({verdict.ci95[0]:+.5f}, {verdict.ci95[1]:+.5f})")
    print(f"  n_seeds         = {verdict.n_seeds}")
    print(f"  backtest_delta  = {verdict.backtest_delta}")
    print(f"  reason          = {verdict.reason}")


def main() -> None:
    start = time.perf_counter()
    assert 1 in MOVE_INDICES, "move 1 is the baseline every other move is compared against"

    generator = ScriptedGenerator()
    n_moves_to_advance = max(MOVE_INDICES)

    # A fresh, purpose-scoped file per set of moves compared — not the
    # earlier moves-2/3 run's journal. Appending this run onto that one
    # would give "move 1" two entries under different node numbers in the
    # same log, which would misrender in executor.report.
    journal_path = REPO_ROOT / "artifacts" / f"journal_compare_moves_{'_'.join(map(str, MOVE_INDICES))}.jsonl"
    journal = Journal(str(journal_path), run_id="compare_moves")
    journal.log_run_start(seeds=list(SEEDS))

    results = {}
    for idx in range(1, n_moves_to_advance + 1):
        fragment, payload = generator.propose(state=None)
        if idx not in MOVE_INDICES:
            continue  # advance the generator past it, but never train it

        target_slot = payload["target_slot"]
        print(f"\n=== move {idx}: target_slot={target_slot!r} impl={fragment.impl!r} params={fragment.params} ===")
        journal.log_hypothesis(
            target_slot,
            payload["rationale"],
            Citation(**payload["citation"]),
            payload["expected_gain"],
            payload["expected_cost_s"],
            tuple(payload["predecessor_evidence"]),
        )

        t0 = time.perf_counter()
        result = run_candidate(fragment, target_slot, seeds=SEEDS, journal=journal)
        elapsed = time.perf_counter() - t0
        print(f"  config_id={result.config_id}  status={result.status}  elapsed={elapsed:.1f}s")

        if result.status is not Status.OK:
            print(f"  FAILED: {result.error_excerpt}")
            raise SystemExit(1)

        for seed in SEEDS:
            print(
                f"    seed {seed} | val primary {result.val[seed].primary:.4f} "
                f"| backtest primary {result.backtest[seed].primary:.4f}"
            )

        results[idx] = result

    journal.log_finalize(stop_reason="compare_moves_script_complete")

    baseline = results[1]
    for idx in MOVE_INDICES:
        if idx == 1:
            continue
        verdict = gate.compare(results[idx], baseline)
        _print_verdict(f"gate.compare(move{idx}, move1=baseline)", verdict)

    elapsed_total = time.perf_counter() - start
    print(f"\ntotal elapsed: {elapsed_total / 60:.1f} min")
    print(f"journal written to {journal_path}")


if __name__ == "__main__":
    main()
