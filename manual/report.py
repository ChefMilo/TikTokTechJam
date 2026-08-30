"""Human-readable output for the manual ceiling, plus the gate wiring.

Two audiences, two functions. `print_candidate_report` is what a person
launching a run reads to see whether it worked at all — per-seed numbers,
so a single bad seed is visible rather than averaged away, and a mean the
reviewer can eyeball against the organizers' published ~0.6016.

`print_comparison` is the honest half: it does not compute a delta itself,
it hands both CandidateResults to harness.gate and prints what the gate
says. Unit 1 does not call it (there is nothing to compare a baseline
against yet), but the wiring lives here now so unit 2's variants have
nowhere else to put their arithmetic. A ceiling that reported its own
hand-rolled "improvement" instead of a Verdict would be exactly the
self-graded number this project's noise gate exists to prevent.
"""

from __future__ import annotations

from typing import Optional

from contracts import CandidateResult, Metrics
from harness import gate

PUBLISHED_BASELINE_PRIMARY = 0.6016
"""The organizers' published FM validation primary, for eyeball comparison
only. Never asserted in a test — reproducing it needs the real dataset,
which is the point of running this pipeline at all."""


def mean_primary(per_seed: dict[int, Metrics]) -> Optional[float]:
    """Mean primary across seeds, or None when there is nothing to mean.

    None rather than 0.0 or nan: a run with no seeds has not measured a
    primary of zero, it has measured nothing, and a printed 0.0000 reads
    like a catastrophic result rather than like an empty run.
    """
    if not per_seed:
        return None
    return sum(m.primary for m in per_seed.values()) / len(per_seed)


def _format_primary(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_per_seed(title: str, per_seed: dict[int, Metrics]) -> None:
    print(f"  {title}")
    if not per_seed:
        print("    (no seeds)")
        return
    for seed in sorted(per_seed):
        values = per_seed[seed].values
        gauc = values.get("GAUC")
        ndcg = values.get("nDCG@5")
        parts = [f"    seed {seed}: primary {per_seed[seed].primary:.4f}"]
        if gauc is not None:
            parts.append(f"GAUC {gauc:.4f}")
        if ndcg is not None:
            parts.append(f"nDCG@5 {ndcg:.4f}")
        print(" | ".join(parts))


def print_candidate_report(result: CandidateResult, label: str = "candidate") -> None:
    """Per-seed validation and backtest numbers, plus the headline mean."""
    print()
    print(f"=== {label} ===")
    print(f"  config_id     : {result.config_id}")
    print(f"  status        : {result.status.value}")
    print(f"  wall_seconds  : {result.wall_seconds:.1f}")

    _print_per_seed("validation:", result.val)
    _print_per_seed("backtest:", result.backtest)

    mean_val = mean_primary(result.val)
    print()
    print(f"  MEAN VAL PRIMARY : {_format_primary(mean_val)}")
    print(
        f"  (organizers' published FM baseline: {PUBLISHED_BASELINE_PRIMARY:.4f} — "
        "the baseline variant should land near this)"
    )
    mean_backtest = mean_primary(result.backtest)
    print(f"  mean backtest primary : {_format_primary(mean_backtest)}")


def print_comparison(
    candidate: CandidateResult,
    incumbent: CandidateResult,
    label: str = "variant vs baseline",
) -> None:
    """Runs harness.gate.compare and prints the Verdict.

    The delta, the interval and the accept/reject decision all come from
    the gate, never from arithmetic here. `compare` needs >= 3 seeds on
    the candidate for a CONFIRM-stage verdict; with exactly 1 it runs the
    SCREEN stage, which can only ever reject.
    """
    verdict = gate.compare(candidate, incumbent)
    print()
    print(f"=== {label} ===")
    print(f"  accept         : {verdict.accept}")
    print(f"  delta          : {verdict.delta:+.5f}")
    print(f"  ci95           : ({verdict.ci95[0]:+.5f}, {verdict.ci95[1]:+.5f})")
    print(f"  n_seeds        : {verdict.n_seeds}")
    backtest_delta = verdict.backtest_delta
    print(
        "  backtest_delta : "
        + ("none" if backtest_delta is None else f"{backtest_delta:+.5f}")
    )
    print(f"  reason         : {verdict.reason}")
