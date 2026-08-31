"""The unattended agent run, driven by the REAL Controller.

WHAT THIS IS FOR. scripts/run_agent.py demonstrates the ten scripted
moves, but it does so with a hand-rolled loop: its own node counter, its
own per-slot failure dict, its own hand-constructed baseline Verdict. The
audited state machine in controller/controller.py — stages, budget,
convergence, circuit breaker, RUN_END — is not involved. For an autonomy
artifact that matters: the claim we want to make is "our agent ran
unattended", and the agent is the Controller.

This script wires the Controller to the real components:

    executor   autonomy.RunCandidateExecutor over executor.run.run_candidate
    gate       harness.gate (the module object satisfies GatePort as-is)
    generator  autonomy.SlotScriptedGenerator over methods.scripted's moves
    realizer   autonomy.MovesRealizer (lookup, not inference)
    policy     controller.policy.FixedOrderPolicy (no randomness)
    journal    executor.journal.Journal, durable JSONL + fsync per append
    budget     contracts.Budget

No prompts, no interactive input, no hardcoded wall-clock: the elapsed
time reported to executor.report.render is measured here.


WHY THE DEFAULT POLICY ORDER OMITS `features`
---------------------------------------------
The ten scripted moves target five slots — model (4), objective (2),
calibration (2), weighting (1), data_view (1) — and never `features`.
FixedOrderPolicy steps over slots that are not on offer, but a slot IN
its order that the stage DOES offer will be selected, and asking the
generator for a slot it has no move for raises GeneratorExhausted, which
ends the whole run (controller.py treats it as a normal termination and
still finalises, but the run stops). `features` is offered by both
structural stages, so leaving it in the order would end the run early
for no useful reason. It is omitted from the default rather than special
-cased anywhere, and --policy-order can put it back.

The same mechanism still bounds the run: any slot's moves can run out.
That is a real property of driving a ten-move script through a search
loop that may ask for more, and it terminates cleanly through FINALIZE
and RUN_END rather than by crashing.


KNOWN GAP: executor/report.py CANNOT RENDER A CONTROLLER JOURNAL YET
--------------------------------------------------------------------
Rendering is therefore OPT-IN (`--report`), default off. This is a real
incompatibility found while wiring this up, not a preference, and it is
two separate payload-shape mismatches with the same root cause: the
durable journal's `log_*` helpers and the Controller's own `_emit` write
different payloads for the same EventKind, and executor/report.py was
written against the helpers.

  CONVERGENCE_CHECK — FATAL. Journal.log_convergence_check writes
  {delta, epsilon, clears_epsilon, accept}; the Controller writes
  {iteration_definition, converged, by_rule, organizers_converged,
  internal_converged, recent_deltas, recent_significant,
  iterations_considered, epsilon, n_required}. report.py's
  `_format_convergence_check` does `f"{payload.get('delta'):+.5f}"`,
  which raises TypeError on None. The render dies partway through
  iterations.md.

  EVAL_RESULT — cosmetic. Journal.log_eval_result writes `per_seed`;
  the Controller writes {config_id, status, primary, wall_seconds,
  gpu_seconds, tokens}. report.py reads per-seed metrics out of
  `per_seed`, so every metric row would render `n/a` even if the crash
  above were fixed.

Both payload shapes are correct for their authors — contracts.py is
explicit that payload shape is "documented, not enforced" and varies by
EventKind. They are simply not the same shape. Reconciling them means
editing executor/report.py (Terry's file) or writing a W2-owned
renderer; both are out of scope here. tests/test_run_controller.py pins
both mismatches so this note cannot quietly outlive its cause.

Pass --report anyway to see it fail, or to pick up whatever renders once
the shapes are reconciled.

Usage:
    python scripts/run_controller.py
    python scripts/run_controller.py --max-nodes-per-stage 2 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts import Budget, EventKind, SlotName  # noqa: E402
from controller.controller import Controller  # noqa: E402
from controller.policy import FixedOrderPolicy  # noqa: E402
from executor import report  # noqa: E402
from executor.journal import Journal  # noqa: E402
from harness import gate  # noqa: E402

from autonomy.adapters import (  # noqa: E402
    DurableJournal,
    MovesRealizer,
    RunCandidateExecutor,
    SlotScriptedGenerator,
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
"""Three seeds so harness.gate takes its CONFIRM path.

Not a detail, and the same reason tests/test_false_positive_rate.py gives:
with one seed the gate dispatches to SCREEN, which hardcodes accept=False
and can only ever reject.
"""

DEFAULT_POLICY_ORDER: tuple[SlotName, ...] = (
    "model",
    "objective",
    "weighting",
    "data_view",
    "calibration",
)
"""Every slot the ten scripted moves actually target. See module docstring."""

DEFAULT_MAX_NODES_PER_STAGE = 2
"""Small on purpose. Three search stages x this many attempts, and each
attempt that reaches a real FM trains once per seed (~350-450s for three
seeds, measured — see scripts/run_agent.py's notes on moves 5 and 10). A
default of 2 keeps a demonstration run to a few candidates rather than a
few dozen; raise it deliberately."""

DEFAULT_JOURNAL_PATH = REPO_ROOT / "artifacts" / "journal_controller.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "artifacts" / "report_controller"


def parse_seeds(raw: str) -> tuple[int, ...]:
    """"0,1,2" -> (0, 1, 2). Rejects an empty list loudly.

    Mirrors manual/run.py's own parse_seeds for the same reason: a run
    with no seeds trains nothing and reports zeros, which looks like a
    result rather than like a mistake.
    """
    seeds = tuple(int(part) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError(f"no seeds parsed from {raw!r}; expected e.g. '0,1,2'")
    return seeds


def parse_slots(raw: str) -> tuple[SlotName, ...]:
    """"model,objective" -> ("model", "objective"). Rejects empty."""
    slots = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not slots:
        raise ValueError(f"no slots parsed from {raw!r}; expected e.g. 'model,objective'")
    return slots  # type: ignore[return-value]


def build_controller(
    *,
    journal: Journal,
    seeds: Sequence[int],
    max_nodes_per_stage: int,
    policy_order: Sequence[SlotName],
    run_id: str,
    failures_before_block: int = 2,
) -> tuple[Controller, RunCandidateExecutor]:
    """Wire the real Controller. Returns it and the executor adapter,
    which carries the delegation log the run summary prints.

    Split out from main() so the wiring is testable without launching a
    multi-hour training run — the same discipline manual/run.py applies
    to its own argument parsing.
    """
    executor = RunCandidateExecutor(
        # No journal: the Controller is the sole journaller for a
        # Controller-driven run. See RunCandidateExecutor's docstring.
        journal=None,
    )
    controller = Controller(
        executor=executor,
        # The module object itself. ports.GatePort's docstring spells out
        # why no adapter is needed, and tests/test_false_positive_rate.py
        # already wires the real gate exactly this way.
        gate=gate,
        generator=SlotScriptedGenerator(),
        realizer=MovesRealizer(),
        policy=FixedOrderPolicy(policy_order),
        journal=DurableJournal(journal),
        budget=Budget(),
        seeds=tuple(seeds),
        max_nodes_per_stage=max_nodes_per_stage,
        failures_before_block=failures_before_block,
        run_id=run_id,
    )
    return controller, executor


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_controller.py",
        description=(
            "Drive the real controller.Controller against the real executor "
            "and the real noise gate, unattended, writing a durable journal."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma-separated seeds (default: 0,1,2; three is the gate's CONFIRM bar)",
    )
    parser.add_argument(
        "--max-nodes-per-stage",
        type=int,
        default=DEFAULT_MAX_NODES_PER_STAGE,
        help=(
            "candidate attempts per search stage (default: "
            f"{DEFAULT_MAX_NODES_PER_STAGE}). Three search stages, so the run "
            "attempts at most 1 + 3x this many candidates."
        ),
    )
    parser.add_argument(
        "--policy-order",
        default=",".join(DEFAULT_POLICY_ORDER),
        help=(
            "comma-separated slot order for FixedOrderPolicy (default: the "
            "five slots the scripted moves target; 'features' is omitted "
            "because no scripted move targets it)"
        ),
    )
    parser.add_argument(
        "--failures-before-block",
        type=int,
        default=2,
        help="consecutive executor failures before a slot is blocked (default: 2)",
    )
    parser.add_argument(
        "--journal",
        default=str(DEFAULT_JOURNAL_PATH),
        help=f"durable journal path (default: {DEFAULT_JOURNAL_PATH})",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help=f"where executor.report.render writes (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "run id stamped on every journal event (default: a fresh "
            "run-<uuid12>, so two runs sharing a journal file stay "
            "distinguishable)"
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help=(
            "also render executor.report into --report-dir. OFF BY DEFAULT: "
            "executor/report.py cannot render a Controller-produced journal "
            "today and raises TypeError on the CONVERGENCE_CHECK payload — "
            "see this module's docstring for both mismatches"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    seeds = parse_seeds(args.seeds)
    policy_order = parse_slots(args.policy_order)
    run_id = args.run_id or f"run-{uuid.uuid4().hex[:12]}"

    journal_path = Path(args.journal)
    report_dir = Path(args.report_dir)

    print("controller run")
    print(f"  run_id               = {run_id}")
    print(f"  seeds                = {seeds}")
    print(f"  max_nodes_per_stage  = {args.max_nodes_per_stage}")
    print(f"  policy_order         = {policy_order}")
    print(f"  journal              = {journal_path}")
    print(f"  report_dir           = {report_dir}")
    print()

    journal = Journal(str(journal_path), run_id=run_id)
    controller, executor = build_controller(
        journal=journal,
        seeds=seeds,
        max_nodes_per_stage=args.max_nodes_per_stage,
        policy_order=policy_order,
        run_id=run_id,
        failures_before_block=args.failures_before_block,
    )

    start = time.perf_counter()
    state = controller.run()
    elapsed = time.perf_counter() - start

    print()
    print("run complete")
    print(f"  stage                = {state.stage.value}")
    print(f"  iteration            = {state.iteration}")
    print(f"  node                 = {state.node}")
    print(f"  incumbent_config_id  = {state.incumbent_config.config_id if state.incumbent_config else None}")
    print(f"  blocked_slots        = {sorted(state.blocked_slots)}")
    print(f"  elapsed              = {elapsed:.1f}s  (measured, not assumed)")

    print()
    print(f"  executor delegations ({len(executor.calls)}):")
    for fragment, target_slot, delegated_seeds in executor.calls:
        print(f"    {target_slot:<12} impl={fragment.impl!r:<24} seeds={delegated_seeds}")

    events = Journal.replay(str(journal_path))
    this_run = [e for e in events if e.run_id == run_id]
    kinds: dict[str, int] = {}
    for event in this_run:
        kinds[event.kind.value] = kinds.get(event.kind.value, 0) + 1
    print()
    print(f"  journal              = {journal_path} ({len(this_run)} events this run)")
    for kind in sorted(kinds):
        print(f"    {kind:<20} {kinds[kind]}")
    terminal = this_run[-1].kind if this_run else None
    print(
        f"  terminal event       = {terminal.value if terminal else None}"
        f"{'  (clean finish)' if terminal is EventKind.RUN_END else '  (NOT run_end — the run did not finish through its normal path)'}"
    )

    if args.report:
        # Deliberately unguarded. If this raises, the run itself already
        # finished and its journal is durable on disk — swallowing the
        # error would hide a real incompatibility behind a report that
        # silently is not there. See the module docstring.
        report.render(str(journal_path), str(report_dir), training_wall_clock_seconds=elapsed)
        print()
        print(f"  report rendered to {report_dir}:")
        for path in sorted(report_dir.iterdir()):
            print(f"    {path.name} ({path.stat().st_size} bytes)")
    else:
        print()
        print("  report               = not rendered (--report is off by default;")
        print("                         executor/report.py cannot render a Controller")
        print("                         journal yet — see this script's docstring)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
