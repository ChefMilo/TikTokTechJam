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


RUN INTEGRITY — WHAT MAKES "0 INTERVENTIONS" MEAN ANYTHING
-----------------------------------------------------------
A count of zero proves nothing when the event is simply never emitted,
which is what the repo produced before this. So the run records what it
CHECKED, positively, in its own journal:

  AT LAUNCH, before anything is appended, autonomy.integrity captures the
  commit, whether the working tree was clean, and a sha256 over the source
  that actually runs (executor/, harness/, controller/, methods/,
  autonomy/). It also reads whatever is already in the target journal and
  classifies why this process is starting — fresh, an autonomous resume,
  or a manual restart.

  DURING THE RUN, CheckedExecutor re-hashes that source before every
  candidate. A node boundary is the natural checkpoint and costs ~5ms
  against evaluations measured in hundreds of seconds. Drift is logged
  once as INTERVENTION(type="code_changed_midrun").

  AT BOTH ENDS, the Controller reads its `run_metadata` to build RUN_START
  and RUN_END, and that read is itself an integrity check (see
  IntegrityMetadata). RUN_END therefore carries a summary that is true at
  RUN_END: commit, tree state at launch, how many checks ran, whether the
  fingerprint held, and how many interventions were counted.

Only genuine manual touches become EventKind.INTERVENTION events, so this
script's tally and executor/report.py's existing counter are the same
number by construction — the run cross-checks them before exiting and
returns 2 if they disagree. An autonomous resume is recorded in RUN_START
metadata and deliberately counted as nothing.

autonomy/INTERVENTION_POLICY.md is the reviewable definition of what
counts. Read that before trusting the number.

`--resume` says this launch is an autonomous relaunch. It does NOT resume
the Controller's state — the run starts fresh at Stage.INIT — and the
policy document is explicit that calling it a resume of the RUN would
overclaim.


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


REPORT RENDERING: ON BY DEFAULT (was opt-in)
--------------------------------------------------------------------
executor/report.py now reads BOTH journal payload shapes, so rendering
is the default and `--no-report` turns it off. What follows is why it
was ever opt-in, and what the reconciliation did and did not fix —
because one half is fixed and the other half is not a bug at all.

The two shapes: the durable journal's `log_*` helpers and the
Controller's own `_emit` write different payloads for the same
EventKind, and executor/report.py was originally written against the
helpers only.

  CONVERGENCE_CHECK — WAS FATAL, NOW FIXED. Journal.log_convergence_check
  writes {delta, epsilon, clears_epsilon, accept}; the Controller writes
  {iteration_definition, converged, by_rule, organizers_converged,
  internal_converged, recent_deltas, recent_significant,
  iterations_considered, epsilon, n_required}. report.py's
  `_format_convergence_check` used to do `f"{payload.get('delta'):+.5f}"`,
  which raised TypeError on None and killed the render partway through
  iterations.md. It now branches on `"delta" in payload` and formats the
  Controller's multi-rule shape on its own terms.

  EVAL_RESULT — NOT A BUG, AND STILL VISIBLE. Journal.log_eval_result
  writes `per_seed`; the Controller writes {config_id, status, primary,
  wall_seconds, gpu_seconds, tokens}. report.py's `_eval_result_metrics`
  now reads both, so `primary` renders a REAL number from a Controller
  journal (results.md's "Validation-best primary" and its delta vs the
  official baseline, iterations.md's per-node primary, trajectory.csv's
  val_primary column).

  But GAUC and nDCG@5 still render `n/a` on a Controller journal, and
  no change to report.py can fix that: the Controller's flat payload
  does not CARRY GAUC or nDCG@5. `primary` is already its own
  seed-blended mean (controller/controller.py's `_mean_primary`). Those
  `n/a`s are missing data, not a rendering failure, and reading them as
  a regression is the mistake this paragraph exists to prevent.

Both payload shapes are correct for their authors — contracts.py is
explicit that payload shape is "documented, not enforced" and varies by
EventKind. tests/test_run_controller.py pins the reconciled behaviour:
that the render succeeds, that primary is real, and that GAUC/nDCG@5
are `n/a` for the payload-content reason above.

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
from typing import Mapping, Optional, Sequence

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
from autonomy.integrity import (  # noqa: E402
    MIN_CANDIDATES_FOR_VERIFIED,
    UNKNOWN,
    CheckedExecutor,
    IntegrityMetadata,
    IntegrityMonitor,
    classify_relaunch,
    launch_fingerprint,
)
from autonomy.render import render_autonomy  # noqa: E402

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
    monitor: Optional[IntegrityMonitor] = None,
    run_metadata: Optional[Mapping[str, object]] = None,
) -> tuple[Controller, RunCandidateExecutor]:
    """Wire the real Controller. Returns it and the executor adapter,
    which carries the delegation log the run summary prints.

    Split out from main() so the wiring is testable without launching a
    multi-hour training run — the same discipline manual/run.py applies
    to its own argument parsing.

    `monitor` wraps the executor in CheckedExecutor so the source is
    re-verified before every candidate; `run_metadata` reaches the
    Controller's RUN_START and RUN_END payloads. Both default to None so
    the Controller can still be built without the integrity machinery,
    which is what the adapter-level tests do.
    """
    executor = RunCandidateExecutor(
        # No journal: the Controller is the sole journaller for a
        # Controller-driven run. See RunCandidateExecutor's docstring.
        journal=None,
    )
    # The Controller sees the wrapper; the caller keeps the adapter, whose
    # `calls` log the run summary prints. CheckedExecutor delegates
    # unknown attributes, so either reference reaches the same state.
    port = executor if monitor is None else CheckedExecutor(executor, monitor)
    controller = Controller(
        executor=port,
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
        run_metadata=run_metadata,
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
        "--require-clean",
        action="store_true",
        help=(
            "refuse to start unless git reports a clean working tree and a "
            "known commit. OFF by default so development runs work; the real "
            "artifact run should pass it, because a run that cannot say what "
            "code it ran cannot claim to have run it unattended"
        ),
    )
    parser.add_argument(
        "--min-verified-candidates",
        type=int,
        default=MIN_CANDIDATES_FOR_VERIFIED,
        help=(
            "how many candidates must be integrity-checked before the run may "
            f"call itself VERIFIED AUTONOMOUS (default: "
            f"{MIN_CANDIDATES_FOR_VERIFIED}). A floor against a zero-work run "
            "claiming a clean badge for doing nothing"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "this launch is an autonomous relaunch of an interrupted run at "
            "--journal. Combined with an unchanged source fingerprint, the "
            "relaunch is recorded as autonomous and NOT counted as a manual "
            "intervention. Note this does NOT resume the Controller's state: "
            "the run starts fresh at Stage.INIT. See "
            "autonomy/INTERVENTION_POLICY.md"
        ),
    )
    parser.add_argument(
        "--report",
        dest="report",
        action="store_true",
        default=True,
        help=(
            "also render executor.report into --report-dir. ON BY DEFAULT: "
            "executor/report.py now reads the Controller's payload shapes. "
            "Kept as an explicit no-op opt-in so existing invocations that "
            "pass it keep working — see --no-report to turn rendering off"
        ),
    )
    parser.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help=(
            "skip the executor.report render. The autonomy section is "
            "written either way; the two documents fail independently"
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

    # --- integrity: establish provenance BEFORE anything else runs -----
    #
    # Order matters. The relaunch classification is read off whatever is
    # already in the journal, so it must be computed before this run
    # appends anything to it.
    launch = launch_fingerprint()
    prior_events = Journal.replay(str(journal_path))
    relaunch = classify_relaunch(
        prior_events,
        code_hash=launch["code_hash"],
        resume_requested=args.resume,
    )

    tree_state = {True: "DIRTY", False: "clean", None: "unknown"}[launch["dirty"]]
    if launch["dirty"]:
        tree_state += f" ({len(launch['dirty_files'])} uncommitted file(s))"
    counted = " (COUNTS as a manual intervention)" if relaunch.counts_as_intervention else ""

    print("integrity")
    print(f"  commit               = {launch['commit']}")
    print(f"  tree at launch       = {tree_state}")
    print(f"  code_hash            = {launch['code_hash'][:16]}...")
    print(f"  relaunch             = {relaunch.kind}{counted}")
    print(f"    {relaunch.reason}")
    print()

    # --- --require-clean: refuse BEFORE touching the journal -----------
    #
    # Checked here, before the Journal is even constructed, so a refused
    # launch leaves no trace. Appending a RUN_START and then bailing would
    # put an interrupted run in the file, which the next launch would
    # correctly classify as a manual restart — manufacturing an
    # intervention out of a run that never started.
    if args.require_clean and launch["dirty"] is not False:
        print("REFUSING TO START (--require-clean)")
        if launch["dirty"] is None:
            print("  git could not determine the working tree state, so this")
            print("  run's provenance could not be pinned to a commit.")
        else:
            print(f"  the working tree has {len(launch['dirty_files'])} uncommitted change(s):")
            for entry in launch["dirty_files"]:
                print(f"    {entry}")
        if launch["commit"] == UNKNOWN:
            print("  the commit could not be determined either.")
        print()
        print("  An artifact run must start from a committed, clean tree — a")
        print("  run that cannot say what code it ran cannot claim to have run")
        print("  it unattended. Commit or stash, then relaunch. Drop")
        print("  --require-clean for a development run that does not need the")
        print("  provenance claim.")
        return 3

    journal = Journal(str(journal_path), run_id=run_id)
    monitor = IntegrityMonitor(
        launch=launch,
        # The journal's own helper. Every call becomes an
        # EventKind.INTERVENTION event, which is exactly what
        # executor/report.py counts — so its number and ours cannot
        # disagree. See autonomy/INTERVENTION_POLICY.md.
        on_intervention=journal.log_intervention,
        min_candidates=args.min_verified_candidates,
    )
    # Logged BEFORE the Controller's RUN_START, so the journal reads in the
    # order the events happened: a human restarted this, then the run
    # began. An autonomous resume records nothing here — it is visible in
    # RUN_START metadata instead, and counting it would inflate the very
    # number it is supposed to leave alone.
    monitor.record_relaunch(relaunch)

    controller, executor = build_controller(
        journal=journal,
        seeds=seeds,
        max_nodes_per_stage=args.max_nodes_per_stage,
        policy_order=policy_order,
        run_id=run_id,
        failures_before_block=args.failures_before_block,
        monitor=monitor,
        run_metadata=IntegrityMetadata(monitor, relaunch=relaunch.as_payload()),
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

    # --- integrity: the positive record -------------------------------
    #
    # Printed from the same summary that RUN_END carries, so the console
    # and the journal cannot tell different stories. Note the summary is
    # NOT recomputed here: the authoritative one was built when the
    # Controller read run_metadata to emit RUN_END, which is also when the
    # final check ran. See IntegrityMetadata.
    summary = monitor.summary()
    print()
    print("  integrity summary")
    print(f"    {monitor.one_line()}")
    print(f"    code fingerprint checks = {summary['checks_performed']}")
    print(f"    fingerprint stable      = {summary['code_fingerprint_stable']}")
    print(f"    candidates checked      = {summary['candidates_checked']} "
          f"(floor {summary['min_candidates_for_verified']})")
    print(f"    manual interventions    = {summary['manual_interventions']}")
    if summary["intervention_types"]:
        for entry in monitor.interventions:
            print(f"      - {entry['type']}: {entry['reason']}")
    print(f"    VERIFIED AUTONOMOUS     = {summary['verified']}")
    for reason in summary["unverified_because"]:
        print(f"      not verified: {reason}")
    # THE CROSS-CHECK. The number executor/report.py will render is the
    # count of INTERVENTION events in the journal; the number printed above
    # is the monitor's own tally. If those ever disagree, two artifacts
    # make different claims about the same run and neither can be trusted.
    #
    # An explicit check rather than `assert`, for the reason harness/data.py
    # and harness/metrics.py give for theirs: `python -O` strips asserts,
    # and this is a correctness invariant about the headline number, not a
    # debugging aid. Reported rather than raised — the run itself finished
    # and its journal is durable on disk, so tearing down here would
    # destroy nothing but would hide that the run completed. The exit code
    # carries the failure instead.
    logged = [e for e in this_run if e.kind is EventKind.INTERVENTION]
    if len(logged) != summary["manual_interventions"]:
        print()
        print("  *** INTEGRITY ACCOUNTING MISMATCH ***")
        print(f"    journal holds {len(logged)} INTERVENTION event(s)")
        print(f"    monitor counted {summary['manual_interventions']}")
        print("    the rendered report and this summary would disagree; do not")
        print("    publish either number until this is explained.")
        return 2

    # --- the autonomy section, ALWAYS written --------------------------
    #
    # Not gated behind --report. That flag exists because
    # executor/report.py cannot render a Controller journal yet; this
    # renderer reads the journal directly and shares none of that
    # machinery, so gating it behind an unrelated flag would make the
    # autonomy evidence hostage to a metric-rendering bug in someone
    # else's file. The two documents defend different claims and should
    # fail independently.
    render_autonomy(str(journal_path), str(report_dir), run_id=run_id)
    print()
    print(f"  autonomy section     = {report_dir / 'autonomy.md'}")

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
        print("    note: GAUC and nDCG@5 read n/a in results.md — the")
        print("          Controller's EVAL_RESULT payload does not carry them;")
        print("          primary is real. See this script's docstring.")
    else:
        print()
        print("  report               = not rendered (--no-report was passed;")
        print("                         rendering is ON by default)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
