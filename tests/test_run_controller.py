"""Tests for scripts/run_controller.py — the unattended-run entrypoint.

WHAT IS AND IS NOT UNDER TEST. The argument contract and the component
wiring are, because both are cheap to get wrong and expensive to discover
wrong three hours into a training run — the same reasoning manual/run.py
gives for splitting `_parse_args` out of `main`. Actually running the
Controller is NOT: `build_controller` returns a controller wired to the
real executor.run.run_candidate, and calling `.run()` on it would train
factorization machines. No test here calls it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from contracts import Budget
from controller.controller import Controller
from controller.policy import FixedOrderPolicy
from controller.ports import ExecutorPort, GeneratorPort, JournalPort, RealizerPort
from executor.journal import Journal
from harness import gate as harness_gate

from autonomy.adapters import (
    DurableJournal,
    MovesRealizer,
    RunCandidateExecutor,
    ScriptedMoves,
    SlotScriptedGenerator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_run_controller():
    """Import scripts/run_controller.py by path.

    scripts/ is not a package (no __init__.py), so this is the same
    file-path import the repo already uses for vendor modules in
    harness/data.py and harness/validate.py.
    """
    spec = importlib.util.spec_from_file_location(
        "_scripts_run_controller", REPO_ROOT / "scripts" / "run_controller.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_controller = _load_run_controller()


# ---------------------------------------------------------------------------
# Argument contract
# ---------------------------------------------------------------------------


def test_defaults_are_the_documented_ones():
    args = run_controller._parse_args([])

    assert run_controller.parse_seeds(args.seeds) == (0, 1, 2)
    assert args.max_nodes_per_stage == 2
    assert args.failures_before_block == 2
    # Rendering is ON by default: executor/report.py now reads the
    # Controller's payload shapes. See the two tests at the bottom of
    # this file for exactly what it does and does not render.
    assert args.report is True
    assert args.run_id is None


def test_default_policy_order_covers_every_slot_the_script_targets():
    """A slot in the policy order with no scripted move raises
    GeneratorExhausted, which ends the run. The default must name only
    slots the catalog can serve — and all of them, so nothing is
    unreachable."""
    catalog = ScriptedMoves()
    order = run_controller.DEFAULT_POLICY_ORDER

    for slot in order:
        assert catalog.for_slot(slot), f"{slot} is in the default order but has no moves"
    served = {h.target_slot for _, h in catalog._moves}
    assert set(order) == served
    assert "features" not in order


@pytest.mark.parametrize(
    "raw, expected",
    [("0", (0,)), ("0,1,2", (0, 1, 2)), (" 3 , 4 ", (3, 4)), ("5,", (5,))],
)
def test_parse_seeds(raw, expected):
    assert run_controller.parse_seeds(raw) == expected


def test_parse_seeds_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no seeds parsed"):
        run_controller.parse_seeds(" , ")


def test_parse_slots_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no slots parsed"):
        run_controller.parse_slots("  ")


def test_cli_overrides_reach_the_namespace():
    args = run_controller._parse_args(
        ["--seeds", "7,8", "--max-nodes-per-stage", "5", "--policy-order", "model", "--report"]
    )

    assert run_controller.parse_seeds(args.seeds) == (7, 8)
    assert args.max_nodes_per_stage == 5
    assert run_controller.parse_slots(args.policy_order) == ("model",)
    assert args.report is True


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_build_controller_wires_every_port_to_a_real_component(tmp_path):
    """The point of this whole change: the audited state machine, not a
    hand-rolled loop, and every seam filled by something real rather than
    a double."""
    journal = Journal(str(tmp_path / "j.jsonl"), run_id="run-1")

    controller, executor = run_controller.build_controller(
        journal=journal,
        seeds=(0, 1, 2),
        max_nodes_per_stage=2,
        policy_order=run_controller.DEFAULT_POLICY_ORDER,
        run_id="run-1",
    )

    assert isinstance(controller, Controller)
    assert isinstance(executor, RunCandidateExecutor)
    assert isinstance(executor, ExecutorPort)
    # The gate is the real module object, not a double.
    assert controller._gate is harness_gate
    assert isinstance(controller._generator, SlotScriptedGenerator)
    assert isinstance(controller._generator, GeneratorPort)
    assert isinstance(controller._realizer, MovesRealizer)
    assert isinstance(controller._realizer, RealizerPort)
    assert isinstance(controller._policy, FixedOrderPolicy)
    assert isinstance(controller._journal, DurableJournal)
    assert isinstance(controller._journal, JournalPort)
    assert isinstance(controller._budget, Budget)
    assert controller._seeds == (0, 1, 2)
    assert controller._run_id == "run-1"


def test_build_controller_delegates_to_the_real_run_candidate(tmp_path):
    """Not a stub: the adapter's default runner is executor.run's own
    function, so an unattended run trains for real."""
    from executor.run import run_candidate

    journal = Journal(str(tmp_path / "j.jsonl"), run_id="run-1")
    _, executor = run_controller.build_controller(
        journal=journal,
        seeds=(0,),
        max_nodes_per_stage=1,
        policy_order=("model",),
        run_id="run-1",
    )

    assert executor._runner is run_candidate


def test_build_controller_keeps_the_executor_out_of_the_journal(tmp_path):
    """Two journallers for one attempt would write overlapping accounts at
    disagreeing node numbers. The Controller owns the journal."""
    journal = Journal(str(tmp_path / "j.jsonl"), run_id="run-1")

    _, executor = run_controller.build_controller(
        journal=journal,
        seeds=(0,),
        max_nodes_per_stage=1,
        policy_order=("model",),
        run_id="run-1",
    )

    assert executor._journal is None


def test_build_controller_passes_the_policy_order_through(tmp_path):
    journal = Journal(str(tmp_path / "j.jsonl"), run_id="run-1")

    controller, _ = run_controller.build_controller(
        journal=journal,
        seeds=(0,),
        max_nodes_per_stage=1,
        policy_order=("objective", "model"),
        run_id="run-1",
    )

    assert controller._policy.order == ("objective", "model")


# ---------------------------------------------------------------------------
# main() end to end, with training stubbed out
# ---------------------------------------------------------------------------


def _stub_training(monkeypatch):
    """Replace the real executor with a synthetic one.

    Patches `autonomy.adapters._run_candidate` rather than
    `executor.run.run_candidate`, because the adapter binds the function
    at import time; patching the original module would leave the already
    -bound reference untouched and quietly train an FM inside a unit test.
    """
    from contracts import CandidateResult, Metrics, Status

    def fake_run_candidate(fragment, target_slot, seeds=(0, 1, 2), journal=None):
        return CandidateResult(
            config_id=f"stub_{fragment.impl}",
            status=Status.OK,
            val={s: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60}) for s in seeds},
            backtest={s: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60}) for s in seeds},
            wall_seconds=0.0,
        )

    monkeypatch.setattr("autonomy.adapters._run_candidate", fake_run_candidate)


def test_main_runs_and_journals_without_training(tmp_path, monkeypatch, capsys):
    """The whole entrypoint — argument parsing, wiring, Controller.run and
    the run summary — on a stubbed executor. The REAL gate participates."""
    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal_controller.jsonl"

    exit_code = run_controller.main(
        [
            "--seeds", "0,1,2",
            "--max-nodes-per-stage", "1",
            "--journal", str(journal_path),
            "--run-id", "smoke-run",
        ]
    )

    assert exit_code == 0

    replayed = Journal.replay(str(journal_path))
    assert replayed[0].kind.value == "run_start"
    assert replayed[-1].kind.value == "run_end"
    assert all(e.run_id == "smoke-run" for e in replayed)

    out = capsys.readouterr().out
    assert "run complete" in out
    assert "(clean finish)" in out
    # The elapsed figure is measured, never a constant.
    assert "elapsed" in out and "measured, not assumed" in out
    # Rendering is on by default, so the summary names the rendered
    # files rather than explaining why there are none.
    assert "report rendered to" in out
    assert "not rendered" not in out


def test_a_second_run_into_the_same_journal_stays_distinguishable(tmp_path, monkeypatch):
    """Journal.__init__ silently picks up an existing file. Two runs
    sharing one file must remain separable by run_id — which is why this
    script generates a fresh one per run instead of hardcoding a constant
    the way every other script in scripts/ does."""
    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal_controller.jsonl"

    for run_id in ("run-one", "run-two"):
        run_controller.main(
            ["--max-nodes-per-stage", "1", "--journal", str(journal_path), "--run-id", run_id]
        )

    replayed = Journal.replay(str(journal_path))
    assert {e.run_id for e in replayed} == {"run-one", "run-two"}
    for run_id in ("run-one", "run-two"):
        this_run = [e for e in replayed if e.run_id == run_id]
        assert this_run[0].kind.value == "run_start"
        assert this_run[-1].kind.value == "run_end"


# ---------------------------------------------------------------------------
# What --report renders now that report.py reads both payload shapes
#
# These two tests replace a pair that pinned the OLD incompatibility
# (a TypeError, and every metric row reading `n/a`). report.py's
# capability genuinely changed, so the expectation changed with it:
# the render must now SUCCEED and carry real numbers. The second test
# also pins the one thing the reconciliation did NOT fix, and cannot —
# see its docstring.
# ---------------------------------------------------------------------------


def test_report_render_succeeds_on_a_controller_convergence_check(tmp_path, monkeypatch):
    """THE FATAL MISMATCH, now fixed — pinned from the other side.

    The Controller's CONVERGENCE_CHECK payload still carries
    {converged, by_rule, recent_deltas, epsilon, n_required, ...} and
    still has no `delta` key: the payload did not change, report.py did.
    `_format_convergence_check` now branches on the shape instead of
    formatting a missing `delta` with `:+.5f` and dying on the None.

    So this asserts the same payload facts as before, and then that the
    render completes and puts the Controller's own convergence vocabulary
    into iterations.md.
    """
    from executor.report import render

    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal_controller.jsonl"
    run_controller.main(
        ["--max-nodes-per-stage", "1", "--journal", str(journal_path), "--run-id", "r",
         "--report-dir", str(tmp_path / "rundir")]
    )

    replayed = Journal.replay(str(journal_path))
    convergence = [e for e in replayed if e.kind.value == "convergence_check"]
    assert convergence, "the run produced no CONVERGENCE_CHECK to characterise"
    # Unchanged: the Controller payload is what it always was.
    assert "delta" not in convergence[0].payload
    assert "clears_epsilon" not in convergence[0].payload
    assert "epsilon" in convergence[0].payload

    # Changed: this no longer raises TypeError.
    out_dir = tmp_path / "report"
    render(str(journal_path), output_dir=str(out_dir))

    # The render ran to completion rather than dying partway through
    # iterations.md, so every document exists.
    written = {path.name for path in out_dir.iterdir()}
    assert {"results.md", "iterations.md", "trajectory.csv"} <= written

    iterations = (out_dir / "iterations.md").read_text(encoding="utf-8")
    assert "**Convergence check**" in iterations
    # Formatted from the Controller's own keys, not the helper's.
    assert "epsilon=" in iterations
    assert "recent_deltas=[" in iterations
    assert "iterations_considered=" in iterations


def test_controller_eval_result_renders_a_real_primary_but_no_gauc_or_ndcg(
    tmp_path, monkeypatch
):
    """THE COSMETIC MISMATCH — half fixed, half not fixable, and the
    difference matters when reading results.md.

    FIXED: report.py's `_eval_result_metrics` reads both shapes, so
    `primary` comes off the Controller's flat payload and renders as a
    real number everywhere it appears.

    NOT FIXED, AND NOT FIXABLE IN report.py: GAUC and nDCG@5 still read
    `n/a`, because the Controller's EVAL_RESULT payload does not CARRY
    them — it writes {config_id, status, primary, wall_seconds,
    gpu_seconds, tokens} and nothing else, and `primary` is already its
    own seed-blended mean. Those `n/a`s are missing data, not a broken
    renderer. This test pins that distinction so nobody 'fixes'
    report.py chasing them, and so the day the Controller starts
    emitting per-metric values this test fails and says so.
    """
    from executor.report import render

    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal_controller.jsonl"
    run_controller.main(
        ["--max-nodes-per-stage", "1", "--journal", str(journal_path), "--run-id", "r",
         "--report-dir", str(tmp_path / "rundir")]
    )

    # Unchanged: the payload is still the flat Controller shape.
    results = [e for e in Journal.replay(str(journal_path)) if e.kind.value == "eval_result"]
    assert results
    assert "per_seed" not in results[0].payload
    assert {"config_id", "status", "primary", "wall_seconds", "gpu_seconds", "tokens"} <= set(
        results[0].payload
    )
    assert "GAUC" not in str(results[0].payload)

    out_dir = tmp_path / "report"
    render(str(journal_path), output_dir=str(out_dir))
    text = (out_dir / "results.md").read_text(encoding="utf-8")

    # The stub evaluates every candidate at primary 0.60, so the
    # best-primary row is a real number and its delta against the
    # official baseline (0.6016) is computed, not skipped.
    assert "| Validation-best primary | 0.6000 |" in text
    assert "| Validation-best primary | n/a |" not in text
    assert "Delta vs official baseline primary (0.6016) | -0.0016 |" in text

    # ...and the two the payload does not carry are still n/a.
    assert "| Validation-best GAUC | n/a |" in text
    assert "| Validation-best nDCG@5 | n/a |" in text

    # The same real primary reaches the per-node narrative and the CSV.
    assert "primary: 0.6000" in (out_dir / "iterations.md").read_text(encoding="utf-8")
    csv_text = (out_dir / "trajectory.csv").read_text(encoding="utf-8")
    assert "0.600000" in csv_text
