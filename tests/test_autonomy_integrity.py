"""Tests for autonomy.integrity and its wiring into the unattended run.

FAST BY CONSTRUCTION: nothing here trains a factorization machine.
`code_fingerprint` is exercised against a throwaway directory tree, the
relaunch classifier is a pure function over replayed events, and the
end-to-end runs reuse PR1's stubbed-executor pattern (patching
`autonomy.adapters._run_candidate`, which the adapter binds at import
time).

WHAT THESE TESTS ARE REALLY FOR. The artifact's headline claim is a
number — "N manual interventions" — and a number is only as good as the
rule that produced it. So the assertions here are mostly about the RULE:
that an autonomous resume is not counted, that a manual restart is, that
a mid-run edit is caught once rather than a dozen times, and that the
journal's INTERVENTION count and the monitor's own tally can never
disagree. See autonomy/INTERVENTION_POLICY.md.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from contracts import CandidateResult, EventKind, Metrics, Status
from executor.journal import Journal

from autonomy.integrity import (
    DEFAULT_SOURCE_DIRS,
    UNKNOWN,
    CheckedExecutor,
    IntegrityMetadata,
    IntegrityMonitor,
    classify_relaunch,
    code_fingerprint,
    git_state,
    launch_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_run_controller():
    """Import scripts/run_controller.py by path — scripts/ is not a
    package, same file-path import harness/data.py uses for vendor code."""
    spec = importlib.util.spec_from_file_location(
        "_scripts_run_controller_integrity", REPO_ROOT / "scripts" / "run_controller.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_controller = _load_run_controller()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def source_tree(tmp_path):
    """A miniature repo whose 'source' we can edit under the fingerprint."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("B = 2\n", encoding="utf-8")
    (tmp_path / "pkg" / "notes.md").write_text("not source\n", encoding="utf-8")
    return tmp_path


def _stub_training(monkeypatch, *, results=None):
    """Replace the real executor. See PR1's identical helper for why the
    patch target is the adapter's bound name, not executor.run's."""
    results = results or {}

    def fake_run_candidate(fragment, target_slot, seeds=(0, 1, 2), journal=None):
        if fragment.impl in results:
            return results[fragment.impl]
        return CandidateResult(
            config_id=f"stub_{fragment.impl}",
            status=Status.OK,
            val={s: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60}) for s in seeds},
            backtest={s: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60}) for s in seeds},
            wall_seconds=0.0,
        )

    monkeypatch.setattr("autonomy.adapters._run_candidate", fake_run_candidate)


def _events(path):
    return Journal.replay(str(path))


def _interventions(path, run_id=None):
    return [
        e
        for e in _events(path)
        if e.kind is EventKind.INTERVENTION and (run_id is None or e.run_id == run_id)
    ]


def _run_start(path, run_id):
    for event in _events(path):
        if event.kind is EventKind.RUN_START and event.run_id == run_id:
            return event
    raise AssertionError(f"no RUN_START for {run_id}")


def _run_end(path, run_id):
    for event in _events(path):
        if event.kind is EventKind.RUN_END and event.run_id == run_id:
            return event
    raise AssertionError(f"no RUN_END for {run_id}")


# ---------------------------------------------------------------------------
# code_fingerprint
# ---------------------------------------------------------------------------


def test_code_fingerprint_is_stable_across_identical_inputs(source_tree):
    first = code_fingerprint(["pkg"], repo_root=source_tree)
    second = code_fingerprint(["pkg"], repo_root=source_tree)

    assert first == second
    assert len(first) == 64


def test_code_fingerprint_changes_when_a_source_file_changes(source_tree):
    before = code_fingerprint(["pkg"], repo_root=source_tree)

    (source_tree / "pkg" / "a.py").write_text("A = 2\n", encoding="utf-8")

    assert code_fingerprint(["pkg"], repo_root=source_tree) != before


def test_code_fingerprint_changes_when_a_source_file_is_added(source_tree):
    before = code_fingerprint(["pkg"], repo_root=source_tree)

    (source_tree / "pkg" / "c.py").write_text("C = 3\n", encoding="utf-8")

    assert code_fingerprint(["pkg"], repo_root=source_tree) != before


def test_code_fingerprint_changes_when_a_source_file_is_renamed(source_tree):
    """The path is hashed as well as the bytes, so a pure rename moves it."""
    before = code_fingerprint(["pkg"], repo_root=source_tree)

    (source_tree / "pkg" / "a.py").rename(source_tree / "pkg" / "renamed.py")

    assert code_fingerprint(["pkg"], repo_root=source_tree) != before


def test_code_fingerprint_ignores_non_python_files(source_tree):
    """Editing a README mid-run is not an intervention."""
    before = code_fingerprint(["pkg"], repo_root=source_tree)

    (source_tree / "pkg" / "notes.md").write_text("edited\n", encoding="utf-8")

    assert code_fingerprint(["pkg"], repo_root=source_tree) == before


def test_code_fingerprint_ignores_pycache(source_tree):
    before = code_fingerprint(["pkg"], repo_root=source_tree)

    cache = source_tree / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-312.pyc").write_bytes(b"\x00compiled")
    (cache / "stray.py").write_text("X = 1\n", encoding="utf-8")

    assert code_fingerprint(["pkg"], repo_root=source_tree) == before


def test_code_fingerprint_tolerates_a_missing_directory(source_tree):
    """DEFAULT_SOURCE_DIRS names five directories; a checkout missing one
    must fingerprint rather than raise."""
    assert code_fingerprint(["pkg", "does_not_exist"], repo_root=source_tree)


# ---------------------------------------------------------------------------
# git_state / launch_fingerprint
# ---------------------------------------------------------------------------


def test_launch_fingerprint_returns_the_documented_keys():
    fingerprint = launch_fingerprint()

    assert set(fingerprint) == {
        "commit",
        "dirty",
        "dirty_files",
        "code_hash",
        "source_dirs",
        "python_version",
        "platform",
        "started_ts",
    }
    assert len(fingerprint["code_hash"]) == 64
    assert fingerprint["source_dirs"] == list(DEFAULT_SOURCE_DIRS)
    assert isinstance(fingerprint["dirty_files"], list)


def test_git_state_reports_a_real_commit_in_this_repo():
    state = git_state(REPO_ROOT)

    assert state["commit"] != UNKNOWN
    assert len(state["commit"]) == 40
    assert state["dirty"] in (True, False)


def test_git_state_reports_clean_on_a_clean_checkout(tmp_path):
    """dirty=False on a tree with nothing uncommitted — the case the
    artifact's provenance claim depends on."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=tmp_path, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, capture_output=True)
    (tmp_path / "f.py").write_text("X = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True
    )

    state = git_state(tmp_path)

    assert state["dirty"] is False
    assert state["dirty_files"] == []
    assert state["commit"] != UNKNOWN


def test_git_state_reports_dirty_and_names_the_files(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "untracked.py").write_text("X = 1\n", encoding="utf-8")

    state = git_state(tmp_path)

    assert state["dirty"] is True
    assert any("untracked.py" in line for line in state["dirty_files"])


def test_git_state_degrades_to_unknown_outside_a_repository(tmp_path):
    """Never raises. An unattended run must not die because provenance
    could not be established."""
    state = git_state(tmp_path)

    assert state["commit"] == UNKNOWN
    assert state["dirty"] is None
    assert state["dirty_files"] == []


def test_git_state_degrades_to_unknown_when_git_is_missing(monkeypatch, tmp_path):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git not on PATH")

    monkeypatch.setattr(subprocess, "run", no_git)

    state = git_state(tmp_path)

    assert state["commit"] == UNKNOWN
    assert state["dirty"] is None


def test_dirty_is_none_not_false_when_git_cannot_answer(source_tree):
    """THE DISTINCTION THAT MATTERS. "no idea" must never render as
    "clean" — that would manufacture the reassurance this module exists
    to make earnable. An unknown tree state cannot claim `verified`."""
    monitor = IntegrityMonitor(
        launch={
            "commit": UNKNOWN,
            "dirty": None,
            "dirty_files": [],
            "code_hash": code_fingerprint(["pkg"], repo_root=source_tree),
        },
        on_intervention=lambda *a: None,
        dirs=["pkg"],
        repo_root=source_tree,
    )
    monitor.check()

    summary = monitor.summary()
    assert summary["tree_clean_at_launch"] is False
    assert summary["verified"] is False
    assert summary["code_fingerprint_stable"] is True, "the source itself never moved"
    assert "tree at launch=unknown" in monitor.one_line()


def test_one_line_does_not_describe_a_dirty_tree_as_clean(source_tree):
    """`dirty` is the git sense: True means uncommitted changes existed.
    Reading it as "clean" is the exact inversion that would let a dirty
    launch describe itself as a clean one."""
    monitor = IntegrityMonitor(
        launch={
            "commit": "c0ffee",
            "dirty": True,
            "dirty_files": [" M pkg/a.py"],
            "code_hash": code_fingerprint(["pkg"], repo_root=source_tree),
        },
        on_intervention=lambda *a: None,
        dirs=["pkg"],
        repo_root=source_tree,
    )

    assert "tree at launch=DIRTY" in monitor.one_line()


# ---------------------------------------------------------------------------
# classify_relaunch — the rule the count rests on
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, kind, run_id="r1", payload=None):
        self.kind = kind
        self.run_id = run_id
        self.payload = payload or {}


def _prior_run(last_kind, *, code_hash="abc", run_id="r1", with_fingerprint=True):
    payload = {}
    if with_fingerprint:
        payload = {"run_metadata": {"integrity": {"launch": {"code_hash": code_hash}}}}
    return [
        _FakeEvent("run_start", run_id, payload),
        _FakeEvent(last_kind, run_id),
    ]


def test_empty_journal_is_a_fresh_run():
    result = classify_relaunch([], code_hash="abc", resume_requested=False)

    assert result.kind == "fresh"
    assert result.counts_as_intervention is False


def test_a_prior_run_that_reached_run_end_is_a_fresh_run():
    result = classify_relaunch(
        _prior_run("run_end"), code_hash="abc", resume_requested=False
    )

    assert result.kind == "fresh"
    assert result.counts_as_intervention is False


def test_resume_over_identical_source_is_autonomous_and_not_counted():
    """THE CASE THAT MUST NOT INFLATE THE COUNT. A machine restarting
    itself over unchanged code is the autonomy story working."""
    result = classify_relaunch(
        _prior_run("eval_result", code_hash="abc"),
        code_hash="abc",
        resume_requested=True,
    )

    assert result.kind == "autonomous_resume"
    assert result.counts_as_intervention is False
    assert result.prior_code_hash == "abc"


def test_resume_over_changed_source_is_a_manual_restart():
    result = classify_relaunch(
        _prior_run("eval_result", code_hash="abc"),
        code_hash="def",
        resume_requested=True,
    )

    assert result.kind == "manual_restart"
    assert result.counts_as_intervention is True
    assert "not what was running then" in result.reason


def test_an_interrupted_run_without_resume_is_a_manual_restart():
    result = classify_relaunch(
        _prior_run("eval_result", code_hash="abc"),
        code_hash="abc",
        resume_requested=False,
    )

    assert result.kind == "manual_restart"
    assert result.counts_as_intervention is True


def test_an_interrupted_run_with_no_fingerprint_counts_as_manual():
    """An unverifiable resume is not a verified one; the tie breaks
    against us on purpose."""
    result = classify_relaunch(
        _prior_run("eval_result", with_fingerprint=False),
        code_hash="abc",
        resume_requested=True,
    )

    assert result.kind == "unknown_prior"
    assert result.counts_as_intervention is True


# ---------------------------------------------------------------------------
# IntegrityMonitor
# ---------------------------------------------------------------------------


def _monitor(tmp_path, dirs=("pkg",)):
    logged = []
    monitor = IntegrityMonitor(
        launch={
            "commit": "c0ffee",
            "dirty": False,
            "dirty_files": [],
            "code_hash": code_fingerprint(dirs, repo_root=tmp_path),
        },
        on_intervention=lambda who, kind, reason: logged.append((who, kind, reason)),
        dirs=dirs,
        repo_root=tmp_path,
    )
    return monitor, logged


def test_a_stable_run_records_no_interventions_and_counts_its_checks(source_tree):
    monitor, logged = _monitor(source_tree)

    for _ in range(3):
        assert monitor.check() is True

    summary = monitor.summary()
    assert logged == []
    assert summary["checks_performed"] == 3
    assert summary["code_fingerprint_stable"] is True
    assert summary["manual_interventions"] == 0
    # Stable and clean, but these were endpoint-style checks with no
    # candidate behind them, so the run has not earned the badge — see
    # test_endpoint_checks_alone_do_not_earn_the_badge below.
    assert summary["candidates_checked"] == 0
    assert summary["verified"] is False


def test_a_mid_run_edit_is_recorded_as_an_intervention(source_tree):
    monitor, logged = _monitor(source_tree)
    monitor.check()

    (source_tree / "pkg" / "a.py").write_text("A = 999\n", encoding="utf-8")
    assert monitor.check() is False

    assert len(logged) == 1
    who, kind, reason = logged[0]
    assert kind == "code_changed_midrun"
    assert "fingerprint changed during the run" in reason
    summary = monitor.summary()
    assert summary["manual_interventions"] == 1
    assert summary["code_fingerprint_stable"] is False
    assert summary["verified"] is False


def test_one_edit_is_reported_once_not_at_every_later_check(source_tree):
    """Inflating the count is as dishonest as suppressing it: one human
    action must be one intervention."""
    monitor, logged = _monitor(source_tree)
    monitor.check()
    (source_tree / "pkg" / "a.py").write_text("A = 999\n", encoding="utf-8")

    monitor.check()
    monitor.check()
    monitor.check()

    assert len(logged) == 1


def test_a_second_distinct_edit_is_still_caught(source_tree):
    monitor, logged = _monitor(source_tree)
    (source_tree / "pkg" / "a.py").write_text("A = 2\n", encoding="utf-8")
    monitor.check()
    (source_tree / "pkg" / "b.py").write_text("B = 3\n", encoding="utf-8")
    monitor.check()

    assert len(logged) == 2
    assert [entry[1] for entry in logged] == ["code_changed_midrun"] * 2


def test_a_dirty_launch_cannot_claim_verified(source_tree):
    logged = []
    monitor = IntegrityMonitor(
        launch={
            "commit": "c0ffee",
            "dirty": True,
            "dirty_files": [" M pkg/a.py"],
            "code_hash": code_fingerprint(["pkg"], repo_root=source_tree),
        },
        on_intervention=lambda *a: logged.append(a),
        dirs=["pkg"],
        repo_root=source_tree,
    )
    monitor.check()

    summary = monitor.summary()
    assert summary["verified"] is False
    assert summary["tree_clean_at_launch"] is False
    assert summary["dirty_files_at_launch"] == [" M pkg/a.py"]
    # A dirty launch is not itself an intervention — it is recorded, not counted.
    assert logged == []
    assert summary["manual_interventions"] == 0


def test_record_relaunch_only_logs_the_counted_kinds(source_tree):
    monitor, logged = _monitor(source_tree)

    monitor.record_relaunch(
        classify_relaunch(_prior_run("run_end"), code_hash="x", resume_requested=False)
    )
    assert logged == []

    monitor.record_relaunch(
        classify_relaunch(
            _prior_run("eval_result", code_hash="abc"), code_hash="abc", resume_requested=True
        )
    )
    assert logged == [], "an autonomous resume must not be counted"

    monitor.record_relaunch(
        classify_relaunch(
            _prior_run("eval_result", code_hash="abc"), code_hash="zzz", resume_requested=True
        )
    )
    assert len(logged) == 1
    assert logged[0][1] == "manual_restart"


def test_integrity_metadata_read_performs_a_check(source_tree):
    """The Controller reads run_metadata exactly twice — at RUN_START and
    at RUN_END — and each read verifies the source. That is what lets
    RUN_END carry a summary true AT RUN_END."""
    monitor, _ = _monitor(source_tree)
    metadata = IntegrityMetadata(monitor)

    first = dict(metadata)
    assert monitor.checks_performed == 1
    assert set(first["integrity"]) == {"policy", "launch", "summary"}

    second = dict(metadata)
    assert monitor.checks_performed == 2
    assert second["integrity"]["summary"]["checks_performed"] == 2


def test_checked_executor_verifies_before_delegating_and_forwards_attributes(source_tree):
    monitor, logged = _monitor(source_tree)
    checks_at_delegation = []

    class _Inner:
        calls = ["sentinel"]

        def run(self, config, seeds):
            # Captured INSIDE the delegate, so this asserts the check
            # happened BEFORE the candidate ran, not merely that it
            # happened at some point during the call.
            checks_at_delegation.append(monitor.checks_performed)
            return ("delegated", config.config_id, tuple(seeds))

    class _Config:
        config_id = "cfg1"

    wrapped = CheckedExecutor(_Inner(), monitor)
    result = wrapped.run(_Config(), (0, 1))

    assert result == ("delegated", "cfg1", (0, 1))
    assert checks_at_delegation == [1]
    assert monitor.checks_performed == 1
    assert logged == []
    # Attribute delegation keeps the wrapped adapter's own surface
    # reachable — the run summary reads `calls` through this wrapper.
    assert wrapped.calls == ["sentinel"]


def test_checked_executor_catches_an_edit_made_between_candidates(source_tree):
    """The reason for wrapping the executor at all: a run checked only at
    its two ends leaves the whole middle unobserved."""
    monitor, logged = _monitor(source_tree)

    class _Inner:
        def run(self, config, seeds):
            return "ok"

    class _Config:
        config_id = "cfg1"

    wrapped = CheckedExecutor(_Inner(), monitor)
    wrapped.run(_Config(), (0,))
    (source_tree / "pkg" / "a.py").write_text("A = 42\n", encoding="utf-8")
    wrapped.run(_Config(), (0,))

    assert [entry[1] for entry in logged] == ["code_changed_midrun"]
    assert "node boundary before cfg1" in logged[0][2]


# ---------------------------------------------------------------------------
# End to end through scripts/run_controller.py
# ---------------------------------------------------------------------------


def test_a_clean_run_logs_the_fingerprint_and_zero_interventions(
    tmp_path, monkeypatch, capsys
):
    """THE ARTIFACT'S CENTRAL CLAIM, end to end: a clean unattended run
    records positive integrity evidence and counts nothing."""
    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal.jsonl"

    exit_code = run_controller.main(
        ["--max-nodes-per-stage", "1", "--journal", str(journal_path), "--run-id", "clean"]
    )
    assert exit_code == 0

    # The launch fingerprint reached the durable journal.
    launch = _run_start(journal_path, "clean").payload["run_metadata"]["integrity"]["launch"]
    assert len(launch["code_hash"]) == 64
    assert launch["commit"]
    assert launch["source_dirs"] == list(DEFAULT_SOURCE_DIRS)

    # The positive record reached RUN_END.
    summary = _run_end(journal_path, "clean").payload["run_metadata"]["integrity"]["summary"]
    assert summary["code_fingerprint_stable"] is True
    assert summary["manual_interventions"] == 0
    # Launch read + one per candidate + the RUN_END read.
    assert summary["checks_performed"] >= 3

    # And nothing was counted.
    assert _interventions(journal_path, "clean") == []

    out = capsys.readouterr().out
    assert "integrity summary" in out
    assert "manual interventions    = 0" in out


def test_the_journal_and_the_monitor_agree_on_the_count(tmp_path, monkeypatch, capsys):
    """executor/report.py renders the journal's count; the console prints
    the monitor's. main() returns 2 if they ever diverge."""
    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal.jsonl"

    run_controller.main(
        ["--max-nodes-per-stage", "1", "--journal", str(journal_path), "--run-id", "agree"]
    )

    summary = _run_end(journal_path, "agree").payload["run_metadata"]["integrity"]["summary"]
    assert len(_interventions(journal_path, "agree")) == summary["manual_interventions"]
    assert "INTEGRITY ACCOUNTING MISMATCH" not in capsys.readouterr().out


def test_relaunching_an_interrupted_run_without_resume_logs_one_intervention(
    tmp_path, monkeypatch
):
    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal.jsonl"
    # A prior run that died before RUN_END.
    journal = Journal(str(journal_path), run_id="died")
    journal.log_run_start(
        run_metadata={"integrity": {"launch": {"code_hash": code_fingerprint()}}}
    )
    journal.log_eval_start("cfg1", node=1)

    run_controller.main(
        ["--max-nodes-per-stage", "1", "--journal", str(journal_path), "--run-id", "restarted"]
    )

    logged = _interventions(journal_path, "restarted")
    assert len(logged) == 1
    assert logged[0].payload["type"] == "manual_restart"
    assert logged[0].payload["who"] == "scripts/run_controller.py"
    relaunch = _run_start(journal_path, "restarted").payload["run_metadata"]["integrity"][
        "relaunch"
    ]
    assert relaunch["kind"] == "manual_restart"
    assert relaunch["counts_as_intervention"] is True


def test_relaunching_with_resume_over_identical_source_counts_nothing(
    tmp_path, monkeypatch
):
    """THE ONE THAT MUST NOT INFLATE THE NUMBER."""
    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal.jsonl"
    journal = Journal(str(journal_path), run_id="died")
    journal.log_run_start(
        run_metadata={"integrity": {"launch": {"code_hash": code_fingerprint()}}}
    )
    journal.log_eval_start("cfg1", node=1)

    exit_code = run_controller.main(
        [
            "--max-nodes-per-stage", "1",
            "--journal", str(journal_path),
            "--run-id", "resumed",
            "--resume",
        ]
    )
    assert exit_code == 0

    assert _interventions(journal_path, "resumed") == []
    relaunch = _run_start(journal_path, "resumed").payload["run_metadata"]["integrity"][
        "relaunch"
    ]
    assert relaunch["kind"] == "autonomous_resume"
    assert relaunch["counts_as_intervention"] is False
    # Recorded, though — not silently dropped.
    assert "no human touched the code" in relaunch["reason"]
    summary = _run_end(journal_path, "resumed").payload["run_metadata"]["integrity"][
        "summary"
    ]
    assert summary["manual_interventions"] == 0


def test_resume_after_a_prior_run_finished_cleanly_is_just_a_fresh_run(
    tmp_path, monkeypatch
):
    _stub_training(monkeypatch)
    journal_path = tmp_path / "journal.jsonl"

    run_controller.main(
        ["--max-nodes-per-stage", "1", "--journal", str(journal_path), "--run-id", "first"]
    )
    run_controller.main(
        ["--max-nodes-per-stage", "1", "--journal", str(journal_path), "--run-id", "second"]
    )

    assert _interventions(journal_path, "second") == []
    relaunch = _run_start(journal_path, "second").payload["run_metadata"]["integrity"][
        "relaunch"
    ]
    assert relaunch["kind"] == "fresh"


def test_run_metadata_is_absent_when_no_metadata_is_supplied(tmp_path, monkeypatch):
    """The Controller change is additive: a caller passing nothing gets the
    payload it got before this parameter existed."""
    from controller.controller import Controller
    from controller.fakes import AlwaysRejectGate, InMemoryJournal
    from controller.policy import FixedOrderPolicy

    from autonomy.adapters import MovesRealizer, RunCandidateExecutor, SlotScriptedGenerator

    _stub_training(monkeypatch)
    journal = InMemoryJournal()
    Controller(
        executor=RunCandidateExecutor(),
        gate=AlwaysRejectGate(),
        generator=SlotScriptedGenerator(),
        realizer=MovesRealizer(),
        policy=FixedOrderPolicy(("model",)),
        journal=journal,
        max_nodes_per_stage=1,
        run_id="no-metadata",
    ).run()

    start = next(e for e in journal.events if e.kind is EventKind.RUN_START)
    end = next(e for e in journal.events if e.kind is EventKind.RUN_END)
    assert set(start.payload) == {"seeds", "max_nodes_per_stage", "stage_order"}
    assert set(end.payload) == {"stop_reason", "iteration", "node"}


def test_run_metadata_cannot_shadow_a_field_the_controller_owns(tmp_path, monkeypatch):
    """Nested under one key on purpose: caller context must never rewrite
    the run's actual stop_reason."""
    from controller.controller import Controller
    from controller.fakes import AlwaysRejectGate, InMemoryJournal
    from controller.policy import FixedOrderPolicy

    from autonomy.adapters import MovesRealizer, RunCandidateExecutor, SlotScriptedGenerator

    _stub_training(monkeypatch)
    journal = InMemoryJournal()
    Controller(
        executor=RunCandidateExecutor(),
        gate=AlwaysRejectGate(),
        generator=SlotScriptedGenerator(),
        realizer=MovesRealizer(),
        policy=FixedOrderPolicy(("model",)),
        journal=journal,
        max_nodes_per_stage=1,
        run_id="shadow",
        run_metadata={"stop_reason": "LIES", "seeds": "LIES"},
    ).run()

    end = next(e for e in journal.events if e.kind is EventKind.RUN_END)
    assert end.payload["stop_reason"] != "LIES"
    assert end.payload["run_metadata"] == {"stop_reason": "LIES", "seeds": "LIES"}
