"""Tests for autonomy.render and the two policy tightenings that ship
with it — the `verified` candidate floor and --require-clean.

FAST BY CONSTRUCTION: the end-to-end cases drive scripts/run_controller.py
against the stubbed executor PR1 introduced (patching
`autonomy.adapters._run_candidate`, which the adapter binds at import
time), so a real Controller journal is produced in milliseconds and the
renderer is exercised against genuine output rather than a fixture that
could drift from it.

WHAT THESE ARE FOR. The renderer turns evidence into a claim a reviewer
reads, so the assertions are about whether the claim is FAITHFUL: that a
short run cannot show a green badge, that a journal with no evidence says
so instead of rendering a reassuring blank, that two runs in one file are
never merged, and that a refused launch leaves nothing behind.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from contracts import CandidateResult, EventKind, Metrics, Status
from executor.journal import Journal

from autonomy.integrity import (
    MIN_CANDIDATES_FOR_VERIFIED,
    UNKNOWN,
    IntegrityMonitor,
    code_fingerprint,
)
from autonomy.render import ABSENT_NOTE, POLICY_PATH, render_autonomy, select_run

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_run_controller():
    spec = importlib.util.spec_from_file_location(
        "_scripts_run_controller_render", REPO_ROOT / "scripts" / "run_controller.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_controller = _load_run_controller()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_training(monkeypatch):
    def fake_run_candidate(fragment, target_slot, seeds=(0, 1, 2), journal=None):
        return CandidateResult(
            config_id=f"stub_{fragment.impl}",
            status=Status.OK,
            val={s: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60}) for s in seeds},
            backtest={s: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60}) for s in seeds},
            wall_seconds=0.0,
        )

    monkeypatch.setattr("autonomy.adapters._run_candidate", fake_run_candidate)


def _pretend_clean(monkeypatch, dirty=False, commit="c0ffee" * 6 + "abcd"):
    """Force the launch fingerprint's git half.

    The test suite runs from a working tree that is dirty by definition —
    it holds the change under review — so a test asserting on the
    clean-tree path has to say what git should report rather than hoping.
    """
    real = run_controller.launch_fingerprint

    def fake_launch_fingerprint(*args, **kwargs):
        fingerprint = real(*args, **kwargs)
        fingerprint["dirty"] = dirty
        fingerprint["dirty_files"] = [] if not dirty else [" M executor/run.py"]
        fingerprint["commit"] = commit
        return fingerprint

    monkeypatch.setattr(run_controller, "launch_fingerprint", fake_launch_fingerprint)


def _run(tmp_path, monkeypatch, *extra, run_id="r1", nodes="2"):
    journal_path = tmp_path / "journal.jsonl"
    code = run_controller.main(
        [
            "--max-nodes-per-stage", nodes,
            "--journal", str(journal_path),
            "--report-dir", str(tmp_path / "report"),
            "--run-id", run_id,
            *extra,
        ]
    )
    return code, journal_path


def _monitor(source_tree, *, min_candidates=MIN_CANDIDATES_FOR_VERIFIED, dirty=False):
    return IntegrityMonitor(
        launch={
            "commit": "c0ffee",
            "dirty": dirty,
            "dirty_files": [] if not dirty else [" M pkg/a.py"],
            "code_hash": code_fingerprint(["pkg"], repo_root=source_tree),
        },
        on_intervention=lambda *a: None,
        dirs=["pkg"],
        repo_root=source_tree,
        min_candidates=min_candidates,
    )


@pytest.fixture
def source_tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# The tightened `verified` rule
# ---------------------------------------------------------------------------


def test_endpoint_checks_alone_do_not_earn_the_badge(source_tree):
    """THE DEGENERATE CASE THE FLOOR EXISTS FOR. Check twice, evaluate
    nothing, claim "stable". True, and worth nothing."""
    monitor = _monitor(source_tree)
    monitor.check()
    monitor.check()

    summary = monitor.summary()
    assert summary["code_fingerprint_stable"] is True
    assert summary["manual_interventions"] == 0
    assert summary["candidates_checked"] == 0
    assert summary["verified"] is False
    assert any("below the floor of 3" in r for r in summary["unverified_because"])


def test_enough_checked_candidates_earns_the_badge(source_tree):
    monitor = _monitor(source_tree)
    for i in range(MIN_CANDIDATES_FOR_VERIFIED):
        monitor.check_at_node_boundary(f"cfg{i}")

    summary = monitor.summary()
    assert summary["candidates_checked"] == MIN_CANDIDATES_FOR_VERIFIED
    assert summary["verified"] is True
    assert summary["unverified_because"] == []


def test_one_candidate_short_of_the_floor_is_not_verified(source_tree):
    monitor = _monitor(source_tree)
    for i in range(MIN_CANDIDATES_FOR_VERIFIED - 1):
        monitor.check_at_node_boundary(f"cfg{i}")

    assert monitor.summary()["verified"] is False


def test_the_floor_is_configurable(source_tree):
    monitor = _monitor(source_tree, min_candidates=1)
    monitor.check_at_node_boundary("cfg0")

    summary = monitor.summary()
    assert summary["min_candidates_for_verified"] == 1
    assert summary["verified"] is True


def test_unverified_because_names_every_failed_condition(source_tree):
    """An unexplained false is barely more useful than no field."""
    monitor = _monitor(source_tree, dirty=True)
    monitor.check_at_node_boundary("cfg0")
    (source_tree / "pkg" / "a.py").write_text("A = 2\n", encoding="utf-8")
    monitor.check_at_node_boundary("cfg1")

    reasons = " ".join(monitor.summary()["unverified_because"])
    assert "dirty at launch" in reasons
    assert "fingerprint changed" in reasons
    assert "manual intervention" in reasons
    assert "below the floor" in reasons


def test_an_unknown_tree_state_is_reported_as_unknown_not_dirty(source_tree):
    monitor = IntegrityMonitor(
        launch={"commit": UNKNOWN, "dirty": None, "dirty_files": [], "code_hash": "x" * 64},
        on_intervention=lambda *a: None,
        dirs=["pkg"],
        repo_root=source_tree,
    )

    reasons = " ".join(monitor.summary()["unverified_because"])
    assert "unknown at launch" in reasons


def test_one_line_reports_candidate_coverage(source_tree):
    monitor = _monitor(source_tree)
    monitor.check_at_node_boundary("cfg0")

    assert "covering 1 candidate(s)" in monitor.one_line()


# ---------------------------------------------------------------------------
# select_run — never merge two runs
# ---------------------------------------------------------------------------


def test_select_run_defaults_to_the_most_recent_run(tmp_path):
    path = tmp_path / "j.jsonl"
    for run_id in ("old", "new"):
        journal = Journal(str(path), run_id=run_id)
        journal.log_run_start()
        journal.log_finalize(stop_reason="cap")

    run_id, events = select_run(Journal.replay(str(path)))

    assert run_id == "new"
    assert {e.run_id for e in events} == {"new"}


def test_select_run_honours_an_explicit_run_id(tmp_path):
    path = tmp_path / "j.jsonl"
    for run_id in ("old", "new"):
        journal = Journal(str(path), run_id=run_id)
        journal.log_run_start()

    run_id, events = select_run(Journal.replay(str(path)), "old")

    assert run_id == "old"
    assert {e.run_id for e in events} == {"old"}


def test_select_run_on_an_empty_journal(tmp_path):
    assert select_run([]) == (None, [])


# ---------------------------------------------------------------------------
# render_autonomy — the clean run
# ---------------------------------------------------------------------------


def test_a_clean_run_renders_a_verified_section(tmp_path, monkeypatch):
    """THE ARTIFACT'S HEADLINE DOCUMENT, end to end."""
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch)
    code, journal_path = _run(tmp_path, monkeypatch, run_id="clean")
    assert code == 0

    markdown = (tmp_path / "report" / "autonomy.md").read_text(encoding="utf-8")

    assert "# Autonomy" in markdown
    assert "**VERIFIED AUTONOMOUS — YES**" in markdown
    assert "Not verified because" not in markdown
    # Launch provenance.
    assert "## Launch provenance" in markdown
    assert "c0ffee" in markdown
    assert "**clean**" in markdown
    assert "Source fingerprint" in markdown
    assert "Started" in markdown
    # Integrity during the run.
    assert "## Integrity during the run" in markdown
    assert "Fingerprint checks" in markdown
    assert "Candidates checked" in markdown
    assert "**yes**" in markdown
    assert "reached RUN_END" in markdown
    assert "`fresh`" in markdown
    # Interventions.
    assert "## Interventions" in markdown
    assert "**0 — none recorded.**" in markdown
    # And the pointer to the definition.
    assert POLICY_PATH in markdown


def test_render_autonomy_returns_the_markdown_it_writes(tmp_path, monkeypatch):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch)
    _, journal_path = _run(tmp_path, monkeypatch, run_id="clean")

    returned = render_autonomy(str(journal_path), str(tmp_path / "out"), run_id="clean")

    assert returned == (tmp_path / "out" / "autonomy.md").read_text(encoding="utf-8")
    assert returned.startswith("# Autonomy")


def test_render_autonomy_without_out_dir_writes_nothing(tmp_path, monkeypatch):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch)
    _, journal_path = _run(tmp_path, monkeypatch, run_id="clean")

    markdown = render_autonomy(str(journal_path))

    assert "# Autonomy" in markdown
    assert not (tmp_path / "autonomy.md").exists()


def test_the_autonomy_section_is_written_even_without_the_report_flag(
    tmp_path, monkeypatch, capsys
):
    """It reads the journal directly, so it must not be hostage to
    executor/report.py's unrelated rendering bug."""
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch)
    _run(tmp_path, monkeypatch, run_id="clean")

    assert (tmp_path / "report" / "autonomy.md").exists()
    out = capsys.readouterr().out
    assert "autonomy section" in out
    # The metric report stays gated.
    assert "not rendered" in out
    assert not (tmp_path / "report" / "results.md").exists()


# ---------------------------------------------------------------------------
# render_autonomy — the unhappy paths
# ---------------------------------------------------------------------------


def test_a_short_run_renders_an_unverified_section_naming_the_reason(
    tmp_path, monkeypatch
):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch)
    code, _ = _run(
        tmp_path, monkeypatch, "--min-verified-candidates", "99", run_id="short"
    )
    assert code == 0

    markdown = (tmp_path / "report" / "autonomy.md").read_text(encoding="utf-8")

    assert "**VERIFIED AUTONOMOUS — NO**" in markdown
    assert "Not verified because:" in markdown
    assert "below the floor of 99" in markdown


def test_a_dirty_launch_renders_the_uncommitted_files(tmp_path, monkeypatch):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch, dirty=True)
    _run(tmp_path, monkeypatch, run_id="dirty")

    markdown = (tmp_path / "report" / "autonomy.md").read_text(encoding="utf-8")

    assert "**VERIFIED AUTONOMOUS — NO**" in markdown
    assert "DIRTY" in markdown
    assert "Uncommitted at launch:" in markdown
    assert "executor/run.py" in markdown


def test_a_manual_restart_renders_an_intervention_row(tmp_path, monkeypatch):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch)
    journal_path = tmp_path / "journal.jsonl"
    # A prior run that died before RUN_END.
    prior = Journal(str(journal_path), run_id="died")
    prior.log_run_start(
        run_metadata={"integrity": {"launch": {"code_hash": code_fingerprint()}}}
    )
    prior.log_eval_start("cfg1", node=1)

    _run(tmp_path, monkeypatch, run_id="restarted")

    markdown = (tmp_path / "report" / "autonomy.md").read_text(encoding="utf-8")
    assert "**1 recorded.**" in markdown
    assert "`manual_restart`" in markdown
    assert "`scripts/run_controller.py`" in markdown
    assert "| iteration | who | type | reason |" in markdown
    assert "**VERIFIED AUTONOMOUS — NO**" in markdown
    # The other run in the same file is named but not merged in.
    assert "`died`" in markdown
    assert "not shown here" in markdown


def test_a_journal_with_no_integrity_evidence_says_so_and_does_not_raise(tmp_path):
    """A run_agent.py-style journal: real events, no run_metadata. This is
    the case that must never crash a report generator."""
    path = tmp_path / "legacy.jsonl"
    journal = Journal(str(path), run_id="run_agent")
    journal.log_run_start(moves=[1, 2, 3], seeds=[0, 1, 2])
    journal.log_eval_start("bce19171850a", node=1)
    journal.log_eval_result(
        "bce19171850a", {0: Metrics(values={"GAUC": 0.6, "nDCG@5": 0.6})}, 40.0, node=1
    )
    journal.log_finalize(stop_reason="scripted_moves_exhausted")

    markdown = render_autonomy(str(path), str(tmp_path / "out"))

    assert ABSENT_NOTE in markdown
    assert "run_agent.py" in markdown
    assert "not the same as a run with clean evidence" in markdown
    assert "VERIFIED AUTONOMOUS" not in markdown
    assert (tmp_path / "out" / "autonomy.md").exists()


def test_an_empty_journal_renders_without_raising(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    markdown = render_autonomy(str(path))

    assert ABSENT_NOTE in markdown


def test_a_missing_journal_renders_without_raising(tmp_path):
    markdown = render_autonomy(str(tmp_path / "does_not_exist.jsonl"))

    assert ABSENT_NOTE in markdown


def test_an_interrupted_run_is_rendered_as_unfinished(tmp_path):
    """No RUN_END means the process died. The section must say so rather
    than quietly rendering a partial run as a complete one."""
    path = tmp_path / "j.jsonl"
    journal = Journal(str(path), run_id="died")
    journal.log_run_start(
        run_metadata={
            "integrity": {
                "launch": {"commit": "abc", "dirty": False, "code_hash": "f" * 64},
                "summary": {"verified": False},
            }
        }
    )
    journal.log_eval_start("cfg1", node=1)

    markdown = render_autonomy(str(path))

    assert "no RUN_END" in markdown
    assert "did not finish through its normal path" in markdown


# ---------------------------------------------------------------------------
# --require-clean
# ---------------------------------------------------------------------------


def test_require_clean_refuses_on_a_dirty_tree(tmp_path, monkeypatch, capsys):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch, dirty=True)

    code, journal_path = _run(tmp_path, monkeypatch, "--require-clean", run_id="refused")

    assert code == 3
    out = capsys.readouterr().out
    assert "REFUSING TO START (--require-clean)" in out
    assert "executor/run.py" in out
    # NOTHING was appended — a half-started run would be misread as an
    # interruption by the next launch, manufacturing an intervention.
    assert not journal_path.exists()


def test_require_clean_refuses_when_git_cannot_answer(tmp_path, monkeypatch, capsys):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch, dirty=None, commit=UNKNOWN)

    code, journal_path = _run(tmp_path, monkeypatch, "--require-clean", run_id="unknown")

    assert code == 3
    out = capsys.readouterr().out
    assert "could not determine the working tree state" in out
    assert not journal_path.exists()


def test_require_clean_proceeds_on_a_clean_tree(tmp_path, monkeypatch):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch, dirty=False)

    code, journal_path = _run(tmp_path, monkeypatch, "--require-clean", run_id="clean")

    assert code == 0
    assert journal_path.exists()
    markdown = (tmp_path / "report" / "autonomy.md").read_text(encoding="utf-8")
    assert "**VERIFIED AUTONOMOUS — YES**" in markdown


def test_require_clean_is_off_by_default(tmp_path, monkeypatch):
    """Development runs, and this test suite, run from a dirty tree."""
    assert run_controller._parse_args([]).require_clean is False
    assert run_controller._parse_args([]).min_verified_candidates == MIN_CANDIDATES_FOR_VERIFIED

    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch, dirty=True)
    code, _ = _run(tmp_path, monkeypatch, run_id="dirty-ok")

    assert code == 0


# ---------------------------------------------------------------------------
# Cross-check: the rendered count and the journal cannot disagree
# ---------------------------------------------------------------------------


def test_the_rendered_intervention_count_matches_the_journals_events(
    tmp_path, monkeypatch
):
    _stub_training(monkeypatch)
    _pretend_clean(monkeypatch)
    journal_path = tmp_path / "journal.jsonl"
    prior = Journal(str(journal_path), run_id="died")
    prior.log_run_start(
        run_metadata={"integrity": {"launch": {"code_hash": code_fingerprint()}}}
    )
    prior.log_eval_start("cfg1", node=1)

    _run(tmp_path, monkeypatch, run_id="restarted")

    markdown = (tmp_path / "report" / "autonomy.md").read_text(encoding="utf-8")
    logged = [
        e
        for e in Journal.replay(str(journal_path))
        if e.kind is EventKind.INTERVENTION and e.run_id == "restarted"
    ]

    assert len(logged) == 1
    assert f"**{len(logged)} recorded.**" in markdown
    assert "Accounting mismatch" not in markdown
