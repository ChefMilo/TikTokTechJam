"""Tests for executor.journal.Journal — append/replay durability and the
log_* helpers' payload shapes.
"""

from contracts import Citation, EventKind, JournalEvent, Metrics, Verdict
from executor.journal import Journal


def test_append_then_replay_round_trips_events_in_order(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(str(path), run_id="run-1")

    journal.log_run_start(seeds=[0, 1, 2])
    journal.log_hypothesis(
        "model",
        "try bigger k",
        Citation(key="rendle2010fm", url="https://example.com", library_entry="methods/library/fm.yaml#fm"),
        0.006,
        45.0,
        node=1,
    )
    journal.log_eval_start("cfg1", node=1)
    journal.log_eval_result("cfg1", {0: Metrics(values={"GAUC": 0.6, "nDCG@5": 0.6})}, 40.0, node=1)

    events = Journal.replay(str(path))

    assert [e.kind for e in events] == [
        EventKind.RUN_START,
        EventKind.HYPOTHESIS,
        EventKind.EVAL_START,
        EventKind.EVAL_RESULT,
    ]
    assert all(e.run_id == "run-1" for e in events)
    # Order is preserved exactly as appended.
    assert events[1].payload["target_slot"] == "model"
    assert events[3].payload["config_id"] == "cfg1"


def test_truncated_file_still_replays_every_complete_event(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(str(path), run_id="run-1")
    journal.log_run_start()
    journal.log_stage_change(from_stage="warmup", to_stage="search")
    journal.log_finalize(stop_reason="cap")

    intact_bytes = path.read_bytes()
    # Simulate a crash mid-write: append a torn, incomplete JSON line
    # with no trailing newline, as a real crash would leave one.
    with open(path, "ab") as fh:
        fh.write(b'{"ts": "2026-01-01T00:00:00+00:00", "run_id": "run-1", "ite')

    events = Journal.replay(str(path))

    assert len(events) == 3
    assert [e.kind for e in events] == [EventKind.RUN_START, EventKind.STAGE_CHANGE, EventKind.FINALIZE]
    # And the intact prefix is untouched by the torn trailing bytes.
    assert path.read_bytes().startswith(intact_bytes)


def test_journal_resumes_position_from_an_existing_file(tmp_path):
    path = tmp_path / "journal.jsonl"
    first = Journal(str(path), run_id="run-1")
    first.log_run_start()
    first.log_eval_start("cfg1", node=1)
    first.log_eval_result("cfg1", {0: Metrics(values={"GAUC": 0.6, "nDCG@5": 0.6})}, 40.0, node=1)
    verdict = Verdict(accept=True, delta=0.003, ci95=(0.001, 0.005), n_seeds=3, backtest_delta=0.002, reason="ok")
    first.log_decision(verdict, node=1)

    # A fresh Journal instance pointed at the same path (the crash-resume
    # scenario) must pick up exactly where the last one left off.
    resumed = Journal(str(path), run_id="run-1")

    assert resumed.current_iteration == 1
    assert resumed.current_node == 1


def test_log_decision_records_n_seeds(tmp_path):
    """CONTROLLER_AUDIT.md found verdict.n_seeds computed by the gate and
    then dropped everywhere downstream — this is the fix."""
    path = tmp_path / "journal.jsonl"
    journal = Journal(str(path), run_id="run-1")
    verdict = Verdict(accept=True, delta=0.004, ci95=(0.001, 0.007), n_seeds=5, backtest_delta=0.003, reason="ok")

    event = journal.log_decision(verdict)

    assert event.payload["n_seeds"] == 5

    replayed = Journal.replay(str(path))
    assert replayed[-1].payload["n_seeds"] == 5


def test_log_decision_advances_iteration_only_on_accept(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(str(path), run_id="run-1")

    rejected = Verdict(accept=False, delta=-0.001, ci95=(-0.003, 0.001), n_seeds=3, backtest_delta=None, reason="ci_includes_zero")
    journal.log_decision(rejected)
    assert journal.current_iteration == 0

    accepted = Verdict(accept=True, delta=0.004, ci95=(0.001, 0.007), n_seeds=3, backtest_delta=0.003, reason="ok")
    journal.log_decision(accepted)
    assert journal.current_iteration == 1
