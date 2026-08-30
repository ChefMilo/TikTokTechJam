"""Tests for executor.run.run_candidate's optional journal wiring.

Monkeypatches executor.realize.realize (and the harness calls around it)
so this exercises the wiring — EVAL_START before, EVAL_RESULT after, and
ERROR on failure — without a real multi-minute FM training run. Real
end-to-end behaviour (actual training, actual gate) is covered by
scripts/i1_smoke.py.
"""

import numpy as np
import pytest

from contracts import SlotConfig, Status
from executor import run as run_module
from executor.journal import Journal


def _fake_train_val_data():
    return [(20220409, "u1", "v1", "a1", "t1", 10.0, 1)], [(20220422, "u1", "v1", "a1", "t1", 10.0, 1)]


@pytest.fixture(autouse=True)
def _fake_harness(monkeypatch):
    train_rows, val_rows = _fake_train_val_data()
    monkeypatch.setattr(run_module.data, "load", lambda split: train_rows if split == "train" else val_rows)
    monkeypatch.setattr(run_module.backtest, "split", lambda: (train_rows, val_rows))

    def _fake_realize(config, fit_rows, score_rows, seed):
        n = len(score_rows)
        user_ids = np.array([row[1] for row in score_rows])
        labels = np.array([row[6] for row in score_rows])
        scores = np.full(n, 0.5 + seed * 0.01)
        return user_ids, labels, scores

    monkeypatch.setattr(run_module.realize_module, "realize", _fake_realize)
    monkeypatch.setattr(run_module.cache, "save_predictions", lambda *a, **k: None)


def test_run_candidate_without_journal_still_works():
    fragment = SlotConfig(impl="fm", params={"k": 16, "lr": 0.001})
    result = run_module.run_candidate(fragment, "model", seeds=(0, 1))

    assert result.status is Status.OK
    assert set(result.val) == {0, 1}


def test_run_candidate_with_journal_emits_eval_start_and_result(tmp_path):
    journal = Journal(str(tmp_path / "journal.jsonl"), run_id="run-1")
    fragment = SlotConfig(impl="fm", params={"k": 16, "lr": 0.001})

    result = run_module.run_candidate(fragment, "model", seeds=(0, 1), journal=journal)

    events = Journal.replay(str(tmp_path / "journal.jsonl"))
    kinds = [e.kind.value for e in events]
    assert kinds == ["eval_start", "eval_result"]
    assert events[0].node == events[1].node == 1
    assert events[1].payload["config_id"] == result.config_id
    assert events[1].payload["target_slot"] == "model"
    assert events[1].payload["fragment_impl"] == "fm"


def test_run_candidate_with_journal_emits_error_on_failure(tmp_path, monkeypatch):
    journal = Journal(str(tmp_path / "journal.jsonl"), run_id="run-1")
    fragment = SlotConfig(impl="fm", params={"k": 16, "lr": 0.001})

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(run_module.realize_module, "realize", _boom)

    result = run_module.run_candidate(fragment, "model", seeds=(0,), journal=journal)

    assert result.status is Status.FAILED
    assert "synthetic training failure" in result.error_excerpt

    events = Journal.replay(str(tmp_path / "journal.jsonl"))
    assert [e.kind.value for e in events] == ["eval_start", "error"]
    assert "synthetic training failure" in events[1].payload["excerpt"]
