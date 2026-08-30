"""Tests for harness.backtest — the rolling-origin split carved entirely
out of the TRAINING window.
"""

from harness import backtest, data


def test_fit_and_score_windows_do_not_overlap():
    fit_rows, score_rows = backtest.split()

    fit_dates = {row[0] for row in fit_rows}
    score_dates = {row[0] for row in score_rows}
    assert fit_dates.isdisjoint(score_dates)


def test_fit_and_score_union_equals_full_train_split():
    fit_rows, score_rows = backtest.split()
    train_rows = data.load("train")

    assert len(fit_rows) + len(score_rows) == len(train_rows)
    assert sorted(fit_rows + score_rows) == sorted(train_rows)


def test_no_row_exceeds_train_end():
    fit_rows, score_rows = backtest.split()

    assert all(row[0] <= 20220421 for row in fit_rows)
    assert all(row[0] <= 20220421 for row in score_rows)


def test_split_only_ever_requests_the_train_split(monkeypatch):
    real_load = data.load

    def guarded_load(split):
        assert split == "train", f"harness.backtest must only load 'train', got {split!r}"
        return real_load(split)

    monkeypatch.setattr(backtest.data, "load", guarded_load)
    backtest.split()
