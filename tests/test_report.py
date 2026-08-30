"""Tests for executor.report.render — exercised against a synthetic
journal so it doesn't need a real controller loop or a real training run.
"""

import csv

from contracts import Citation, Metrics, Verdict
from executor.journal import Journal
from executor.report import render


def _build_synthetic_journal(path: str) -> None:
    """A synthetic 5-event journal: RUN_START, HYPOTHESIS, EVAL_RESULT,
    DECISION, FINALIZE — one full node's story plus the run bookends.
    """
    journal = Journal(path, run_id="run-1")
    journal.log_run_start(seeds=[0, 1, 2])
    journal.log_hypothesis(
        "model",
        "bump FM capacity from k=16 to k=32",
        Citation(key="rendle2010fm", url="https://example.com", library_entry="methods/library/fm.yaml#fm"),
        expected_gain=0.006,
        expected_cost_s=45.0,
        node=1,
    )
    journal.log_eval_result(
        "cfg1",
        {
            0: Metrics(values={"GAUC": 0.6671, "nDCG@5": 0.5358}),
            1: Metrics(values={"GAUC": 0.6674, "nDCG@5": 0.5361}),
        },
        wall_seconds=90.0,
        target_slot="model",
        fragment_impl="fm",
        fragment_params={"k": 32, "lr": 0.001},
        node=1,
    )
    verdict = Verdict(
        accept=True, delta=0.0034, ci95=(0.0012, 0.0056), n_seeds=2, backtest_delta=0.002,
        reason="paired CI excludes zero",
    )
    journal.log_decision(verdict, node=1)
    journal.log_finalize(stop_reason="cap")


def test_render_produces_all_four_files(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    _build_synthetic_journal(str(journal_path))
    out_dir = tmp_path / "report"

    render(str(journal_path), output_dir=str(out_dir))

    for name in ("iterations.md", "results.md", "trajectory.csv", "forecast_calibration.md"):
        assert (out_dir / name).exists(), f"{name} was not written"


def test_iterations_md_contains_hypothesis_result_and_verdict(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    _build_synthetic_journal(str(journal_path))
    out_dir = tmp_path / "report"
    render(str(journal_path), output_dir=str(out_dir))

    content = (out_dir / "iterations.md").read_text(encoding="utf-8")

    assert "bump FM capacity from k=16 to k=32" in content
    assert "rendle2010fm" in content
    assert "cfg1" in content
    assert "ACCEPTED" in content
    assert "n_seeds=2" in content


def test_results_md_reports_best_metrics_and_deltas(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    _build_synthetic_journal(str(journal_path))
    out_dir = tmp_path / "report"
    render(str(journal_path), output_dir=str(out_dir))

    content = (out_dir / "results.md").read_text(encoding="utf-8")

    assert "Validation-best GAUC" in content
    assert "Total tokens | 0" in content
    assert "Iterations used | 1 / 50" in content


def test_trajectory_csv_has_one_row_with_expected_columns(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    _build_synthetic_journal(str(journal_path))
    out_dir = tmp_path / "report"
    render(str(journal_path), output_dir=str(out_dir))

    with open(out_dir / "trajectory.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["config_id"] == "cfg1"
    assert row["accepted"] == "True"
    assert float(row["delta"]) == 0.0034


def test_forecast_calibration_pairs_hypothesis_to_decision_by_node(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    _build_synthetic_journal(str(journal_path))
    out_dir = tmp_path / "report"
    render(str(journal_path), output_dir=str(out_dir))

    content = (out_dir / "forecast_calibration.md").read_text(encoding="utf-8")

    assert "+0.0060" in content  # expected_gain
    assert "+0.0034" in content  # realized_delta
    assert "0.0026" in content  # abs_error = |0.006 - 0.0034|
    assert "Mean absolute error: 0.0026" in content


def test_forecast_calibration_skips_nodes_missing_either_side(tmp_path):
    """A HYPOTHESIS with no matching DECISION (still running, or the
    candidate failed before a verdict) must not appear in the table."""
    journal_path = tmp_path / "journal.jsonl"
    journal = Journal(str(journal_path), run_id="run-1")
    journal.log_hypothesis(
        "model", "untested idea",
        Citation(key="k", url="u", library_entry="l"),
        expected_gain=0.01, expected_cost_s=30.0, node=1,
    )
    out_dir = tmp_path / "report"

    render(str(journal_path), output_dir=str(out_dir))

    content = (out_dir / "forecast_calibration.md").read_text(encoding="utf-8")
    assert "no node has both" in content
