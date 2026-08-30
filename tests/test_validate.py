"""Tests for harness.validate — a thin wrapper over the vendor's
submit.py. None of the eight checks (see harness/SCHEMA_NOTES.md
submit.py Q17) are reimplemented here; these tests confirm the wrapper
delegates correctly and that each failure mode still surfaces the
vendor's own specific reason.
"""

import pytest

from harness import data, validate


def test_valid_submission_passes(tmp_path):
    rows = data.load("val")
    path = tmp_path / "valid_submission.csv"
    scores = [0.5] * len(rows)

    validate.write_submission(str(path), rows, scores)

    assert validate.validate_submission(str(path), split="val") is True


def test_swapped_header_fails_with_header_reason(tmp_path):
    path = tmp_path / "swapped_header.csv"
    path.write_text("video_id,user_id,row_id,score\n0,0,7531,0.5\n")

    with pytest.raises(ValueError, match="表头"):
        validate.validate_submission(str(path), split="val")


def test_row_id_gap_fails_with_row_id_reason(tmp_path):
    rows = data.load("val")
    path = tmp_path / "row_id_gap.csv"
    with open(path, "w", newline="") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        fh.write(f"0,{rows[0][1]},{rows[0][2]},0.1\n")
        # Skips row_id=1 and jumps straight to 2 — must be 0-based and
        # contiguous.
        fh.write(f"2,{rows[2][1]},{rows[2][2]},0.2\n")

    with pytest.raises(ValueError, match="row_id="):
        validate.validate_submission(str(path), split="val")


def test_nan_score_fails_with_nan_inf_reason(tmp_path):
    rows = data.load("val")
    path = tmp_path / "nan_score.csv"
    with open(path, "w", newline="") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        fh.write(f"0,{rows[0][1]},{rows[0][2]},nan\n")

    with pytest.raises(ValueError, match="NaN/Inf"):
        validate.validate_submission(str(path), split="val")


def test_truncated_row_count_fails_with_count_mismatch_reason(tmp_path):
    rows = data.load("val")
    path = tmp_path / "truncated.csv"
    truncated_rows = rows[:100]
    scores = [0.5] * len(truncated_rows)

    validate.write_submission(str(path), truncated_rows, scores)

    with pytest.raises(ValueError, match="数量不符"):
        validate.validate_submission(str(path), split="val")
