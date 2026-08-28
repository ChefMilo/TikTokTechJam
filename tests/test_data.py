"""Tests for harness.data — the vendor-loader wrapper and its test-split
lockout. The row-count assertions inside harness.data.load() are
themselves the split unit test; these tests exercise the public
behaviour around them (no overlap, no leakage, correct error types).
"""

import pytest

from harness import data


def test_train_and_val_return_expected_row_counts():
    train_rows = data.load("train")
    val_rows = data.load("val")

    assert len(train_rows) == data.EXPECTED_ROWS["train"]
    assert len(val_rows) == data.EXPECTED_ROWS["val"]


def test_load_test_split_raises_permission_error():
    with pytest.raises(PermissionError):
        data.load("test")


def test_train_max_date_precedes_val_min_date():
    train_rows = data.load("train")
    val_rows = data.load("val")

    max_train_date = max(row[0] for row in train_rows)
    min_val_date = min(row[0] for row in val_rows)

    assert max_train_date < min_val_date


def test_no_train_or_val_row_reaches_test_start():
    train_rows = data.load("train")
    val_rows = data.load("val")

    assert all(row[0] < data.TEST_START for row in train_rows)
    assert all(row[0] < data.TEST_START for row in val_rows)


def test_unknown_split_raises_value_error():
    with pytest.raises(ValueError):
        data.load("bogus")
