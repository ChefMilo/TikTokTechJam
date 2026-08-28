"""Data loading and split construction for the KuaiRand dataset.

This module WRAPS the vendor loader (vendor/kuairand-starter-kit/data.py)
rather than reimplementing CSV parsing or row construction — see
harness/SCHEMA_NOTES.md for the vendor loader's exact signature, return
shape, and column semantics (data.py Q7-Q11). All row-tuple parsing below
is the vendor's; the only thing this module adds is our own split
boundaries, the test-split lockout, and the row-count/date assertions.

Row shape returned by load(): unchanged from the vendor loader, a
7-tuple per row —
    (date: int, user_id: str, video_id: str, author_id: str,
     tab: str, duration_ms: float, label: int)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
_VENDOR_DATA_PY = REPO_ROOT / "vendor" / "kuairand-starter-kit" / "data.py"

TRAIN_END = 20220421
VAL_START = 20220422
VAL_END = 20220428
TEST_START = 20220429

# Published by the organizers (see harness/SCHEMA_NOTES.md). These are
# the split unit test: if load() doesn't hit them exactly, our date
# boundaries are wrong, not the constants.
EXPECTED_ROWS = {"train": 1_141_112, "val": 124_909, "test": 170_588}


def _load_vendor_data_module():
    """Imports vendor/kuairand-starter-kit/data.py by file path.

    Not a normal package import: the vendor directory name contains a
    hyphen (not a valid Python identifier) and its module is also named
    `data.py`, which would collide with this file if the vendor
    directory were added to sys.path. Loading by explicit file path
    avoids both problems and any risk of accidentally shadowing this
    module with the vendor one (or vice versa).
    """
    spec = importlib.util.spec_from_file_location(
        "_vendor_kuairand_data", _VENDOR_DATA_PY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vendor = _load_vendor_data_module()

# Memoized raw frame — keyed on nothing, since there is only one raw
# dataset. Deliberately holds ONLY train+val rows, never test: the
# vendor loader parses both log CSVs (and therefore necessarily reads
# test-window lines out of log_standard_4_22_to_5_08_pure.csv, since
# validation and test rows share that file) but its 'test' list is
# dropped immediately in _raw_rows() and never stored here. That way a
# future bug in load() can't leak test rows through this cache, because
# they were never retained in memory past the vendor call.
_RAW_ROWS: list[tuple] | None = None


def _raw_rows() -> list[tuple]:
    global _RAW_ROWS
    if _RAW_ROWS is None:
        vendor_splits = _vendor.load(str(DATA_DIR))
        _RAW_ROWS = vendor_splits["train"] + vendor_splits["valid"]
    return _RAW_ROWS


def load(split: str) -> list[tuple]:
    """Returns the rows for `split` as a list of the vendor's row tuples.

    split == "test" always raises PermissionError, before any data is
    even parsed — see the module docstring on why this is structural
    rather than a filter applied after the fact. Validation and hidden
    test rows live in the same source CSV, separated only by date; a
    slicing bug there is a silent leak that inflates validation scores
    and collapses on the real test set.
    """
    if split == "test":
        raise PermissionError(
            "test split is hidden during development; train+val only."
        )
    if split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}; expected 'train', 'val', or 'test'")

    rows = _raw_rows()
    if split == "train":
        result = [row for row in rows if row[0] <= TRAIN_END]
        upper_bound = TRAIN_END
    else:  # split == "val"
        result = [row for row in rows if VAL_START <= row[0] <= VAL_END]
        upper_bound = VAL_END

    max_date = max((row[0] for row in result), default=None)
    assert max_date is None or max_date <= upper_bound, (
        f"{split} split leaked rows past date {upper_bound}: max date found {max_date}"
    )
    assert len(result) == EXPECTED_ROWS[split], (
        f"{split} split row count mismatch: got {len(result)}, "
        f"expected {EXPECTED_ROWS[split]}"
    )
    return result


def load_side_features() -> tuple[Any, Any]:
    """Loads user-side and video-side feature tables directly from data/.

    The vendor loader does not expose these as first-class tables (see
    harness/SCHEMA_NOTES.md data.py Q11): vendor `load()` only extracts a
    video_id -> author_id lookup from video_features_basic_pure.csv, and
    never touches user_features_pure.csv at all — that file is read only
    by the standalone vendor/kuairand-starter-kit/ablation_features.py
    ablation script, not by the core loader. So this function reads the
    two feature CSVs directly rather than going through the vendor
    module.

    Source (both expected under data/):
        - data/user_features_pure.csv
        - data/video_features_basic_pure.csv

    Returns (user_features, video_features) as pandas DataFrames.
    """
    user_features = pd.read_csv(DATA_DIR / "user_features_pure.csv")
    video_features = pd.read_csv(DATA_DIR / "video_features_basic_pure.csv")
    return user_features, video_features
