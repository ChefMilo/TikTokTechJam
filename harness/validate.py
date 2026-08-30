"""Validation of the SUBMISSION FILE, not of splits — split integrity
(train/val/test leakage) is data.py's responsibility, checked at split
construction time.

Thin wrapper around the vendor's submit.py (read_submission /
write_submission), imported by file path like the other vendor wrappers
in this package. Per harness/SCHEMA_NOTES.md submit.py Q17,
read_submission already performs eight checks — header, field count,
row_id contiguity, no overflow, user_id/video_id alignment, score
parses, score is finite, and exact row count. None of them are
reimplemented here; on failure the vendor's own error (naming the
specific check that failed) propagates unchanged.

Row order/alignment for a split comes from the VENDOR's own loader here,
not harness.data: unlike training or model selection, checking a
submission's row_id/user_id/video_id alignment against the real hidden
test split is the sanctioned use of test-split structure that the
organizers' own `submit.py --check` performs — it only ever looks at
row structure, never at test labels.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from harness import data

REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_DIR = REPO_ROOT / "vendor" / "kuairand-starter-kit"

# harness.data intentionally never exposes "test", so this module's own
# naming is spelled out here rather than reusing a shared alias table.
_SPLIT_TO_VENDOR_KEY = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def _load_vendor_submit():
    """Imports vendor/kuairand-starter-kit/submit.py by file path — same
    approach as harness/data.py, harness/metrics.py, and
    tests/test_rungs.py's baseline import. submit.py's own top-level
    `from data import ...` / `from evaluate import ...` rely on Python's
    normal sys.path search, so the vendor directory is put on sys.path
    just long enough to exec it.
    """
    vendor_dir_str = str(_VENDOR_DIR)
    sys.path.insert(0, vendor_dir_str)
    try:
        spec = importlib.util.spec_from_file_location(
            "_vendor_kuairand_submit", _VENDOR_DIR / "submit.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(vendor_dir_str)
    return module


_vendor = _load_vendor_submit()

# Cache of the vendor's own full train/valid/test split dict — separate
# from harness.data's cache, which deliberately never retains test rows.
# This module is the one sanctioned place allowed to hold test row
# structure (never labels) for the reason in the module docstring above.
_vendor_splits_cache: dict | None = None


def _vendor_splits():
    global _vendor_splits_cache
    if _vendor_splits_cache is None:
        _vendor_splits_cache = _vendor.load(str(data.DATA_DIR))
    return _vendor_splits_cache


def validate_submission(path: str, split: str = "test") -> bool:
    """Returns True if the submission at `path` is well-formed for
    `split`. Raises ValueError (the vendor's own message, naming the
    specific check that failed) otherwise.
    """
    if split not in _SPLIT_TO_VENDOR_KEY:
        raise ValueError(f"unknown split {split!r}; expected 'val' or 'test'")
    rows = _vendor_splits()[_SPLIT_TO_VENDOR_KEY[split]]
    _vendor.read_submission(str(path), rows)
    return True


def write_submission(path: str, rows, scores) -> None:
    """Wraps the vendor's write_submission so the whole team produces
    byte-identical formatting (header row_id,user_id,video_id,score;
    score written as f"{float(s):.6g}" — see harness/SCHEMA_NOTES.md
    submit.py Q19). Not reimplemented here.
    """
    _vendor.write_submission(str(path), rows, scores)
