"""Evaluation metrics for recommender-system methods on KuaiRand.

Thin wrapper around the vendor scorer (vendor/kuairand-starter-kit/evaluate.py)
— see harness/SCHEMA_NOTES.md evaluate.py Q1-Q6 for the exact signature
and conventions (GAUC + nDCG@5, zero-positive-user handling, per-user
GAUC weighting). The metric math itself is NOT reimplemented here; this
module only reshapes the vendor's dict into contracts.Metrics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from contracts import Metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_EVALUATE_PY = REPO_ROOT / "vendor" / "kuairand-starter-kit" / "evaluate.py"


def _load_vendor_evaluate_module():
    """Imports vendor/kuairand-starter-kit/evaluate.py by file path — see
    harness/data.py's _load_vendor_data_module for why (hyphenated
    vendor dir name, module filename that would otherwise collide).
    """
    spec = importlib.util.spec_from_file_location(
        "_vendor_kuairand_evaluate", _VENDOR_EVALUATE_PY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vendor = _load_vendor_evaluate_module()


def evaluate(user_ids, labels, scores, k: int = 5) -> Metrics:
    """Scores one (user_ids, labels, scores) triple via the vendor's
    evaluate() and returns the result as a contracts.Metrics.

    The vendor dict also carries 'primary', 'users', and 'rows'; 'users'
    and 'rows' are dropped here since Metrics only carries named metric
    values, and 'primary' is recomputed by Metrics itself. The assertion
    below is deliberately loud: our Metrics.primary is defined as an
    unweighted mean, which happens to match the vendor's
    (GAUC + nDCG@k) / 2 today — if either side's formula ever changes,
    we want that divergence to fail a test immediately, not silently
    skew every downstream ranking decision.
    """
    vendor_result = _vendor.evaluate(user_ids, labels, scores, k=k)
    metrics = Metrics(
        values={
            "GAUC": vendor_result["GAUC"],
            f"nDCG@{k}": vendor_result[f"nDCG@{k}"],
        }
    )
    assert abs(metrics.primary - vendor_result["primary"]) < 1e-9, (
        f"Metrics.primary ({metrics.primary}) diverged from vendor's "
        f"primary ({vendor_result['primary']}) — unweighted-mean assumption broke"
    )
    return metrics
