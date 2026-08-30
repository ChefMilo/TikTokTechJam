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

import numpy as np

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
    values, and 'primary' is recomputed by Metrics itself. The check
    below is deliberately loud: our Metrics.primary is defined as an
    unweighted mean, which happens to match the vendor's
    (GAUC + nDCG@k) / 2 today — if either side's formula ever changes,
    we want that divergence to fail immediately, not silently skew every
    downstream ranking decision.
    """
    # Upcast labels here, once, for every caller: harness/cache.py stores
    # labels as int8 on disk (correct for that schema — each value is
    # 0/1), but vendor evaluate()'s internal per-user arithmetic
    # (npos * (npos + 1) inside auc()) is computed in whatever dtype
    # it's handed. A real validation user has npos=24 — 24*25=600
    # overflows int8's -128..127 range — which silently produced
    # GAUC=inf/NaN the first time a script read cached predictions back
    # and called evaluate() directly on them, undetected until printed.
    # harness/gate.py's bootstrap already upcasts before resampling for
    # the same reason; fixing it here too, centrally, means no future
    # caller has to remember to.
    labels = np.asarray(labels, dtype=np.int64)
    vendor_result = _vendor.evaluate(user_ids, labels, scores, k=k)
    # Cast to plain float here, once, for every caller: when `scores` is a
    # numpy array (as it always is coming out of a real model), vendor
    # evaluate()'s internal arithmetic on numpy scalars propagates
    # numpy.float32/float64 into its returned dict instead of Python
    # floats. Metrics is a shared contract type — it must not leak a
    # numpy-specific dtype into every downstream consumer (json.dumps,
    # in particular, rejects numpy floats outright; this surfaced as a
    # journal-write crash after 5+ minutes of real training).
    metrics = Metrics(
        values={
            "GAUC": float(vendor_result["GAUC"]),
            f"nDCG@{k}": float(vendor_result[f"nDCG@{k}"]),
        }
    )
    # Explicit raise, not assert — must survive python -O. This is a
    # correctness invariant, not a debugging aid; `assert` is stripped
    # entirely under -O and would let a silent divergence through.
    if abs(metrics.primary - vendor_result["primary"]) > 1e-9:
        raise AssertionError(
            f"Metrics.primary ({metrics.primary}) diverged from vendor's "
            f"primary ({vendor_result['primary']}) — unweighted-mean assumption broke"
        )
    return metrics
