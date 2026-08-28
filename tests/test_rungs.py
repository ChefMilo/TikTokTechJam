"""Tests that our harness reproduces the organizers' reference "rungs":
the three baseline scores published in the Starter Kit (random scoring,
item popularity, and the official Factorization Machine baseline).

These are not about harness.backtest's rolling-origin split. If these
fail, our split boundaries or metric conventions are wrong, and every
downstream number the agent produces is meaningless regardless of how
good a method looks.

Everything here runs on VALIDATION only — the hidden test split is never
touched (harness.data.load("test") is structurally locked out; the
vendor's own baseline functions additionally require a 'test' key to run
at all — see the vendor_splits fixture below — but nothing here reads or
asserts against its output).
"""

import collections
import importlib.util
import sys
from pathlib import Path

import pytest

from harness import data, metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_DIR = REPO_ROOT / "vendor" / "kuairand-starter-kit"

# Published for the VALIDATION split (harness/SCHEMA_NOTES.md baseline.py
# Q14 / README "Baseline 阶梯"). Random and item-popularity only publish
# hidden-TEST primary (0.4753 / 0.5715 respectively) — those are never
# asserted here, only the FM row has a published validation figure.
FM_VALID_PUBLISHED = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
FM_PRIMARY_TOLERANCE = 0.002


def _load_vendor_baseline():
    """Imports vendor/kuairand-starter-kit/baseline.py by file path, same
    approach as harness/data.py and harness/metrics.py use for the vendor
    loader/evaluator. Unlike those two, baseline.py's own top-level `from
    data import ...` / `from evaluate import ...` are plain imports that
    rely on Python's normal sys.path search (the script is designed to be
    run from inside its own directory) — so the vendor directory is put
    on sys.path just long enough to exec this one module.
    """
    vendor_dir_str = str(_VENDOR_DIR)
    sys.path.insert(0, vendor_dir_str)
    try:
        spec = importlib.util.spec_from_file_location(
            "_vendor_kuairand_baseline", _VENDOR_DIR / "baseline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(vendor_dir_str)
    return module


@pytest.fixture(scope="module")
def vendor_baseline():
    return _load_vendor_baseline()


@pytest.fixture(scope="module")
def vendor_splits(vendor_baseline):
    """Full train/valid/test split dict, in the vendor's own shape.

    Required because baseline.run_pop/run_random hardcode
    `for name in ('valid', 'test')` internally (vendor/kuairand-starter-kit/baseline.py:22,32)
    — they cannot be called with a test-less splits dict. Only ['valid']
    is ever read off their results below.
    """
    return vendor_baseline.load(str(data.DATA_DIR))


@pytest.fixture(scope="module")
def random_valid_result(vendor_baseline, vendor_splits):
    return vendor_baseline.run_random(vendor_splits, seed=0)["valid"]


@pytest.fixture(scope="module")
def pop_valid_result(vendor_baseline, vendor_splits):
    return vendor_baseline.run_pop(vendor_splits)["valid"]


@pytest.fixture(scope="module")
def fm_valid_result(vendor_baseline, vendor_splits):
    # k=16, lr=0.001, seed=0 is the official FM baseline configuration
    # (harness/SCHEMA_NOTES.md baseline.py Q13-Q14 / baseline_scores.json).
    return vendor_baseline.run_fm(
        vendor_splits, k=16, lr=0.001, seed=0, verbose=False
    )["valid"]


def test_random_popularity_fm_ordering_on_validation(
    random_valid_result, pop_valid_result, fm_valid_result
):
    print(
        f"\nvalidation primary — random: {random_valid_result['primary']:.4f} | "
        f"popularity: {pop_valid_result['primary']:.4f} | "
        f"FM: {fm_valid_result['primary']:.4f}"
    )
    assert random_valid_result["primary"] < pop_valid_result["primary"] < fm_valid_result["primary"]


def test_fm_baseline_matches_published_validation_primary(fm_valid_result):
    print(
        f"\nFM validation — GAUC {fm_valid_result['GAUC']:.4f} | "
        f"nDCG@5 {fm_valid_result['nDCG@5']:.4f} | "
        f"primary {fm_valid_result['primary']:.4f} "
        f"(published: {FM_VALID_PUBLISHED['primary']})"
    )
    assert abs(fm_valid_result["primary"] - FM_VALID_PUBLISHED["primary"]) <= FM_PRIMARY_TOLERANCE, (
        f"FM validation primary {fm_valid_result['primary']:.4f} misses published "
        f"{FM_VALID_PUBLISHED['primary']} by more than {FM_PRIMARY_TOLERANCE}"
    )


def test_oracle_ceiling_on_validation():
    """Organizers publish oracle (true-labels-as-scores) figures only on
    hidden TEST: GAUC 1.0, nDCG@5 0.7289, primary 0.8645, with 27.1%
    zero-positive / 9.2% all-positive users (harness/SCHEMA_NOTES.md
    README section). We can't score test, so we reproduce this on
    validation and print it for comparison instead of asserting the
    split-specific (nDCG@5, primary, user-mix) numbers, which have no
    reason to match exactly on a different split.
    """
    val_rows = data.load("val")
    user_ids = [row[1] for row in val_rows]
    labels = [row[6] for row in val_rows]

    oracle = metrics.evaluate(user_ids, labels, labels)

    labels_by_user = collections.defaultdict(list)
    for row in val_rows:
        labels_by_user[row[1]].append(row[6])
    n_users = len(labels_by_user)
    zero_positive_frac = sum(1 for labs in labels_by_user.values() if sum(labs) == 0) / n_users
    all_positive_frac = sum(1 for labs in labels_by_user.values() if sum(labs) == len(labs)) / n_users

    print(
        f"\nvalidation oracle — GAUC {oracle.values['GAUC']:.4f} | "
        f"nDCG@5 {oracle.values['nDCG@5']:.4f} | primary {oracle.primary:.4f} | "
        f"zero-positive users {zero_positive_frac:.1%} | "
        f"all-positive users {all_positive_frac:.1%}  "
        f"(organizers, hidden test: GAUC 1.0000, nDCG@5 0.7289, primary 0.8645, "
        f"zero-positive 27.1%, all-positive 9.2%)"
    )

    # GAUC=1.0 for the oracle is split-invariant: true labels used as
    # scores perfectly separate positives from negatives for every
    # discriminative user, regardless of which split we're on.
    assert oracle.values["GAUC"] == pytest.approx(1.0)
