"""Measures FM baseline seed variance on the VALIDATION split.

Runs the vendor Factorization Machine baseline (k=16, lr=0.001) at seeds
0-4 on validation and reports the sample standard deviation of GAUC,
nDCG@5, and primary across seeds — the noise gate's acceptance threshold
(epsilon) is calibrated against this number.

The organizers publish std=0.0008 on primary, but that figure is for the
HIDDEN TEST split (170,588 rows). Validation has 124,909 rows, so a
different sigma here is expected, not a bug — this script measures our
own validation sigma rather than assuming it matches, and never asserts
against the organizers' figure.

Usage: python scripts/seed_variance.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import data  # noqa: E402  (path bootstrap must run first)

_VENDOR_DIR = REPO_ROOT / "vendor" / "kuairand-starter-kit"
_ARTIFACTS_DIR = REPO_ROOT / "artifacts"

SEEDS = [0, 1, 2, 3, 4]
FM_K = 16
FM_LR = 0.001
CONVERGENCE_EPSILON = 0.002

# Organizers' published figure — HIDDEN TEST split, not validation. Kept
# only for the printed comparison line below; never asserted against.
ORGANIZERS_TEST_PRIMARY_STD = 0.0008


def _load_vendor_baseline():
    """Imports vendor/kuairand-starter-kit/baseline.py by file path — the
    same approach tests/test_rungs.py and harness/data.py use for vendor
    modules. baseline.py's own top-level `from data import ...` / `from
    evaluate import ...` rely on Python's normal sys.path search (it's
    designed to run as a script from inside its own directory), so the
    vendor directory is put on sys.path just long enough to exec it.
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


def main():
    baseline = _load_vendor_baseline()
    # baseline.run_fm requires the full train/valid/test dict shape (it
    # builds vocab from 'train' and encodes every key in the dict); only
    # ['valid'] is ever read from its result below.
    splits = baseline.load(str(data.DATA_DIR))

    metric_names = ["GAUC", "nDCG@5", "primary"]
    per_seed = []
    print(f"running FM (k={FM_K}, lr={FM_LR}) on validation for seeds {SEEDS} ...")
    for seed in SEEDS:
        result = baseline.run_fm(splits, k=FM_K, lr=FM_LR, seed=seed, verbose=False)["valid"]
        # vendor evaluate() returns numpy.float32 (scores/labels are numpy
        # arrays) — cast to plain float so this is both json-serializable
        # and unsurprising to consumers of per_seed.
        row = {"seed": seed, **{name: float(result[name]) for name in metric_names}}
        per_seed.append(row)
        print(
            f"  seed {seed} | GAUC {row['GAUC']:.4f} | nDCG@5 {row['nDCG@5']:.4f} "
            f"| primary {row['primary']:.4f}"
        )

    summary = {}
    for name in metric_names:
        values = [row[name] for row in per_seed]
        summary[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),  # sample std, ddof=1
        }

    print()
    header = f"{'seed':>6} " + " ".join(f"{name:>10}" for name in metric_names)
    print(header)
    for row in per_seed:
        print(f"{row['seed']:>6} " + " ".join(f"{row[name]:>10.4f}" for name in metric_names))
    print(f"{'mean':>6} " + " ".join(f"{summary[name]['mean']:>10.4f}" for name in metric_names))
    print(f"{'std':>6} " + " ".join(f"{summary[name]['std']:>10.4f}" for name in metric_names))

    primary_std = summary["primary"]["std"]
    epsilon_over_sigma = CONVERGENCE_EPSILON / primary_std if primary_std else float("inf")

    print()
    print(
        f"validation primary std = {primary_std:.6f} over seeds {SEEDS} "
        f"(124,909 rows) vs organizers' hidden-test std = {ORGANIZERS_TEST_PRIMARY_STD} "
        f"(170,588 rows) — different split, expected to differ, not asserted against."
    )
    print(
        f"convergence epsilon ({CONVERGENCE_EPSILON}) = {epsilon_over_sigma:.2f}x our measured "
        f"validation sigma (organizers calibrated ~2.5x on their test-split sigma)."
    )

    output = {
        "seeds": SEEDS,
        "fm_config": {"k": FM_K, "lr": FM_LR},
        "split": "val",
        "per_seed": per_seed,
        "summary": summary,
        # Top-level, not just nested under summary["primary"]["std"]:
        # harness/gate.py reads this exact key to calibrate the noise
        # gate's acceptance threshold, so it needs a stable, unnested path
        # rather than reaching into summary's structure.
        "sigma_primary": primary_std,
        "organizers_test_primary_std": ORGANIZERS_TEST_PRIMARY_STD,
        "convergence_epsilon": CONVERGENCE_EPSILON,
        "epsilon_over_measured_sigma": epsilon_over_sigma,
    }
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _ARTIFACTS_DIR / "seed_variance.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
