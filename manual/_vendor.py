"""Loads the read-only vendored baseline module by file path.

The vendored starter kit is never edited, only wrapped — so the FM class,
`encode`, and `FIELDS` are reached by importing
vendor/kuairand-starter-kit/baseline.py as a module object rather than by
copying its code.

WHY BY FILE PATH AND NOT A NORMAL IMPORT. Two reasons, both structural:
the vendor directory name contains a hyphen (not a valid Python
identifier, so `import kuairand-starter-kit.baseline` cannot be written
at all), and its modules are named `data.py` and `evaluate.py`, which
would collide with harness modules of the same name if the vendor
directory were left on sys.path.

baseline.py's own top-level `from data import load, encode, FIELDS` and
`from evaluate import evaluate` rely on Python's ordinary sys.path
search, so the vendor directory goes onto sys.path just long enough to
exec the module and is removed in a `finally` — the path is never left
mutated, even if exec raises.

THIS PATTERN IS COPIED FROM executor/realize.py's `_load_vendor_baseline`,
NOT IMPORTED FROM IT. See manual/__init__.py on the import boundary: this
package must not depend on W3. Copying ~20 lines of importlib boilerplate
is the whole cost of that independence, and harness/data.py and
harness/metrics.py each carry their own copy for the same reason.

Because baseline.py re-exports its own imports, the returned module
exposes everything this package needs from one object:

    vendor.FM         the factorization machine class
    vendor.encode     the vendor encoder (the fidelity reference)
    vendor.FIELDS     ['user_id','video_id','author_id','tab','dur_bucket']
    vendor.evaluate   the official scorer (harness.metrics wraps this)
    vendor.sigmoid    used by FM.step
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "vendor" / "kuairand-starter-kit"
VENDOR_BASELINE_PY = VENDOR_DIR / "baseline.py"


def load_vendor_baseline():
    """Imports vendor/kuairand-starter-kit/baseline.py and returns it."""
    vendor_dir_str = str(VENDOR_DIR)
    sys.path.insert(0, vendor_dir_str)
    try:
        spec = importlib.util.spec_from_file_location(
            "_manual_vendor_kuairand_baseline", VENDOR_BASELINE_PY
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(vendor_dir_str)
    return module


vendor = load_vendor_baseline()
"""Module-level singleton.

Loaded once at import: exec'ing baseline.py re-parses three files, and
the module is stateless (a class, some pure functions, and a constant),
so there is nothing to gain from re-loading it per call. Note the module
name given to importlib is distinct from executor/realize.py's, so both
packages can be imported into one process without either clobbering the
other's entry in sys.modules.
"""
