"""Two independent caching responsibilities, both content-addressed so
concurrent workers can never collide on a key that means something else.

(a) FEATURE/ARTIFACT CACHE — keyed on PipelineConfig.slot_hash(slot).

    get(slot_hash) -> artifact or None
    put(slot_hash, artifact) -> None

    Stored under artifacts/cache/<slot_hash>/. `seed` is deliberately
    excluded from slot_hash (see contracts.PipelineConfig), so the 3-5
    seeds the noise gate runs for one config share a single cached
    feature build instead of rebuilding it per seed — that sharing is
    the whole point of the cascading hash.

(b) PREDICTION VECTORS — the schema every other reader depends on.

    save_predictions(config_id, seed, split, user_ids, labels, scores)
    load_predictions(config_id, seed, split) -> (user_ids, labels, scores)
    exists(config_id, seed, split) -> bool

    Stored as ONE file per (config_id, seed, split):
        artifacts/preds/<config_id>__<seed>__<split>.npz
    containing exactly three same-length arrays:
        user_ids : whatever dtype was passed in (typically str, matching
                   the vendor row's user_id field — see harness/data.py)
        labels   : int8, 0/1
        scores   : float32
    One file holding all three arrays is deliberate: it makes the three
    arrays impossible to load out of alignment with each other (no
    separate files that could be regenerated independently and drift
    apart), and impossible to save partially (np.savez writes one file).

    THIS SCHEMA IS READ BY: harness.gate (the noise gate's per-user
    bootstrap), the controller's blending step, and the report renderer.
    Changing array names, dtypes, or the path format is a cross-cutting
    change — update all three.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = REPO_ROOT / "artifacts" / "cache"
_PREDS_DIR = REPO_ROOT / "artifacts" / "preds"


# ---------------------------------------------------------------------------
# (a) Feature/artifact cache
# ---------------------------------------------------------------------------


def _artifact_path(slot_hash: str) -> Path:
    return _CACHE_DIR / slot_hash / "artifact.pkl"


def get(slot_hash: str) -> Any | None:
    """Returns the cached artifact for `slot_hash`, or None if absent."""
    path = _artifact_path(slot_hash)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def put(slot_hash: str, artifact: Any) -> None:
    """Caches `artifact` under `slot_hash`, overwriting any prior value."""
    path = _artifact_path(slot_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)


# ---------------------------------------------------------------------------
# (b) Prediction vectors
# ---------------------------------------------------------------------------


def _pred_path(config_id: str, seed: int, split: str) -> Path:
    return _PREDS_DIR / f"{config_id}__{seed}__{split}.npz"


def save_predictions(
    config_id: str,
    seed: int,
    split: str,
    user_ids,
    labels,
    scores,
) -> None:
    """Writes the three arrays for one (config_id, seed, split) in a
    single .npz, per the module docstring's schema.
    """
    path = _pred_path(config_id, seed, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        user_ids=np.asarray(user_ids),
        labels=np.asarray(labels, dtype=np.int8),
        scores=np.asarray(scores, dtype=np.float32),
    )


def load_predictions(config_id: str, seed: int, split: str):
    """Returns (user_ids, labels, scores) for one (config_id, seed, split).

    Raises FileNotFoundError, naming the exact key, if nothing was ever
    saved for it — callers (harness.gate in particular) must not treat a
    missing prediction file the same as an empty one.
    """
    path = _pred_path(config_id, seed, split)
    if not path.exists():
        raise FileNotFoundError(
            f"no predictions cached for config_id={config_id!r} seed={seed} "
            f"split={split!r} (expected {path}); call save_predictions first"
        )
    with np.load(path) as data:
        return data["user_ids"], data["labels"], data["scores"]


def exists(config_id: str, seed: int, split: str) -> bool:
    return _pred_path(config_id, seed, split).exists()
