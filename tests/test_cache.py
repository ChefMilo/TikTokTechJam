"""Tests for harness.cache — both the slot_hash-keyed artifact cache and
the (config_id, seed, split)-keyed prediction store. All writes go
through a per-test tmp_path so the real artifacts/ directory is never
touched.
"""

import numpy as np
import pytest

from contracts import PipelineConfig, SlotConfig


def _slots(**overrides):
    base = {
        "data_view": SlotConfig(impl="default_view", params={}),
        "features": SlotConfig(impl="basic_features", params={"window": 7}),
        "weighting": SlotConfig(impl="uniform", params={}),
        "model": SlotConfig(impl="lightgbm", params={"n_estimators": 100}),
        "objective": SlotConfig(impl="binary_logloss", params={}),
        "calibration": SlotConfig(impl="none", params={}),
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _isolated_artifacts_dir(tmp_path, monkeypatch):
    """Points harness.cache at a scratch directory instead of the real
    artifacts/ tree, for every test in this file."""
    from harness import cache

    monkeypatch.setattr(cache, "_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "_PREDS_DIR", tmp_path / "preds")
    return cache


def test_artifact_cache_round_trip(_isolated_artifacts_dir):
    cache = _isolated_artifacts_dir
    artifact = {"vocab": {"a": 0, "b": 1}, "field_dims": [2, 3]}

    cache.put("abc123def456", artifact)

    assert cache.get("abc123def456") == artifact


def test_artifact_cache_missing_key_returns_none(_isolated_artifacts_dir):
    cache = _isolated_artifacts_dir

    assert cache.get("does_not_exist") is None


def test_predictions_round_trip_preserves_values_and_order(_isolated_artifacts_dir):
    cache = _isolated_artifacts_dir
    user_ids = np.array(["u3", "u1", "u2", "u1"])
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.7, 0.05], dtype=np.float32)

    cache.save_predictions("cfg1", 0, "val", user_ids, labels, scores)
    loaded_users, loaded_labels, loaded_scores = cache.load_predictions("cfg1", 0, "val")

    assert list(loaded_users) == list(user_ids)
    assert list(loaded_labels) == list(labels)
    assert np.allclose(loaded_scores, scores)
    assert loaded_labels.dtype == np.int8
    assert loaded_scores.dtype == np.float32


def test_load_predictions_missing_key_raises_clear_error(_isolated_artifacts_dir):
    cache = _isolated_artifacts_dir

    with pytest.raises(FileNotFoundError, match="cfgX"):
        cache.load_predictions("cfgX", 3, "val")


def test_exists_reflects_saved_predictions(_isolated_artifacts_dir):
    cache = _isolated_artifacts_dir

    assert cache.exists("cfg2", 1, "val") is False

    cache.save_predictions("cfg2", 1, "val", ["u1"], [1], [0.5])

    assert cache.exists("cfg2", 1, "val") is True


def test_configs_differing_only_in_model_share_a_cache_entry(_isolated_artifacts_dir):
    cache = _isolated_artifacts_dir
    config_a = PipelineConfig(slots=_slots(model=SlotConfig(impl="lightgbm", params={})))
    config_b = PipelineConfig(slots=_slots(model=SlotConfig(impl="xgboost", params={})))

    hash_a = config_a.slot_hash("weighting")
    hash_b = config_b.slot_hash("weighting")
    assert hash_a == hash_b

    artifact = {"built_from": "shared upstream feature matrix"}
    cache.put(hash_a, artifact)

    assert cache.get(hash_b) == artifact
