"""Tests for contracts.py — the frozen shared-interface module.

These pin down the hashing/caching behaviour the rest of the project
relies on (slot_hash stability, cascading reuse, seed exclusion) plus the
Metrics and JournalEvent helpers.
"""

from contracts import (
    EventKind,
    JournalEvent,
    Metrics,
    PipelineConfig,
    SlotConfig,
)


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


def test_slot_hash_stable_across_repeated_calls():
    config = PipelineConfig(slots=_slots())
    assert config.slot_hash("weighting") == config.slot_hash("weighting")
    assert config.config_id == config.config_id


def test_slot_hash_stable_across_dict_insertion_order():
    slots_a = _slots()
    slots_b = {name: slots_a[name] for name in reversed(list(slots_a))}

    config_a = PipelineConfig(slots=slots_a)
    config_b = PipelineConfig(slots=slots_b)

    assert config_a.slot_hash("calibration") == config_b.slot_hash("calibration")
    assert config_a.config_id == config_b.config_id


def test_differing_only_in_model_shares_upstream_slot_hash():
    config_a = PipelineConfig(slots=_slots(model=SlotConfig(impl="lightgbm", params={})))
    config_b = PipelineConfig(slots=_slots(model=SlotConfig(impl="xgboost", params={})))

    assert config_a.slot_hash("weighting") == config_b.slot_hash("weighting")
    assert config_a.config_id != config_b.config_id


def test_differing_only_in_seed_has_same_config_id():
    slots = _slots()
    config_a = PipelineConfig(slots=slots, seed=0)
    config_b = PipelineConfig(slots=slots, seed=1)

    assert config_a.config_id == config_b.config_id


def test_changing_code_blob_changes_hash():
    config_a = PipelineConfig(
        slots=_slots(model=SlotConfig(impl="custom", params={}, code_blob="return x"))
    )
    config_b = PipelineConfig(
        slots=_slots(model=SlotConfig(impl="custom", params={}, code_blob="return x + 1"))
    )

    assert config_a.config_id != config_b.config_id


def test_metrics_primary_averages_correctly():
    metrics = Metrics(values={"auc": 0.8, "ndcg": 0.6})
    assert metrics.primary == 0.7


def test_journal_event_roundtrips_through_jsonl():
    event = JournalEvent(
        ts="2026-08-28T00:00:00Z",
        run_id="run-1",
        iteration=2,
        node=5,
        kind=EventKind.DECISION,
        payload={"verdict": True, "delta_primary": 0.003},
    )

    restored = JournalEvent.from_jsonl(event.to_jsonl())

    assert restored == event
