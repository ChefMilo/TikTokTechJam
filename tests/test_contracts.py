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


# ---------------------------------------------------------------------------
# W2 ADDITIONS — appended. Nothing above this line is modified.
#
# Covers the contract types added for the controller (Citation,
# HypothesisPayload, BudgetCounter, Budget), the pre-existing Verdict, and
# the hardened JournalEvent.from_jsonl error handling.
# ---------------------------------------------------------------------------

import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from contracts import (
    BUDGET_COUNTER_ORDER,
    Budget,
    BudgetCounter,
    Citation,
    HypothesisPayload,
    JournalDecodeError,
    Verdict,
)


def _citation() -> Citation:
    return Citation(
        key="rendle2010fm",
        url="https://ieeexplore.ieee.org/document/5694074",
        library_entry="methods/library/fm.yaml#factorization_machine",
    )


def _hypothesis() -> HypothesisPayload:
    return HypothesisPayload(
        target_slot="features",
        rationale="User-side counts are absent; add rolling engagement windows.",
        citation=_citation(),
        expected_gain=0.004,
        expected_cost_s=180.0,
        predecessor_evidence=("a1b2c3d4e5f6", "0f9e8d7c6b5a"),
    )


def _verdict() -> Verdict:
    return Verdict(
        accept=True,
        delta=0.0031,
        ci95=(0.0009, 0.0053),
        n_seeds=5,
        backtest_delta=0.0024,
        reason="paired CI excludes zero on 5 seeds",
    )


def _budget(**overrides) -> Budget:
    base = dict(
        wall_seconds=BudgetCounter(limit=3600.0),
        tokens=BudgetCounter(limit=200_000.0),
        evaluations=BudgetCounter(limit=50.0),
        gpu_seconds=BudgetCounter(limit=900.0),
    )
    base.update(overrides)
    return Budget(**base)


def _event(kind=EventKind.DECISION, **overrides) -> JournalEvent:
    base = dict(
        ts="2026-08-28T00:00:00Z",
        run_id="run-1",
        iteration=2,
        node=5,
        kind=kind,
        payload={"verdict": True, "delta_primary": 0.003},
    )
    base.update(overrides)
    return JournalEvent(**base)


# --- construction and frozen-ness ------------------------------------------


def test_verdict_constructs_and_is_frozen():
    verdict = _verdict()

    assert verdict.accept is True
    assert verdict.delta == 0.0031
    assert verdict.ci95 == (0.0009, 0.0053)
    assert verdict.n_seeds == 5
    assert verdict.backtest_delta == 0.0024
    assert verdict.reason == "paired CI excludes zero on 5 seeds"

    with pytest.raises(FrozenInstanceError):
        verdict.accept = False


def test_citation_constructs_and_is_frozen():
    citation = _citation()

    assert citation.key == "rendle2010fm"
    assert citation.url.startswith("https://")
    assert citation.library_entry.endswith("factorization_machine")

    with pytest.raises(FrozenInstanceError):
        citation.key = "someone_elses_paper"


def test_hypothesis_payload_constructs_and_is_frozen():
    hypothesis = _hypothesis()

    assert hypothesis.target_slot == "features"
    assert hypothesis.citation == _citation()
    assert hypothesis.expected_gain == 0.004
    assert hypothesis.expected_cost_s == 180.0
    assert hypothesis.predecessor_evidence == ("a1b2c3d4e5f6", "0f9e8d7c6b5a")

    with pytest.raises(FrozenInstanceError):
        hypothesis.expected_gain = 1.0


def test_hypothesis_payload_predecessor_evidence_defaults_to_empty():
    first_of_run = HypothesisPayload(
        target_slot="model",
        rationale="first proposal of the run, nothing to build on yet",
        citation=_citation(),
        expected_gain=0.002,
        expected_cost_s=60.0,
    )

    assert first_of_run.predecessor_evidence == ()


def test_hypothesis_payload_asdict_matches_documented_payload_keys():
    """Journal readers were written against the dict sketch in the
    payload-shapes comment; asdict() must still produce exactly it."""
    payload = asdict(_hypothesis())

    assert set(payload) == {
        "target_slot",
        "rationale",
        "citation",
        "expected_gain",
        "expected_cost_s",
        "predecessor_evidence",
    }
    assert set(payload["citation"]) == {"key", "url", "library_entry"}


def test_budget_counter_constructs_and_is_frozen():
    counter = BudgetCounter(limit=100.0, consumed=25.0)

    assert counter.limit == 100.0
    assert counter.consumed == 25.0
    assert BudgetCounter().limit == float("inf")
    assert BudgetCounter().consumed == 0.0

    with pytest.raises(FrozenInstanceError):
        counter.consumed = 50.0


def test_budget_constructs_and_is_frozen():
    budget = _budget()

    assert budget.wall_seconds.limit == 3600.0
    assert budget.tokens.limit == 200_000.0
    assert budget.evaluations.limit == 50.0
    assert budget.gpu_seconds.limit == 900.0
    assert Budget().wall_seconds == BudgetCounter()

    with pytest.raises(FrozenInstanceError):
        budget.tokens = BudgetCounter()


# --- BudgetCounter boundaries ----------------------------------------------


def test_budget_counter_remaining_and_exhausted_at_boundaries():
    under = BudgetCounter(limit=10.0, consumed=4.0)
    assert under.remaining == 6.0
    assert under.exhausted is False

    exactly_at = BudgetCounter(limit=10.0, consumed=10.0)
    assert exactly_at.remaining == 0.0
    assert exactly_at.exhausted is True

    overshot = BudgetCounter(limit=10.0, consumed=12.5)
    assert overshot.remaining == -2.5
    assert overshot.exhausted is True

    assert BudgetCounter(limit=0.0).exhausted is True
    assert BudgetCounter(limit=float("inf"), consumed=1e12).exhausted is False


# --- Budget aggregation -----------------------------------------------------


def test_budget_counter_order_is_fixed_and_declared():
    assert BUDGET_COUNTER_ORDER == (
        "wall_seconds",
        "tokens",
        "evaluations",
        "gpu_seconds",
    )


def test_fresh_budget_is_not_exhausted_and_trips_nothing():
    budget = _budget()

    assert budget.tripped == ()
    assert budget.exhausted is False


@pytest.mark.parametrize("name", BUDGET_COUNTER_ORDER)
def test_budget_exhausted_trips_independently_on_each_counter(name):
    fresh = _budget()
    assert fresh.exhausted is False

    counter = getattr(fresh, name)
    spent = replace(fresh, **{name: replace(counter, consumed=counter.limit)})

    assert spent.exhausted is True
    assert spent.tripped == (name,)
    for other in BUDGET_COUNTER_ORDER:
        if other != name:
            assert getattr(spent, other).exhausted is False


def test_budget_tripped_reports_every_tripped_counter_in_fixed_order():
    budget = Budget(
        wall_seconds=BudgetCounter(limit=10.0, consumed=10.0),
        tokens=BudgetCounter(limit=10.0, consumed=1.0),
        evaluations=BudgetCounter(limit=5.0, consumed=99.0),
        gpu_seconds=BudgetCounter(limit=10.0, consumed=10.0),
    )

    assert budget.tripped == ("wall_seconds", "evaluations", "gpu_seconds")
    assert budget.exhausted is True


# --- JournalEvent.from_jsonl hardening --------------------------------------


def test_from_jsonl_raises_journal_decode_error_on_malformed_json():
    truncated = _event().to_jsonl()[:25]

    with pytest.raises(JournalDecodeError) as excinfo:
        JournalEvent.from_jsonl(truncated)

    assert "not valid JSON" in str(excinfo.value)


def test_from_jsonl_raises_journal_decode_error_on_missing_required_key():
    data = json.loads(_event().to_jsonl())
    del data["node"]

    with pytest.raises(JournalDecodeError) as excinfo:
        JournalEvent.from_jsonl(json.dumps(data))

    assert "node" in str(excinfo.value)


def test_from_jsonl_raises_journal_decode_error_on_unknown_event_kind():
    data = json.loads(_event().to_jsonl())
    data["kind"] = "not_a_real_kind"

    with pytest.raises(JournalDecodeError) as excinfo:
        JournalEvent.from_jsonl(json.dumps(data))

    assert "not_a_real_kind" in str(excinfo.value)


def test_from_jsonl_never_leaks_a_bare_keyerror():
    """A torn final line is the normal case after a crash; replay must be
    able to catch one exception type rather than three unrelated ones."""
    for broken in (
        '{"not": "an event"}',
        json.dumps({"ts": "t", "run_id": "r"}),
        "[1, 2, 3]",
        "",
    ):
        with pytest.raises(JournalDecodeError):
            JournalEvent.from_jsonl(broken)


def test_journal_decode_error_is_a_valueerror_and_not_a_keyerror():
    assert issubclass(JournalDecodeError, ValueError)
    assert not issubclass(JournalDecodeError, KeyError)


def test_journal_decode_error_truncates_the_offending_line():
    long_line = '{"kind": "' + "x" * 500

    with pytest.raises(JournalDecodeError) as excinfo:
        JournalEvent.from_jsonl(long_line)

    message = str(excinfo.value)
    assert "..." in message
    assert len(message) < 300
    assert excinfo.value.line == long_line


def test_valid_lines_still_round_trip_unchanged_for_every_event_kind():
    """Guards that the error handling did not alter behaviour on valid
    input: every kind must parse to an identical object and re-serialize
    to a byte-identical line."""
    for i, kind in enumerate(EventKind):
        event = _event(
            kind=kind,
            iteration=i,
            node=i * 2,
            payload={"i": i, "nested": {"a": [1, 2, None]}, "s": "line\nbreak"},
        )
        line = event.to_jsonl()

        restored = JournalEvent.from_jsonl(line)

        assert restored == event
        assert restored.kind is kind
        assert restored.to_jsonl() == line
