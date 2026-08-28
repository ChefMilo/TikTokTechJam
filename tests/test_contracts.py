"""Tests for the frozen cross-package contracts in ``contracts.py``.

These lock the behaviour three other workstreams depend on: content
hashing (the artifact cache key), journal round-tripping (crash resume),
per-seed validation storage (the noise gate), and budget tripping.
"""

from __future__ import annotations

import inspect
import json

import pytest

from contracts import (
    PIPELINE_ORDER,
    ArtifactCache,
    Budget,
    BudgetCounter,
    CandidateResult,
    CodeRealizer,
    ErrorClass,
    EventKind,
    Evaluator,
    Executor,
    HypothesisGenerator,
    HypothesisPayload,
    Journal,
    JournalEvent,
    Metrics,
    NoiseGate,
    PipelineConfig,
    RunStatus,
    SlotConfig,
    SlotName,
    Verdict,
    canonical_json,
    prediction_key,
    stable_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_metrics(primary: float = 0.71) -> Metrics:
    return Metrics(
        primary=primary,
        metric_name="gauc",
        secondary={"ndcg@5": 0.42, "recall@20": 0.13},
    )


def make_pipeline(**overrides: SlotConfig) -> PipelineConfig:
    """A fully occupied pipeline; pass a SlotConfig to replace one slot."""
    slots = {
        SlotName.DATA_VIEW: SlotConfig(SlotName.DATA_VIEW, "full", {"min_len": 5}),
        SlotName.FEATURE_BLOCKS: SlotConfig(
            SlotName.FEATURE_BLOCKS, "counts", {"windows": [1, 7, 30]}
        ),
        SlotName.SAMPLE_WEIGHTING: SlotConfig(SlotName.SAMPLE_WEIGHTING, "uniform", {}),
        SlotName.MODEL: SlotConfig(SlotName.MODEL, "lightgbm", {"leaves": 63}),
        SlotName.OBJECTIVE: SlotConfig(SlotName.OBJECTIVE, "logloss", {}),
        SlotName.CALIBRATION: SlotConfig(SlotName.CALIBRATION, "isotonic", {}),
        SlotName.BLEND: SlotConfig(SlotName.BLEND, "mean", {"k": 3}),
    }
    for cfg in overrides.values():
        slots[cfg.slot] = cfg
    return PipelineConfig(slots=slots, data_version="kuairand-pure-v1")


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


def test_metrics_constructs_and_defaults_secondary() -> None:
    m = Metrics(primary=0.7, metric_name="gauc")
    assert m.primary == 0.7
    assert m.metric_name == "gauc"
    assert m.secondary == {}
    assert make_metrics().secondary["ndcg@5"] == 0.42


def test_slot_config_constructs() -> None:
    cfg = SlotConfig(SlotName.MODEL, "fm", {"rank": 16})
    assert cfg.slot is SlotName.MODEL
    assert cfg.impl == "fm"
    assert cfg.params == {"rank": 16}
    assert SlotConfig(SlotName.BLEND, "mean").params == {}


def test_pipeline_config_constructs_and_exposes_config_id() -> None:
    p = make_pipeline()
    assert set(p.slots) == set(PIPELINE_ORDER)
    assert p.data_version == "kuairand-pure-v1"
    assert isinstance(p.config_id, str) and len(p.config_id) == 16


def test_candidate_result_constructs_minimally_and_fully() -> None:
    minimal = CandidateResult(config_id="abc123", status=RunStatus.SKIPPED)
    assert minimal.val == {}
    assert minimal.backtest is None
    assert minimal.error_class is ErrorClass.NONE
    assert minimal.traceback is None
    assert minimal.tokens == {} and minimal.pred_paths == {}

    full = CandidateResult(
        config_id="abc123",
        status=RunStatus.OK,
        val={0: make_metrics(0.70), 1: make_metrics(0.72)},
        backtest=make_metrics(0.69),
        error_class=ErrorClass.NONE,
        traceback=None,
        wall_seconds=12.5,
        tokens={"hypothesis": 900, "realize": 1400, "repair": 0},
        gpu_seconds=3.25,
        pred_paths={0: "artifacts/abc123.s0.npy", 1: "artifacts/abc123.s1.npy"},
    )
    assert full.backtest is not None and full.backtest.primary == 0.69
    assert full.tokens["realize"] == 1400
    assert full.wall_seconds == 12.5 and full.gpu_seconds == 3.25


def test_verdict_constructs() -> None:
    v = Verdict(
        accepted=True,
        delta=0.004,
        ci95=(0.001, 0.007),
        n_seeds=5,
        backtest_delta=0.003,
        reason="CI excludes zero",
    )
    assert v.accepted is True and v.ci95 == (0.001, 0.007) and v.n_seeds == 5
    bare = Verdict(accepted=False, delta=-0.001, ci95=(-0.006, 0.004), n_seeds=3)
    assert bare.backtest_delta is None and bare.reason == ""


def test_hypothesis_payload_constructs() -> None:
    h = HypothesisPayload(
        slot=SlotName.FEATURE_BLOCKS,
        method_id="target_encoding_v2",
        citation="Micci-Barreca (2001)",
        rationale="High-cardinality video ids are currently one-hot.",
        expected_gain=0.005,
        proposed=SlotConfig(SlotName.FEATURE_BLOCKS, "target_encoding", {"smooth": 20}),
    )
    assert h.expected_gain == 0.005
    assert h.proposed.slot is SlotName.FEATURE_BLOCKS


def test_journal_event_and_budget_construct() -> None:
    ev = JournalEvent(kind=EventKind.RUN_START, ts=1.0, run_id="r1", seq=0)
    assert ev.payload == {}
    b = Budget()
    assert isinstance(b.wall_seconds, BudgetCounter)
    assert b.wall_seconds.limit == float("inf")
    assert b.wall_seconds.consumed == 0.0


def test_value_objects_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    for obj, attr, value in [
        (make_metrics(), "primary", 0.9),
        (SlotConfig(SlotName.MODEL, "fm"), "impl", "lightgbm"),
        (Verdict(True, 0.1, (0.0, 0.2), 3), "accepted", False),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attr, value)


# ---------------------------------------------------------------------------
# 2. slot_hash: determinism, order-independence, upstream sensitivity
# ---------------------------------------------------------------------------


def test_slot_hash_is_deterministic_across_calls() -> None:
    p = make_pipeline()
    assert p.slot_hash(SlotName.MODEL) == p.slot_hash(SlotName.MODEL)
    assert len(p.slot_hash(SlotName.MODEL)) == 16


def test_slot_hash_is_independent_of_dict_insertion_order() -> None:
    """Two controllers that build the same config in a different order must
    agree, or they will not share cache entries."""
    forward = {
        SlotName.DATA_VIEW: SlotConfig(SlotName.DATA_VIEW, "full", {"a": 1, "b": 2}),
        SlotName.MODEL: SlotConfig(SlotName.MODEL, "lightgbm", {"leaves": 63}),
    }
    reversed_ = {
        SlotName.MODEL: SlotConfig(SlotName.MODEL, "lightgbm", {"leaves": 63}),
        SlotName.DATA_VIEW: SlotConfig(SlotName.DATA_VIEW, "full", {"b": 2, "a": 1}),
    }
    a = PipelineConfig(slots=forward, data_version="v1")
    b = PipelineConfig(slots=reversed_, data_version="v1")

    assert list(a.slots) != list(b.slots)  # genuinely different orderings
    assert a.slot_hash(SlotName.MODEL) == b.slot_hash(SlotName.MODEL)
    assert a.config_id == b.config_id


def test_slot_hash_changes_when_an_upstream_slot_changes() -> None:
    base = make_pipeline()
    changed_upstream = base.with_slot(
        SlotConfig(SlotName.FEATURE_BLOCKS, "counts", {"windows": [1, 7]})
    )
    assert base.slot_hash(SlotName.MODEL) != changed_upstream.slot_hash(SlotName.MODEL)


def test_slot_hash_is_unchanged_by_a_downstream_slot_change() -> None:
    """This is the property that makes feature reuse worthwhile: sweeping
    the blend must not invalidate the cached feature matrix."""
    base = make_pipeline()
    changed_downstream = base.with_slot(SlotConfig(SlotName.BLEND, "mean", {"k": 9}))
    assert base.slot_hash(SlotName.FEATURE_BLOCKS) == changed_downstream.slot_hash(
        SlotName.FEATURE_BLOCKS
    )
    assert base.config_id != changed_downstream.config_id


def test_slot_hash_changes_with_data_version() -> None:
    base = make_pipeline()
    other = PipelineConfig(slots=dict(base.slots), data_version="kuairand-pure-v2")
    assert base.slot_hash(SlotName.MODEL) != other.slot_hash(SlotName.MODEL)


def test_slot_hash_rejects_an_absent_slot() -> None:
    p = PipelineConfig(slots={SlotName.MODEL: SlotConfig(SlotName.MODEL, "fm")})
    with pytest.raises(KeyError):
        p.slot_hash(SlotName.CALIBRATION)


def test_upstream_chain_is_dag_ordered_and_custom_depends_on_everything() -> None:
    p = make_pipeline()
    assert p.upstream_chain(SlotName.MODEL) == (
        SlotName.DATA_VIEW,
        SlotName.FEATURE_BLOCKS,
        SlotName.SAMPLE_WEIGHTING,
        SlotName.MODEL,
    )
    with_custom = p.with_slot(SlotConfig(SlotName.CUSTOM, "agent_code", {"v": 1}))
    assert with_custom.upstream_chain(SlotName.CUSTOM) == PIPELINE_ORDER + (
        SlotName.CUSTOM,
    )


# ---------------------------------------------------------------------------
# 3. The seed is NOT an input to slot_hash
# ---------------------------------------------------------------------------


def test_slot_hash_takes_no_seed_parameter() -> None:
    """Seed-independence is structural, not incidental: one cached feature
    matrix must serve every seed."""
    params = inspect.signature(PipelineConfig.slot_hash).parameters
    assert list(params) == ["self", "slot"]
    assert "seed" not in params


def test_slot_hash_is_constant_while_prediction_key_varies_by_seed() -> None:
    p = make_pipeline()
    keys = {seed: p.slot_hash(SlotName.FEATURE_BLOCKS) for seed in (0, 1, 2, 99)}
    assert len(set(keys.values())) == 1, "slot_hash must not depend on the seed"

    pred_keys = {prediction_key(p.config_id, seed) for seed in (0, 1, 2, 99)}
    assert len(pred_keys) == 4, "prediction keys must be distinct per seed"
    assert prediction_key(p.config_id, 0) == prediction_key(p.config_id, 0)
    assert prediction_key(p.config_id, 0) not in keys.values()


# ---------------------------------------------------------------------------
# 4. JournalEvent round-trip
# ---------------------------------------------------------------------------


def test_journal_event_round_trips_losslessly() -> None:
    original = JournalEvent(
        kind=EventKind.DECISION,
        ts=1724851200.125,
        run_id="run-2026-08-28-01",
        seq=417,
        payload={
            "accepted": True,
            "delta": 0.0041,
            "ci95": [0.0009, 0.0073],
            "slot": SlotName.MODEL.value,
            "reason": "CI excludes zero\non 5 paired seeds",
            "nested": {"z": 1, "a": [1, 2, {"deep": None}]},
        },
    )
    line = original.to_json()
    assert "\n" not in line, "a journal event must serialise to exactly one line"

    restored = JournalEvent.from_json(line)
    assert restored == original
    assert restored.kind is EventKind.DECISION
    assert isinstance(restored.seq, int) and restored.seq == 417
    assert restored.ts == original.ts
    assert restored.payload["nested"]["a"][2]["deep"] is None
    assert restored.payload["reason"] == original.payload["reason"]


def test_journal_event_round_trips_every_event_kind() -> None:
    for i, kind in enumerate(EventKind):
        ev = JournalEvent(kind=kind, ts=float(i), run_id="r", seq=i, payload={"i": i})
        assert JournalEvent.from_json(ev.to_json()) == ev


def test_journal_lines_are_stable_and_jsonl_appendable() -> None:
    a = JournalEvent(EventKind.BUDGET, 1.0, "r", 1, {"b": 2, "a": 1})
    b = JournalEvent(EventKind.BUDGET, 1.0, "r", 1, {"a": 1, "b": 2})
    assert a.to_json() == b.to_json()
    blob = "\n".join(ev.to_json() for ev in (a, b))
    assert [JournalEvent.from_json(x) for x in blob.splitlines()] == [a, b]


def test_journal_event_rejects_malformed_lines() -> None:
    """A truncated tail is expected after a crash; replay must stop there
    rather than guess."""
    with pytest.raises(ValueError):
        JournalEvent.from_json(json.dumps({"kind": "decision", "ts": 1.0}))
    with pytest.raises(ValueError):
        JournalEvent.from_json(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        JournalEvent.from_json('{"kind":"decision","ts":1.0,"run_i')


# ---------------------------------------------------------------------------
# 5. Budget
# ---------------------------------------------------------------------------


def test_fresh_budget_is_not_exhausted() -> None:
    b = Budget()
    assert b.exhausted() is False
    assert b.tripped() == ()


@pytest.mark.parametrize(
    "name", ["wall_seconds", "tokens", "evaluations", "gpu_seconds"]
)
def test_budget_exhausted_trips_on_each_individual_counter(name: str) -> None:
    b = Budget(
        wall_seconds=BudgetCounter(limit=3600.0),
        tokens=BudgetCounter(limit=200_000.0),
        evaluations=BudgetCounter(limit=50.0),
        gpu_seconds=BudgetCounter(limit=900.0),
    )
    assert b.exhausted() is False

    counter = b.counters()[name]
    counter.consumed = counter.limit
    assert b.exhausted() is True
    assert b.tripped() == (name,)

    counter.consumed = counter.limit + 1
    assert b.exhausted() is True and b.tripped() == (name,)

    counter.consumed = counter.limit - 1
    assert b.exhausted() is False and b.tripped() == ()


def test_budget_reports_every_tripped_counter_in_fixed_order() -> None:
    b = Budget(
        wall_seconds=BudgetCounter(limit=10.0, consumed=10.0),
        tokens=BudgetCounter(limit=10.0),
        evaluations=BudgetCounter(limit=5.0, consumed=99.0),
        gpu_seconds=BudgetCounter(limit=10.0),
    )
    assert b.tripped() == ("wall_seconds", "evaluations")


def test_budget_counter_edges() -> None:
    assert BudgetCounter(limit=0.0).exhausted is True  # zero limit: do not start
    assert BudgetCounter(limit=float("inf"), consumed=1e12).exhausted is False
    assert BudgetCounter(limit=10.0, consumed=4.0).remaining == 6.0


# ---------------------------------------------------------------------------
# 6. CandidateResult stores validation per seed
# ---------------------------------------------------------------------------


def test_candidate_result_val_preserves_multiple_distinct_seeds() -> None:
    """Per-seed storage is what lets the gate do a paired comparison;
    collapsing it would destroy the pairing."""
    per_seed = {0: 0.701, 1: 0.694, 7: 0.712, 42: 0.688, 1337: 0.705}
    r = CandidateResult(
        config_id="cfg",
        status=RunStatus.OK,
        val={seed: make_metrics(v) for seed, v in per_seed.items()},
        pred_paths={seed: f"artifacts/cfg.s{seed}.npy" for seed in per_seed},
    )

    assert len(r.val) == 5
    assert r.seeds() == (0, 1, 7, 42, 1337)
    for seed, value in per_seed.items():
        assert r.val[seed].primary == value
        assert r.val[seed].secondary["ndcg@5"] == 0.42
    assert set(r.pred_paths) == set(r.val), "a prediction path per evaluated seed"


def test_mean_primary_is_display_only_and_handles_empty() -> None:
    r = CandidateResult(
        config_id="cfg",
        status=RunStatus.OK,
        val={0: make_metrics(0.70), 1: make_metrics(0.72)},
    )
    assert r.mean_primary() == pytest.approx(0.71)
    assert CandidateResult(config_id="c", status=RunStatus.FAILED).mean_primary() is None


def test_failed_result_carries_classified_error() -> None:
    r = CandidateResult(
        config_id="cfg",
        status=RunStatus.FAILED,
        error_class=ErrorClass.SHAPE_MISMATCH,
        traceback="ValueError: shapes (10,3) and (4,) not aligned",
    )
    assert r.status is RunStatus.FAILED
    assert r.error_class is ErrorClass.SHAPE_MISMATCH
    assert r.traceback is not None and r.mean_primary() is None


# ---------------------------------------------------------------------------
# Enums, hashing helpers and Protocols
# ---------------------------------------------------------------------------


def test_enum_vocabularies_are_closed_and_complete() -> None:
    assert {s.name for s in SlotName} == {
        "DATA_VIEW",
        "FEATURE_BLOCKS",
        "SAMPLE_WEIGHTING",
        "MODEL",
        "OBJECTIVE",
        "CALIBRATION",
        "BLEND",
        "CUSTOM",
    }
    assert {s.name for s in RunStatus} == {"OK", "FAILED", "SKIPPED"}
    assert {e.name for e in ErrorClass} == {
        "NONE",
        "SYNTAX",
        "CONTRACT",
        "SHAPE_MISMATCH",
        "OOM",
        "TIMEOUT",
        "NAN_LOSS",
        "DEGENERATE",
        "DEPENDENCY",
        "UNKNOWN",
    }
    assert [e.name for e in EventKind] == [
        "RUN_START",
        "STAGE_ENTER",
        "HYPOTHESIS",
        "EVAL_START",
        "EVAL_END",
        "DECISION",
        "ERROR",
        "REPAIR_ATTEMPT",
        "RECOVERY",
        "SLOT_BLOCKED",
        "BUDGET",
        "INTERVENTION",
        "CHECKPOINT",
        "RUN_END",
    ]
    assert PIPELINE_ORDER == tuple(s for s in SlotName if s is not SlotName.CUSTOM)


def test_enums_are_str_valued_and_json_safe() -> None:
    assert SlotName.MODEL == "model"
    assert json.loads(json.dumps({"k": EventKind.DECISION}))["k"] == "decision"
    assert EventKind("decision") is EventKind.DECISION


def test_stable_hash_is_order_insensitive_and_rejects_unhashable_params() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    with pytest.raises(TypeError):
        stable_hash({"bad": object()})


def test_content_hashable_value_objects_can_key_dicts() -> None:
    cfg = SlotConfig(SlotName.MODEL, "fm", {"rank": 16})
    same = SlotConfig(SlotName.MODEL, "fm", {"rank": 16})
    assert cfg == same and hash(cfg) == hash(same)
    assert len({cfg, same, make_metrics(), make_metrics()}) == 2


def test_protocols_accept_conforming_implementations() -> None:
    """Structural checks only -- the Protocols carry no implementation."""

    class FakeGate:
        def compare(
            self, candidate: CandidateResult, incumbent: CandidateResult
        ) -> Verdict:
            return Verdict(True, 0.0, (0.0, 0.0), 0)

    class FakeCache:
        def get(self, key: str): ...
        def put(self, key: str, artifact) -> None: ...

    assert isinstance(FakeGate(), NoiseGate)
    assert isinstance(FakeCache(), ArtifactCache)
    assert not isinstance(FakeGate(), ArtifactCache)


@pytest.mark.parametrize(
    "protocol, method",
    [
        (Evaluator, "evaluate"),
        (NoiseGate, "compare"),
        (ArtifactCache, "get"),
        (Executor, "run"),
        (Journal, "append"),
        (HypothesisGenerator, "propose"),
        (CodeRealizer, "realize"),
    ],
)
def test_every_protocol_declares_its_method(protocol: type, method: str) -> None:
    assert hasattr(protocol, method)
