"""Tests for methods.scripted.ScriptedGenerator — the deterministic,
non-LLM hypothesis generator for integration checkpoint I2.
"""

import pytest

from contracts import SLOT_ORDER, SlotConfig
from methods.scripted import ScriptedGenerator

_EXPECTED_PAYLOAD_KEYS = {
    "target_slot",
    "rationale",
    "citation",
    "expected_gain",
    "expected_cost_s",
    "predecessor_evidence",
}
_EXPECTED_CITATION_KEYS = {"key", "url", "library_entry"}


def test_propose_returns_ten_schema_conforming_moves_then_stops():
    generator = ScriptedGenerator()

    for _ in range(10):
        slot_config, payload = generator.propose(state=None)

        assert isinstance(slot_config, SlotConfig)
        assert isinstance(slot_config.impl, str) and slot_config.impl
        assert isinstance(slot_config.params, dict)

        assert set(payload) == _EXPECTED_PAYLOAD_KEYS
        assert payload["target_slot"] in SLOT_ORDER
        assert isinstance(payload["rationale"], str) and payload["rationale"].strip()
        assert set(payload["citation"]) == _EXPECTED_CITATION_KEYS
        assert all(payload["citation"][key] for key in _EXPECTED_CITATION_KEYS)
        assert isinstance(payload["expected_gain"], float)
        assert isinstance(payload["expected_cost_s"], float)
        assert payload["expected_cost_s"] > 0
        assert payload["predecessor_evidence"] == ()

    with pytest.raises(StopIteration):
        generator.propose(state=None)


def test_order_is_deterministic_across_fresh_instances():
    generator_a = ScriptedGenerator()
    generator_b = ScriptedGenerator()

    moves_a = [generator_a.propose(state=None) for _ in range(10)]
    moves_b = [generator_b.propose(state=None) for _ in range(10)]

    rationales_a = [payload["rationale"] for _, payload in moves_a]
    rationales_b = [payload["rationale"] for _, payload in moves_b]
    assert rationales_a == rationales_b

    configs_a = [(sc.impl, sc.params) for sc, _ in moves_a]
    configs_b = [(sc.impl, sc.params) for sc, _ in moves_b]
    assert configs_a == configs_b


def test_reset_restarts_the_script():
    generator = ScriptedGenerator()
    first_slot_config, first_payload = generator.propose(state=None)
    for _ in range(9):
        generator.propose(state=None)
    with pytest.raises(StopIteration):
        generator.propose(state=None)

    generator.reset()
    restarted_slot_config, restarted_payload = generator.propose(state=None)

    assert restarted_slot_config == first_slot_config
    assert restarted_payload == first_payload


def test_first_move_is_the_baseline_and_last_is_tuning():
    generator = ScriptedGenerator()

    first_slot_config, first_payload = generator.propose(state=None)
    assert first_payload["target_slot"] == "model"
    assert first_slot_config.params.get("k") == 16
    assert first_slot_config.params.get("lr") == 0.001
    assert "baseline" in first_payload["rationale"].lower()

    for _ in range(8):
        generator.propose(state=None)
    last_slot_config, last_payload = generator.propose(state=None)
    assert last_payload["target_slot"] == "model"
    assert "epochs" in last_slot_config.params  # the lr/epoch sweep


def test_no_move_reintroduces_static_side_feature_injection():
    """The organizers' own ablation (harness/SCHEMA_NOTES.md) already
    showed static side features don't help on this dataset — nothing
    scripted here should silently re-add that experiment."""
    banned_impls = {"static_side_features", "user_side_features", "video_side_features", "cwm13_features"}
    generator = ScriptedGenerator()

    for _ in range(10):
        slot_config, _ = generator.propose(state=None)
        assert slot_config.impl not in banned_impls


def test_every_move_has_a_distinct_real_citation_key():
    generator = ScriptedGenerator()
    keys = []
    for _ in range(10):
        _, payload = generator.propose(state=None)
        keys.append(payload["citation"]["key"])

    assert len(keys) == len(set(keys))  # every move cites something distinct
