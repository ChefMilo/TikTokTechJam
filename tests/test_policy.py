"""Tests for controller/policy.py — the real search policies.

These are not fakes, so they get their own file rather than being appended
to tests/test_controller_fakes.py. UniformPolicy is the ablation baseline
the cost-aware bandit will be measured against; if its draws are not
reproducible, that comparison is not a measurement.

Pure unit tests: no Controller, no doubles, no journal. The Controller's
side of the seam (does it actually call the policy, does it validate the
answer) is tested in tests/test_controller.py.
"""

from __future__ import annotations

import pytest

from contracts import SLOT_ORDER, SlotName
from controller.policy import FixedOrderPolicy, UniformPolicy
from controller.ports import PolicyPort
from controller.state import STAGE_SLOTS, Stage

STRUCTURAL: tuple[SlotName, ...] = STAGE_SLOTS[Stage.STAGE_1_STRUCTURAL]

# An empty card. Neither policy here reads it, and that is the point: a
# policy that consulted the state card would not be a control arm.
NO_CARD: dict = {}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [UniformPolicy(), UniformPolicy(seed=17), FixedOrderPolicy(SLOT_ORDER)],
)
def test_policies_satisfy_the_port(policy: object) -> None:
    assert isinstance(policy, PolicyPort)


def test_policy_port_does_not_match_unrelated_objects() -> None:
    """Guards that the check above means something: a runtime_checkable
    Protocol only looks for method names, so a port matching anything
    would make the assertion vacuous."""
    assert not isinstance(object(), PolicyPort)
    assert not isinstance("model", PolicyPort)


# ---------------------------------------------------------------------------
# The contract both must keep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [UniformPolicy(seed=3), FixedOrderPolicy(SLOT_ORDER)],
)
def test_only_ever_returns_a_member_of_the_offered_candidates(policy) -> None:
    """The one thing every policy must do. The Controller raises
    PolicyContractError on a miss, so a violation here is a run-stopper."""
    for _ in range(50):
        assert policy.select_slot(NO_CARD, STRUCTURAL) in STRUCTURAL


@pytest.mark.parametrize(
    "policy",
    [UniformPolicy(seed=3), FixedOrderPolicy(SLOT_ORDER)],
)
def test_a_single_candidate_leaves_no_room_for_creativity(policy) -> None:
    assert policy.select_slot(NO_CARD, ("calibration",)) == "calibration"


# ---------------------------------------------------------------------------
# UniformPolicy
# ---------------------------------------------------------------------------


def _draws(policy, n: int = 40) -> list[SlotName]:
    return [policy.select_slot(NO_CARD, STRUCTURAL) for _ in range(n)]


def test_uniform_policy_is_reproducible_from_its_seed() -> None:
    """Same seed, same sequence — the property the whole ablation rests on.

    Not "same distribution": the exact sequence, because a bandit is
    compared against this arm run for run, and a control that drew a
    different set of slots each time would make any difference between the
    two impossible to attribute.
    """
    assert _draws(UniformPolicy(seed=5)) == _draws(UniformPolicy(seed=5))


def test_uniform_policy_different_seeds_give_different_sequences() -> None:
    """The counterpart: seeding must actually do something. If every seed
    produced the same sequence, the test above would pass while measuring
    nothing at all."""
    assert _draws(UniformPolicy(seed=0)) != _draws(UniformPolicy(seed=1))


def test_uniform_policy_does_not_touch_the_global_random_module() -> None:
    """Instance-owned RNG, so the sequence cannot depend on what else in
    the process happened to draw from `random` first."""
    import random

    random.seed(999)
    before = _draws(UniformPolicy(seed=5))
    random.seed(1)  # anything else in the process reseeds the global stream
    random.random()
    after = _draws(UniformPolicy(seed=5))

    assert before == after


def test_uniform_policy_actually_spreads_across_the_arms() -> None:
    """Uniform must not collapse onto one slot. With 400 draws over four
    arms, every arm appearing is overwhelmingly likely if the draw is
    genuinely uniform and impossible if it is not."""
    counts = {slot: 0 for slot in STRUCTURAL}
    policy = UniformPolicy(seed=11)
    for _ in range(400):
        counts[policy.select_slot(NO_CARD, STRUCTURAL)] += 1

    assert all(count > 0 for count in counts.values()), counts
    assert sum(counts.values()) == 400


def test_uniform_policy_refuses_an_empty_candidate_set() -> None:
    """The Controller ends the stage before this can happen, so reaching it
    is a caller bug and gets a named error rather than an IndexError from
    inside `random.choice`."""
    with pytest.raises(ValueError, match="no candidate slots"):
        UniformPolicy().select_slot(NO_CARD, ())


# ---------------------------------------------------------------------------
# FixedOrderPolicy
# ---------------------------------------------------------------------------


def test_fixed_order_policy_cycles_in_the_given_order() -> None:
    policy = FixedOrderPolicy(("model", "features", "objective"))

    picks = [policy.select_slot(NO_CARD, STRUCTURAL) for _ in range(7)]

    assert picks == [
        "model",
        "features",
        "objective",
        "model",
        "features",
        "objective",
        "model",
    ]


def test_fixed_order_policy_skips_slots_not_currently_on_offer() -> None:
    """`order` is fixed at construction while the offered set changes stage
    by stage and shrinks as slots get blocked, so an order naming an
    unavailable slot is the normal case, not an error."""
    policy = FixedOrderPolicy(("model", "calibration", "features"))
    offered = ("model", "features")  # calibration blocked or out of stage

    picks = [policy.select_slot(NO_CARD, offered) for _ in range(4)]

    assert picks == ["model", "features", "model", "features"]
    assert "calibration" not in picks


def test_fixed_order_policy_resumes_where_it_left_off_across_offer_changes() -> None:
    """The cursor is what makes this a cycle rather than a constant, and it
    must survive a change in what is on offer — a stage boundary must not
    silently restart the rotation."""
    policy = FixedOrderPolicy(("features", "weighting", "model", "objective"))

    first = [policy.select_slot(NO_CARD, STRUCTURAL) for _ in range(2)]
    # STAGE_3_TUNE offers all six; the order still only names four.
    wider = policy.select_slot(NO_CARD, STAGE_SLOTS[Stage.STAGE_3_TUNE])

    assert first == ["features", "weighting"]
    assert wider == "model"  # resumed, not restarted


def test_fixed_order_policy_has_no_randomness_at_all() -> None:
    """Two independently constructed instances agree step for step, with no
    seed passed to either."""
    a = FixedOrderPolicy(SLOT_ORDER)
    b = FixedOrderPolicy(SLOT_ORDER)

    assert [a.select_slot(NO_CARD, STRUCTURAL) for _ in range(12)] == [
        b.select_slot(NO_CARD, STRUCTURAL) for _ in range(12)
    ]


def test_fixed_order_policy_rejects_an_empty_order() -> None:
    with pytest.raises(ValueError, match="at least one slot"):
        FixedOrderPolicy(())


def test_fixed_order_policy_raises_when_nothing_in_its_order_is_offered() -> None:
    """Raises rather than falling back to `candidate_slots[0]`. A silent
    fallback would return a slot this policy was never configured to pick,
    which is exactly the 'the policy secretly did something else' failure
    the Controller's validation exists to catch."""
    policy = FixedOrderPolicy(("calibration",))

    with pytest.raises(ValueError, match="no slot in common"):
        policy.select_slot(NO_CARD, ("model", "features"))


# ---------------------------------------------------------------------------
# Neither may reach outside itself
# ---------------------------------------------------------------------------


def test_policies_ignore_the_state_card_entirely() -> None:
    """Both are history-blind by construction. UniformPolicy in particular
    IS the control arm — a 'uniform policy, but it skips slots that failed
    twice' would be an untested heuristic wearing the control's name, and
    every later measurement would be against a moving target."""
    rich_card = {
        "iteration": 40,
        "blocked_slots": ["model", "features", "weighting", "objective"],
        "convergence": {"flat_streak": 2, "converged": False},
    }

    uniform_with = UniformPolicy(seed=5).select_slot(rich_card, STRUCTURAL)
    uniform_without = UniformPolicy(seed=5).select_slot(NO_CARD, STRUCTURAL)
    fixed_with = FixedOrderPolicy(SLOT_ORDER).select_slot(rich_card, STRUCTURAL)
    fixed_without = FixedOrderPolicy(SLOT_ORDER).select_slot(NO_CARD, STRUCTURAL)

    assert uniform_with == uniform_without
    assert fixed_with == fixed_without
