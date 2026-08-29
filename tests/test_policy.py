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

import json

import pytest

from contracts import SLOT_ORDER, SlotName
from controller.policy import CostAwareBanditPolicy, FixedOrderPolicy, UniformPolicy
from controller.ports import PolicyPort
from controller.state import (
    STAGE_SLOTS,
    RunState,
    Stage,
    slot_stats,
    slot_stats_as_json,
)

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


# ---------------------------------------------------------------------------
# CostAwareBanditPolicy
#
# The card shape these tests hand the policy is not invented here: `_arm`
# is pinned against `slot_stats_as_json` by the first test below, so a
# change to what the Controller actually puts on the card fails loudly
# rather than leaving this file asserting against a fiction.
# ---------------------------------------------------------------------------


def _arm(
    attempts: int,
    deltas: tuple[float, ...] = (),
    total_tokens: int = 1200,
    accepted: int = 0,
    total_wall_seconds: float = 40.0,
    consecutive_failures: int = 0,
) -> dict:
    """One slot's entry, in the shape build_state_card renders."""
    return {
        "attempts": attempts,
        "accepted": accepted,
        "total_tokens": total_tokens,
        "total_wall_seconds": total_wall_seconds,
        "deltas": list(deltas),
        "consecutive_failures": consecutive_failures,
    }


def _card(**per_slot: dict) -> dict:
    """A state card carrying nothing but `slot_stats`.

    Deliberately missing every other key. The bandit reads exactly one, and
    a card with the rest filled in would let a test pass while the policy
    quietly depended on something else.
    """
    return {"slot_stats": dict(per_slot)}


def test_the_hand_built_arm_shape_matches_what_the_card_really_carries():
    """Pins `_arm` to `slot_stats_as_json`. Without this, every test below
    could be asserting against a shape the Controller never produces."""
    state = RunState(run_id="r", stage=Stage.STAGE_1_STRUCTURAL)
    state = state.with_outcome(
        "a", 0.61, True, delta=0.02, target_slot="model",
        wall_seconds=40.0, tokens=1200,
    )
    rendered = slot_stats_as_json(slot_stats(state))

    assert set(rendered["model"]) == set(_arm(attempts=1))
    assert rendered["model"] == _arm(
        attempts=1, deltas=(0.02,), total_tokens=1200, accepted=1
    )
    # And the whole thing survives the JSON hop the card promises.
    assert json.loads(json.dumps(rendered, allow_nan=False)) == rendered


def test_the_bandit_satisfies_the_port():
    assert isinstance(CostAwareBanditPolicy(), PolicyPort)


def test_the_bandit_only_ever_returns_an_offered_slot():
    """The one thing every policy must do; a miss is a PolicyContractError
    and stops the run."""
    policy = CostAwareBanditPolicy(seed=5)
    card = _card(
        features=_arm(attempts=3, deltas=(0.01, 0.02, 0.0)),
        weighting=_arm(attempts=1, deltas=(0.05,), total_tokens=100),
        model=_arm(attempts=2, deltas=()),
        objective=_arm(attempts=4, deltas=(-0.3,)),
    )
    for _ in range(50):
        assert policy.select_slot(card, STRUCTURAL) in STRUCTURAL
    # And on a subset — the shrinking candidate set a blocked slot produces.
    for _ in range(50):
        assert policy.select_slot(card, ("model", "objective")) in {
            "model",
            "objective",
        }


def test_the_bandit_handles_a_single_candidate():
    assert CostAwareBanditPolicy().select_slot(_card(), ("calibration",)) == (
        "calibration"
    )
    assert CostAwareBanditPolicy().select_slot(
        _card(calibration=_arm(attempts=9, deltas=(0.01,))), ("calibration",)
    ) == "calibration"


def test_an_empty_candidate_set_is_named_as_a_caller_bug():
    """The Controller ends a stage before this can happen, so reaching it
    means the caller is wrong — and a bare IndexError would say nothing."""
    with pytest.raises(ValueError, match="no candidate slots"):
        CostAwareBanditPolicy().select_slot(_card(), ())


# -- the initialisation pass -------------------------------------------


def test_untried_arms_are_exhausted_before_any_tried_arm_repeats():
    """Every offered slot is pulled once, in SLOT_ORDER, before anything is
    pulled twice. That pass is what the bandit has to score against later."""
    policy = CostAwareBanditPolicy(seed=11)
    stats: dict = {}
    drawn = []
    for _ in range(len(STRUCTURAL)):
        slot = policy.select_slot({"slot_stats": stats}, STRUCTURAL)
        drawn.append(slot)
        # A pull with a strong result, so a policy that started exploiting
        # early would visibly re-pick it instead of moving on.
        stats[slot] = _arm(attempts=1, deltas=(0.05,), total_tokens=100)

    assert drawn == sorted(STRUCTURAL, key=SLOT_ORDER.index)
    assert len(set(drawn)) == len(STRUCTURAL)  # nothing repeated


def test_an_untried_arm_outranks_a_tried_arm_however_good():
    """No score can beat "no data yet" — the initialisation pass is
    unconditional, not merely a high exploration bonus."""
    card = _card(
        weighting=_arm(attempts=40, deltas=(0.5,) * 40, total_tokens=10),
    )
    assert CostAwareBanditPolicy(exploration_c=0.0).select_slot(
        card, ("weighting", "objective")
    ) == "objective"


def test_the_initialisation_order_ignores_the_order_it_was_offered():
    """SLOT_ORDER, not the caller's sequence: two stages offering the same
    arms in different orders must initialise identically."""
    forwards = CostAwareBanditPolicy().select_slot(
        _card(), ("objective", "model", "features")
    )
    backwards = CostAwareBanditPolicy().select_slot(
        _card(), ("features", "model", "objective")
    )
    assert forwards == backwards == "features"


def test_a_zero_attempt_entry_counts_as_untried():
    """`slot_stats` omits untried slots rather than zero-filling them, so
    the two must mean the same thing here."""
    card = _card(
        model=_arm(attempts=0, total_tokens=0),
        objective=_arm(attempts=6, deltas=(0.05,) * 6, total_tokens=10),
    )
    assert CostAwareBanditPolicy(exploration_c=0.0).select_slot(
        card, ("objective", "model")
    ) == "model"


# -- exploitation ------------------------------------------------------


def test_it_picks_the_arm_with_the_better_improvement_per_token():
    """All arms tried, one clearly better value for money. `model` is LATER
    in SLOT_ORDER than `features`, so a policy that fell back to the
    position tie-break would answer `features` and fail here."""
    card = _card(
        features=_arm(attempts=4, deltas=(0.001, 0.001, 0.001, 0.001)),
        weighting=_arm(attempts=4, deltas=(0.002, 0.0, 0.001, 0.0)),
        model=_arm(attempts=4, deltas=(0.02, 0.02, 0.02, 0.02)),
        objective=_arm(attempts=4, deltas=(0.0, 0.0, 0.0, 0.0)),
    )
    policy = CostAwareBanditPolicy(exploration_c=0.001)

    assert policy.select_slot(card, STRUCTURAL) == "model"


def test_equal_improvement_but_cheaper_wins__this_is_the_cost_awareness():
    """THE TEST THAT PROVES IT IS COST-AWARE AND NOT MERELY GREEDY.

    Identical deltas, identical attempt counts — so identical reward and an
    identical exploration bonus. The only thing separating them is what
    they cost. The cheap arm is `model`, deliberately LATER in SLOT_ORDER
    than the expensive `features`, so neither the position tie-break nor a
    reward-only policy can produce the right answer by accident.
    """
    card = _card(
        features=_arm(attempts=2, deltas=(0.01, 0.01), total_tokens=8000),
        model=_arm(attempts=2, deltas=(0.01, 0.01), total_tokens=800),
    )

    for c in (0.0, 0.001, 0.05):
        policy = CostAwareBanditPolicy(exploration_c=c)
        assert policy.select_slot(card, ("features", "model")) == "model"


def test_cost_is_mean_tokens_per_attempt_not_the_running_total():
    """A slot pulled ten times cheaply must not read as expensive just
    because its total is large. `objective` costs 500/attempt against
    `features` at 2000/attempt, on the same reward."""
    card = _card(
        features=_arm(attempts=1, deltas=(0.01,), total_tokens=2000),
        objective=_arm(attempts=10, deltas=(0.01,) * 10, total_tokens=5000),
    )
    assert CostAwareBanditPolicy(exploration_c=0.0).select_slot(
        card, ("features", "objective")
    ) == "objective"


# -- DECISION 3: improvement, not raw delta ----------------------------


def test_a_regression_does_not_drag_an_arm_below_zero_value():
    """PINS THE `max(delta, 0)` CHOICE.

    `features` has two catastrophic rulings and nothing else; `model` has
    two rulings of exactly zero. Clipped, both arms are worth 0, the scores
    tie, and SLOT_ORDER hands it to `features`. Unclipped, `features` would
    score about -0.9 per 1k tokens and `model` would win. So this asserts
    the floor is exactly zero, not merely "less negative".
    """
    card = _card(
        features=_arm(attempts=2, deltas=(-0.9, -0.9), total_tokens=2000),
        model=_arm(attempts=2, deltas=(0.0, 0.0), total_tokens=2000),
    )
    assert CostAwareBanditPolicy(exploration_c=0.0).select_slot(
        card, ("features", "model")
    ) == "features"


def test_one_bad_candidate_does_not_bury_an_otherwise_strong_arm():
    """The reason for the clipping, stated as behaviour. `objective` landed
    one real gain and one disaster; `features` is reliably tiny. Averaging
    raw deltas would put `objective` far below zero and never try it
    again — which is precisely how a high-variance structural arm gets
    buried by a single bad hypothesis."""
    card = _card(
        features=_arm(attempts=2, deltas=(0.001, 0.001)),
        objective=_arm(attempts=2, deltas=(0.02, -0.5)),
    )
    assert CostAwareBanditPolicy(exploration_c=0.0).select_slot(
        card, ("features", "objective")
    ) == "objective"


def test_an_arm_with_attempts_but_no_rulings_stays_in_play():
    """DELIBERATE: a slot whose candidates all failed has value 0 but keeps
    its cost and its exploration bonus, so it is retried. Writing it off on
    zero measurements would be blocking by the back door — stopping it is
    the circuit breaker's job, on executor failures only."""
    all_failed = _arm(attempts=2, deltas=(), total_tokens=2400)
    inert = _arm(attempts=2, deltas=(0.0, 0.0), total_tokens=2400)
    card = _card(model=all_failed, objective=inert)

    # Same value (0), same cost, same attempts: they tie, and the tie
    # resolves by SLOT_ORDER rather than by penalising the ruling-less arm.
    assert CostAwareBanditPolicy(exploration_c=0.05).select_slot(
        card, ("model", "objective")
    ) == "model"

    # And it is genuinely reachable: under-pulled, it outranks a
    # well-pulled inert arm on the exploration term alone.
    lopsided = _card(
        model=_arm(attempts=1, deltas=(), total_tokens=1200),
        objective=_arm(attempts=30, deltas=(0.0,) * 30, total_tokens=36000),
    )
    assert CostAwareBanditPolicy(exploration_c=0.05).select_slot(
        lopsided, ("model", "objective")
    ) == "model"


# -- exploration -------------------------------------------------------


def test_the_exploration_bonus_pulls_an_under_sampled_arm():
    """With c large relative to the reward scale, the under-sampled arm
    wins despite a worse record — that is the bonus doing its job."""
    card = _card(
        features=_arm(attempts=1, deltas=(0.0,)),
        model=_arm(attempts=30, deltas=(0.02,) * 30),
    )
    assert CostAwareBanditPolicy(exploration_c=1.0).select_slot(
        card, ("features", "model")
    ) == "features"


def test_a_zero_exploration_constant_makes_it_purely_greedy():
    card = _card(
        features=_arm(attempts=1, deltas=(0.0,)),
        model=_arm(attempts=30, deltas=(0.02,) * 30),
    )
    assert CostAwareBanditPolicy(exploration_c=0.0).select_slot(
        card, ("features", "model")
    ) == "model"


def test_the_constructor_rejects_settings_that_would_invert_the_score():
    """A non-positive cost floor reopens the division by zero it exists to
    close; a negative constant turns the exploration bonus into a penalty
    on under-sampled arms."""
    with pytest.raises(ValueError, match="cost_floor_tokens"):
        CostAwareBanditPolicy(cost_floor_tokens=0.0)
    with pytest.raises(ValueError, match="cost_floor_tokens"):
        CostAwareBanditPolicy(cost_floor_tokens=-5.0)
    with pytest.raises(ValueError, match="exploration_c"):
        CostAwareBanditPolicy(exploration_c=-0.1)


# -- every division is guarded -----------------------------------------


def test_zero_tokens_falls_back_to_the_cost_floor_instead_of_dividing_by_zero():
    """An attempt that never reached the executor records zero cost. Two
    such arms must still be comparable, and neither may raise."""
    card = _card(
        features=_arm(attempts=2, deltas=(0.01, 0.01), total_tokens=0),
        model=_arm(attempts=2, deltas=(0.005, 0.005), total_tokens=0),
    )
    policy = CostAwareBanditPolicy(exploration_c=0.0, cost_floor_tokens=1.0)

    assert policy.select_slot(card, ("features", "model")) == "features"


def test_a_tiny_cost_floor_is_honoured_rather_than_ignored():
    """The floor is a floor, not a constant: a real cost above it still
    decides, so cost-awareness is not silently switched off."""
    card = _card(
        features=_arm(attempts=1, deltas=(0.01,), total_tokens=4000),
        model=_arm(attempts=1, deltas=(0.01,), total_tokens=1000),
    )
    assert CostAwareBanditPolicy(
        exploration_c=0.0, cost_floor_tokens=0.001
    ).select_slot(card, ("features", "model")) == "model"


@pytest.mark.parametrize(
    "card",
    [
        {},                                   # no slot_stats key at all
        {"slot_stats": {}},                   # present but empty
        {"slot_stats": None},                 # present but not a mapping
        {"slot_stats": {"model": None}},      # entry is not a mapping
        {"slot_stats": {"model": {}}},        # entry with no attempts key
        {"slot_stats": {"model": {"attempts": "nonsense"}}},
    ],
)
def test_a_missing_or_malformed_card_is_read_as_no_data_not_an_error(card):
    """A policy cannot fix a bad card and must not stop the run over one.
    Every one of these degrades to the deterministic SLOT_ORDER sweep."""
    assert CostAwareBanditPolicy().select_slot(card, STRUCTURAL) == "features"


def test_a_single_arm_with_a_single_attempt_does_not_take_log_of_zero():
    """T == 1 makes ln(T) zero and anything smaller undefined. The bonus
    must be absent, not negative and not a ValueError."""
    card = _card(model=_arm(attempts=1, deltas=(0.01,)))
    assert CostAwareBanditPolicy(exploration_c=1.0).select_slot(
        card, ("model",)
    ) == "model"


def test_no_rulings_at_all_across_every_offered_arm():
    """Every arm attempted, none ever judged: value is 0 everywhere, no
    division by an empty delta list, and a choice still comes back."""
    card = _card(**{slot: _arm(attempts=2, deltas=()) for slot in STRUCTURAL})
    assert CostAwareBanditPolicy(exploration_c=0.3).select_slot(
        card, STRUCTURAL
    ) in STRUCTURAL


# -- determinism -------------------------------------------------------


def test_two_instances_with_the_same_seed_make_the_same_choices():
    """The ablation is only a measurement if replaying the policy replays
    the sequence."""
    def sequence(policy) -> list:
        stats: dict = {}
        out = []
        for i in range(24):
            slot = policy.select_slot({"slot_stats": stats}, STRUCTURAL)
            out.append(slot)
            prev = stats.get(slot) or _arm(attempts=0, total_tokens=0)
            stats[slot] = _arm(
                attempts=prev["attempts"] + 1,
                deltas=tuple(prev["deltas"]) + (0.001 * (i % 5),),
                total_tokens=prev["total_tokens"] + 100 * (i % 7 + 1),
            )
        return out

    assert sequence(CostAwareBanditPolicy(seed=4)) == sequence(
        CostAwareBanditPolicy(seed=4)
    )


def test_the_seed_is_not_load_bearing_for_known_slots():
    """Stronger than "reproducible with the same seed": for slots drawn
    from SLOT_ORDER the choice is a pure function of the card, so two
    DIFFERENTLY seeded instances agree. The RNG is the backstop for a tie
    SLOT_ORDER cannot break, not the tie-break."""
    card = _card(
        features=_arm(attempts=3, deltas=(0.01, 0.01, 0.01)),
        weighting=_arm(attempts=3, deltas=(0.01, 0.01, 0.01)),
        model=_arm(attempts=3, deltas=(0.01, 0.01, 0.01)),
        objective=_arm(attempts=3, deltas=(0.01, 0.01, 0.01)),
    )
    picks = {
        CostAwareBanditPolicy(seed=s).select_slot(card, STRUCTURAL)
        for s in range(8)
    }
    assert picks == {"features"}  # the earliest offered arm in SLOT_ORDER


def test_an_unknown_slot_is_survivable_and_the_rng_breaks_the_residual_tie():
    """`SLOT_ORDER.index` raises on a stranger. A policy that crashed on an
    unrecognised arm would turn a caller's mistake into a run-stopper from
    inside the wrong component — and two strangers are the one tie
    SLOT_ORDER genuinely cannot resolve, so the seeded RNG settles it."""
    strangers = ("not_a_slot", "also_not_a_slot")
    card = _card(**{s: _arm(attempts=2, deltas=(0.01, 0.01)) for s in strangers})

    chosen = CostAwareBanditPolicy(seed=1).select_slot(card, strangers)

    assert chosen in strangers
    # Seeded, so the backstop is still reproducible.
    assert chosen == CostAwareBanditPolicy(seed=1).select_slot(card, strangers)
    # And a known slot still outranks nothing here — it just sorts first.
    mixed = _card(
        model=_arm(attempts=2, deltas=(0.01, 0.01)),
        **{s: _arm(attempts=2, deltas=(0.01, 0.01)) for s in strangers},
    )
    assert CostAwareBanditPolicy(seed=1).select_slot(
        mixed, ("not_a_slot", "model")
    ) == "model"
