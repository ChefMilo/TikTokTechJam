"""Tests for controller/convergence.py — the two stopping rules.

These exercise the rules as pure functions over hand-built windows, so
every boundary is pinned with exact numbers rather than inferred from a
run. The Controller-level integration lives in tests/test_controller.py.
"""

from __future__ import annotations

import json
import pathlib
from typing import NamedTuple, Optional

import pytest

from controller.convergence import (
    EPSILON,
    ITERATION_DEFINITION,
    N_CONSECUTIVE,
    RULE_INTERNAL,
    RULE_ORGANIZERS,
    SIGMA,
    ConvergenceStatus,
    assess,
    flat_streak,
    is_significant,
)

BASELINE_SCORES = (
    pathlib.Path(__file__).resolve().parent.parent
    / "vendor"
    / "kuairand-starter-kit"
    / "baseline_scores.json"
)


class Rev(NamedTuple):
    """Stand-in for a committed revision.

    Structurally identical to controller.state.HistoryEntry's relevant
    fields, which is the point: convergence takes a Protocol, so a test can
    build a window without dragging RunState in.
    """

    primary: float
    delta: Optional[float] = None
    significant: Optional[bool] = None


def _flat_window(n: int = N_CONSECUTIVE, start: float = 0.60) -> list[Rev]:
    """n+1 revisions producing n improvements of exactly zero."""
    return [Rev(primary=start, significant=False) for _ in range(n + 1)]


# ---------------------------------------------------------------------------
# The constants are the organizers', not ours
# ---------------------------------------------------------------------------


def test_constants_match_the_vendored_convergence_rule():
    """Hardcoded with a citation rather than parsed at import time, so this
    test is what catches drift if the organizers ever revise the kit."""
    published = json.loads(BASELINE_SCORES.read_text(encoding="utf-8"))

    assert published["convergence_rule"]["epsilon"] == EPSILON
    assert published["convergence_rule"]["N"] == N_CONSECUTIVE


def test_sigma_matches_the_published_cross_seed_std():
    published = json.loads(BASELINE_SCORES.read_text(encoding="utf-8"))
    std = published["scores"]["fm_official"]["std_over_5_seeds"]

    assert std["test_primary"] == SIGMA


def test_epsilon_is_about_two_and_a_half_sigma():
    """The organizers' stated rationale, kept honest: if either number
    moves, this is the assumption that silently stops holding."""
    assert 2.0 < EPSILON / SIGMA < 3.0


def test_iteration_definition_is_recorded_for_the_journal():
    assert ITERATION_DEFINITION == "committed_revision"


# ---------------------------------------------------------------------------
# Short windows can never converge
# ---------------------------------------------------------------------------


def test_empty_window_does_not_converge():
    status = assess([])

    assert status.converged is False
    assert status.by_rule is None
    assert status.iterations_considered == 0
    assert status.recent_deltas == ()
    assert status.recent_significant == ()


@pytest.mark.parametrize("count", range(1, N_CONSECUTIVE + 1))
def test_fewer_than_n_revisions_never_converges(count: int):
    """Easy off-by-one: the organizers' rule needs N+1 revisions to produce
    N improvements, so even exactly N cannot fire it."""
    window = [Rev(primary=0.60, significant=False) for _ in range(count)]

    status = assess(window)

    assert status.organizers_converged is False
    if count < N_CONSECUTIVE:
        assert status.internal_converged is False


def test_exactly_n_revisions_can_fire_the_internal_rule_only():
    """The internal rule reads N per-revision flags and needs no
    differencing, so N judged revisions suffice for it but not for the
    organizers' rule."""
    window = [Rev(primary=0.60, significant=False) for _ in range(N_CONSECUTIVE)]

    status = assess(window)

    assert status.internal_converged is True
    assert status.organizers_converged is False
    assert status.by_rule == RULE_INTERNAL


# ---------------------------------------------------------------------------
# The organizers' rule
# ---------------------------------------------------------------------------


def test_improvement_of_exactly_epsilon_counts_as_flat():
    """`<=` is inclusive — "improves by no more than epsilon" means an
    improvement of exactly epsilon is flat, not a gain. Pinning the
    boundary because an off-by-one here changes when every run stops.

    Built from 0.0 rather than a realistic ~0.60 primary because only there
    is a step of exactly EPSILON representable in binary floating point: at
    0.60 the intended step lands at 0.0020000000000000018, just over the
    line. That representation error is ~1e-18, fifteen orders of magnitude
    below sigma, so it can never matter to a real measurement — but it
    would make this test assert something untrue, so the precondition below
    is checked explicitly rather than assumed.
    """
    primaries = [0.0 + i * EPSILON for i in range(N_CONSECUTIVE + 1)]
    improvements = [primaries[i] - primaries[i - 1] for i in range(1, len(primaries))]
    assert all(step == EPSILON for step in improvements), "test precondition"

    window = [Rev(primary=p, significant=True) for p in primaries]
    status = assess(window)

    assert status.organizers_converged is True
    assert status.by_rule == RULE_ORGANIZERS
    assert status.recent_deltas == pytest.approx((EPSILON,) * N_CONSECUTIVE)


def test_improvements_below_epsilon_converge_at_realistic_magnitudes():
    """The case that actually occurs: primaries around 0.60, each gaining a
    little less than epsilon."""
    step = EPSILON * 0.9
    primaries = [0.60 + i * step for i in range(N_CONSECUTIVE + 1)]
    window = [Rev(primary=p, significant=True) for p in primaries]

    status = assess(window)

    assert status.organizers_converged is True
    assert all(delta < EPSILON for delta in status.recent_deltas)


def test_improvements_just_over_epsilon_do_not_converge():
    over = EPSILON * 1.5
    primaries = [0.60 + i * over for i in range(N_CONSECUTIVE + 1)]
    window = [Rev(primary=p, significant=True) for p in primaries]

    assert assess(window).organizers_converged is False


def test_negative_improvements_count_as_flat():
    """A regression is not an improvement, so it is certainly "no more than
    epsilon" — a run that keeps getting worse has converged."""
    primaries = [0.60 - i * 0.01 for i in range(N_CONSECUTIVE + 1)]
    window = [Rev(primary=p, significant=True) for p in primaries]

    assert assess(window).organizers_converged is True


def test_n_minus_one_flat_then_a_large_gain_does_not_converge():
    primaries = [0.60, 0.60, 0.60, 0.65]  # three improvements, last is large
    window = [Rev(primary=p, significant=True) for p in primaries]

    assert assess(window).organizers_converged is False


def test_a_large_gain_inside_the_window_resets_it():
    """Only the trailing N improvements matter: a big jump then flatness
    still converges once the jump falls out of the window."""
    with_jump = [Rev(primary=p, significant=True) for p in [0.60, 0.70, 0.70, 0.70]]
    assert assess(with_jump).organizers_converged is False

    then_flat = with_jump + [Rev(primary=0.70, significant=True)]
    assert assess(then_flat).organizers_converged is True


# ---------------------------------------------------------------------------
# The internal rule
# ---------------------------------------------------------------------------


def test_is_significant_reads_the_interval_not_the_point_estimate():
    assert is_significant((0.001, 0.003)) is True     # excludes zero
    assert is_significant((-0.003, -0.001)) is True   # excludes zero, negative
    assert is_significant((-0.001, 0.003)) is False   # straddles zero
    assert is_significant((0.0, 0.003)) is False      # touches zero -> not significant
    assert is_significant((-0.003, 0.0)) is False


def test_n_insignificant_verdicts_converge():
    window = [Rev(primary=0.60, significant=False) for _ in range(N_CONSECUTIVE)]

    status = assess(window)

    assert status.internal_converged is True
    assert status.recent_significant == (False,) * N_CONSECUTIVE


def test_one_significant_verdict_resets_the_internal_rule():
    window = [
        Rev(primary=0.60, significant=False),
        Rev(primary=0.60, significant=True),
        Rev(primary=0.60, significant=False),
    ]

    assert assess(window).internal_converged is False


def test_an_unjudged_revision_breaks_the_internal_streak():
    """`significant is None` means the revision was never put to the gate —
    the baseline adoption is the only such case. Unknown is not evidence of
    convergence, so it must not count toward N."""
    window = [
        Rev(primary=0.60, significant=None),   # the baseline
        Rev(primary=0.60, significant=False),
        Rev(primary=0.60, significant=False),
    ]

    assert assess(window).internal_converged is False


# ---------------------------------------------------------------------------
# Both rules together
# ---------------------------------------------------------------------------


def test_organizers_takes_precedence_when_both_fire():
    """Deterministic reporting: the externally-defined criterion is the one
    we have to defend publicly, so it is the one the log names."""
    window = _flat_window()

    status = assess(window)

    assert status.organizers_converged is True
    assert status.internal_converged is True
    assert status.by_rule == RULE_ORGANIZERS
    assert status.converged is True


def test_either_rule_alone_is_enough_to_converge():
    organizers_only = [Rev(primary=0.60, significant=True) for _ in range(N_CONSECUTIVE + 1)]
    internal_only = [
        Rev(primary=0.60 + i * 0.05, significant=False)
        for i in range(N_CONSECUTIVE + 1)
    ]

    a = assess(organizers_only)
    assert (a.converged, a.by_rule) == (True, RULE_ORGANIZERS)
    assert a.internal_converged is False

    b = assess(internal_only)
    assert (b.converged, b.by_rule) == (True, RULE_INTERNAL)
    assert b.organizers_converged is False


def test_status_is_frozen_and_self_describing():
    from dataclasses import FrozenInstanceError

    status = assess(_flat_window())

    assert status.epsilon == EPSILON
    assert status.n_required == N_CONSECUTIVE
    assert status.iterations_considered == N_CONSECUTIVE + 1
    with pytest.raises(FrozenInstanceError):
        status.converged = False


def test_status_payload_is_json_serialisable():
    """It goes straight into a CONVERGENCE_CHECK journal payload."""
    from dataclasses import asdict

    status = assess(_flat_window())
    blob = json.dumps(asdict(status), allow_nan=False)

    assert json.loads(blob)["by_rule"] == RULE_ORGANIZERS


# ---------------------------------------------------------------------------
# flat_streak, for the state card
# ---------------------------------------------------------------------------


def test_flat_streak_counts_trailing_flat_iterations():
    assert flat_streak([]) == 0
    assert flat_streak([Rev(primary=0.60)]) == 0  # no improvement to judge

    climbing = [Rev(primary=0.60 + i * 0.05) for i in range(4)]
    assert flat_streak(climbing) == 0

    flat_then = climbing + [Rev(primary=climbing[-1].primary)]
    assert flat_streak(flat_then) == 1


def test_flat_streak_is_not_clamped_at_n():
    """A long flat tail is itself informative — the generator should be able
    to see it has been going nowhere for a while, not just 'at the limit'."""
    window = [Rev(primary=0.60) for _ in range(N_CONSECUTIVE + 4)]

    assert flat_streak(window) == N_CONSECUTIVE + 3


def test_flat_streak_stops_at_the_first_real_gain():
    window = [
        Rev(primary=0.60),
        Rev(primary=0.60),  # flat
        Rev(primary=0.70),  # gain, breaks the streak looking backwards
        Rev(primary=0.70),  # flat
        Rev(primary=0.70),  # flat
    ]

    assert flat_streak(window) == 2
