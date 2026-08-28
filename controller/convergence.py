"""When to stop: two convergence rules, tracked in parallel.

WHAT "ITERATION" MEANS HERE — READ THIS FIRST
---------------------------------------------
An iteration is a **committed revision**: a candidate the gate accepted,
which became the new incumbent. It is NOT every evaluation attempted.

The organizers' rule says "three consecutive iterations where the
validation primary improves by no more than epsilon". A rejected dead end
is not a revision of anything — under the other reading every dead end
explored would count toward N, and a search that tries three bad ideas in
a row would declare itself finished before it had searched at all. That
reading kills exploration outright, so this module counts committed
revisions.

The choice is recorded in every CONVERGENCE_CHECK journal payload as
`iteration_definition`, so a reader can see which definition produced the
number rather than having to infer it. The Controller tracks `node`
alongside `iteration` for exactly this reason: the log can be re-rendered
under the attempt-counting definition later without re-running anything.

A useful consequence: only accepted candidates enter the window, and an
accepted candidate always has a primary. Failed and unrealized candidates
therefore need no special-casing in the rules below — they simply never
appear.

TWO RULES, BOTH LOGGED, EITHER TERMINATES
-----------------------------------------
1. ORGANIZERS' RULE — differenced absolute primaries, `improvement <=
   EPSILON` for the last N revisions. This is their criterion, implemented
   exactly as specified.

2. INTERNAL RULE — no candidate in the last N revisions was
   *significantly* better than the incumbent it replaced, judged by the
   gate's 95% interval excluding zero.

Rule 1 is known to be noise-prone and rule 2 exists because of it: see
`_improvements` for the arithmetic. Both are evaluated on every commit,
both land in one CONVERGENCE_CHECK event, and the run stops when either
fires — so the stricter one in any given run is the one that terminates.

This module is pure: no I/O, no clock, no randomness, no dependency on
controller.state (it reads the committed window through a structural
Protocol, so the dependency arrow runs state -> convergence and never
back).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

__all__ = [
    "EPSILON",
    "ITERATION_DEFINITION",
    "N_CONSECUTIVE",
    "RULE_INTERNAL",
    "RULE_ORGANIZERS",
    "SIGMA",
    "CommittedRevision",
    "ConvergenceStatus",
    "assess",
    "flat_streak",
    "is_significant",
]


# ---------------------------------------------------------------------------
# Constants
#
# Hardcoded with a citation rather than parsed from the vendored JSON at
# import time: a runtime file dependency would make this module fail to
# import on a machine where vendor/ has not been unpacked, and it would put
# file I/O inside something whose whole value is being pure. Drift is caught
# instead by a test that parses the JSON and asserts these still match it.
# ---------------------------------------------------------------------------

EPSILON: float = 0.002
"""Improvement at or below which an iteration counts as flat.

Source: vendor/kuairand-starter-kit/baseline_scores.json ->
convergence_rule.epsilon. The organizers chose it as roughly 2.5 sigma.

Deliberately NOT shared with tests/test_rungs.py's FM_PRIMARY_TOLERANCE,
which happens to be the same number. That one is a reproduction tolerance
- how close our re-run of the FM baseline must land to the published
figure - and this is a convergence threshold. Same value today, entirely
different reasons to change, so they stay separate constants.
"""

N_CONSECUTIVE: int = 3
"""How many consecutive flat iterations end the run.

Source: vendor/kuairand-starter-kit/baseline_scores.json ->
convergence_rule.N.
"""

SIGMA: float = 0.0008
"""Seed-to-seed standard deviation of the primary metric.

Source: vendor/kuairand-starter-kit/baseline_scores.json ->
scores.fm_official.std_over_5_seeds.test_primary. Recorded here because it
is what makes EPSILON interpretable: the threshold is only ~2.5 sigma, so
the search operates where signal and noise are the same order of
magnitude.
"""

ITERATION_DEFINITION = "committed_revision"
"""Stamped into every CONVERGENCE_CHECK payload. See the module docstring."""

RULE_ORGANIZERS = "organizers"
RULE_INTERNAL = "internal"


class CommittedRevision(Protocol):
    """What this module needs to know about one accepted candidate.

    A structural Protocol rather than a concrete type so that
    `controller.state.HistoryEntry` satisfies it as-is, with no conversion
    step and — more importantly — no import of controller.state from here.
    state.py imports convergence for the state card; if convergence
    imported state back, the two would be circular.

    `primary` is declared non-Optional because only *accepted* revisions
    are ever passed, and an accepted candidate always produced scores. A
    failed or unrealized attempt has `primary=None` and never reaches this
    module.
    """

    @property
    def primary(self) -> float: ...

    @property
    def delta(self) -> Optional[float]: ...

    @property
    def significant(self) -> Optional[bool]: ...


@dataclass(frozen=True)
class ConvergenceStatus:
    """The full result of one convergence assessment.

    Frozen and complete: everything a journal reader needs to re-check the
    arithmetic is here, so a CONVERGENCE_CHECK event is self-contained
    rather than something you have to reconstruct from surrounding events.
    """

    converged: bool
    by_rule: Optional[str]
    organizers_converged: bool
    internal_converged: bool

    recent_deltas: tuple[float, ...]
    """The differenced absolute primaries the organizers' rule consumed,
    oldest first. NOT the gate's paired deltas — those are a different
    measurement and live on HistoryEntry.delta and in DECISION events."""

    recent_significant: tuple[bool, ...]
    """Per-revision significance flags the internal rule consumed, oldest
    first. A revision never put to the gate (the baseline) contributes
    False here only if it was genuinely judged; see `_internal_rule`."""

    iterations_considered: int
    """Total committed revisions available, not the window size. A reader
    seeing `iterations_considered: 2, n_required: 3` knows immediately why
    nothing fired."""

    epsilon: float
    n_required: int


def is_significant(ci95: tuple[float, float]) -> bool:
    """True when the interval excludes zero.

    The significance test for the internal rule. `ci95[0] <= 0 <= ci95[1]`
    means the measurement cannot distinguish the candidate from the
    incumbent, so the candidate is NOT significantly better — which is
    what "converged" means for this rule.

    Uses the gate's interval rather than a threshold on the point estimate
    because the gate's delta is a paired per-seed comparison; that pairing
    is the only thing that makes an effect near epsilon detectable at all.
    """
    low, high = ci95
    return not (low <= 0.0 <= high)


def _improvements(committed: Sequence[CommittedRevision]) -> tuple[float, ...]:
    """Per-iteration improvement in absolute primary, oldest first.

    Revision k's improvement is `primary[k] - primary[k-1]`, so a window of
    N improvements needs N+1 revisions. The very first committed revision
    has no predecessor and therefore no improvement — which is exactly why
    the baseline cannot, on its own, converge anything.

    KNOWN NOISE PROBLEM, IMPLEMENTED ANYWAY: these are differences of two
    *absolute* means, each carrying sigma ~= 0.0008, so an improvement
    carries sqrt(2) * sigma ~= 0.0011 of noise. That is over half of
    EPSILON = 0.002. This rule will therefore call a real improvement flat,
    and occasionally call noise a real gain. It is implemented as written
    because it is the organizers' published criterion and we are judged
    against it — and the internal rule exists precisely to provide the
    noise-aware second opinion.
    """
    return tuple(
        committed[i].primary - committed[i - 1].primary
        for i in range(1, len(committed))
    )


def _organizers_rule(
    committed: Sequence[CommittedRevision],
) -> tuple[bool, tuple[float, ...]]:
    """Converged when the last N improvements are all <= EPSILON.

    `<=` is inclusive, matching "improves by no more than epsilon": an
    improvement of exactly EPSILON is flat, not a gain.
    """
    improvements = _improvements(committed)
    if len(improvements) < N_CONSECUTIVE:
        return False, improvements
    window = improvements[-N_CONSECUTIVE:]
    return all(value <= EPSILON for value in window), window


def _internal_rule(
    committed: Sequence[CommittedRevision],
) -> tuple[bool, tuple[bool, ...]]:
    """Converged when none of the last N revisions was significantly better.

    A revision whose `significant` is None was never put to the gate — the
    baseline adoption is the only such case. Unknown is not evidence of
    convergence, so it breaks the streak rather than counting toward it.
    That is what stops a run from declaring itself finished on the strength
    of a baseline it never compared against anything.
    """
    if len(committed) < N_CONSECUTIVE:
        return False, tuple(
            bool(r.significant) for r in committed if r.significant is not None
        )
    window = committed[-N_CONSECUTIVE:]
    flags = tuple(bool(r.significant) for r in window)
    converged = all(r.significant is False for r in window)
    return converged, flags


def assess(committed: Sequence[CommittedRevision]) -> ConvergenceStatus:
    """Evaluate both rules over the committed-revision window.

    `committed` is every accepted candidate so far, oldest first. Fewer
    than N_CONSECUTIVE revisions can never converge: the organizers' rule
    needs N+1 revisions to produce N improvements, and the internal rule
    needs N judged revisions. Both short-circuit rather than deciding on
    partial evidence.

    PRECEDENCE: when both rules fire on the same commit, `by_rule` reports
    "organizers". The externally-defined criterion is the one we have to
    defend publicly, so it is the one the log names as the cause.
    """
    organizers_converged, deltas = _organizers_rule(committed)
    internal_converged, flags = _internal_rule(committed)

    by_rule: Optional[str] = None
    if organizers_converged:
        by_rule = RULE_ORGANIZERS
    elif internal_converged:
        by_rule = RULE_INTERNAL

    return ConvergenceStatus(
        converged=organizers_converged or internal_converged,
        by_rule=by_rule,
        organizers_converged=organizers_converged,
        internal_converged=internal_converged,
        recent_deltas=deltas[-N_CONSECUTIVE:],
        recent_significant=flags[-N_CONSECUTIVE:],
        iterations_considered=len(committed),
        epsilon=EPSILON,
        n_required=N_CONSECUTIVE,
    )


def flat_streak(committed: Sequence[CommittedRevision]) -> int:
    """How many consecutive trailing iterations improved by <= EPSILON.

    Reported on the state card so the generator can see how close the run
    is to stopping — "one more flat iteration and this ends" is actionable
    context for proposing something bolder. Counts past N_CONSECUTIVE
    rather than clamping, because a long flat tail is itself informative.
    """
    streak = 0
    for improvement in reversed(_improvements(committed)):
        if improvement > EPSILON:
            break
        streak += 1
    return streak
