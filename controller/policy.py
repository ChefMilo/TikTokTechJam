"""Search policies: which slot the next candidate attacks.

These are REAL COMPONENTS, not test doubles. controller/fakes.py holds
stand-ins for collaborators another workstream will eventually write; this
module holds the Controller's own search policy, which is W2's to build and
nobody else's. A UniformPolicy in fakes.py would be a category error - the
Controller is not pretending to have a policy while waiting for one, it has
one.

WHY UNIFORM IS THE FIRST IMPLEMENTATION
---------------------------------------
The next PR adds a cost-aware bandit that scores slots on realized
delta-per-1k-tokens. A bandit is only a result if you can say what it beat.
UniformPolicy is that comparison: same Controller, same doubles, same seeds,
same budget, one line different in the wiring. Without it in the repo from
the start, the bandit ships as an assertion that it helps, and the honest
version of that claim - "it produced X% more accepted delta per token than
uniform slot selection over N seeded runs" - is not available at all,
because nobody kept the baseline around to run.

So uniform is not a placeholder to be deleted when the real one lands. It
is the ablation arm, and it stays.

DETERMINISM
-----------
Every policy here draws from an instance-owned `random.Random`, never the
global `random` module, and nothing here does I/O, sleeps or reads the
clock. Two runs with the same seed pull the same sequence of arms. That is
what makes an ablation reproducible: if replaying a policy could produce a
different sequence, a difference between two policies would be
indistinguishable from a difference between two runs of either one.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from contracts import SLOT_ORDER, SlotName

__all__ = ["CostAwareBanditPolicy", "FixedOrderPolicy", "UniformPolicy"]

TOKENS_PER_UNIT = 1000.0
"""The exploitation term's denominator unit: delta per 1k tokens.

Named rather than inlined because it is the scale the whole score is
quoted in, and because `exploration_c` has to be chosen against it - see
CostAwareBanditPolicy for why the two terms are not automatically
comparable just because this constant exists.
"""


def _order_index(slot: SlotName) -> int:
    """Position in SLOT_ORDER, or a sentinel past the end for a stranger.

    `SLOT_ORDER.index` raises ValueError on an unknown slot. A policy is
    handed whatever the Controller offers, and while today that is always
    drawn from STAGE_SLOTS (itself guarded at import to be a subset of
    SLOT_ORDER), a policy that crashes on an unrecognised arm would turn a
    caller's mistake into a run-stopper from inside the wrong component.
    Strangers sort last and are then separated by the seeded RNG.
    """
    try:
        return SLOT_ORDER.index(slot)
    except ValueError:
        return len(SLOT_ORDER)


class UniformPolicy:
    """Picks uniformly at random from the offered slots. The ablation baseline.

    THE INSTRUMENT, NOT THE STOPGAP. See this module's docstring: the
    cost-aware bandit in the next PR is measured against this on
    delta-per-1k-tokens, so it has to exist, be seeded, and stay.

    It is also the right *default* for an early run, which is worth saying
    out loud. With no history to learn from, a policy that concentrates on
    one arm is not exploiting knowledge, it is guessing with extra steps.
    Uniform spends the first few evaluations spreading coverage across
    slots, which is exactly the data a bandit needs before it has anything
    to be greedy about.

    WHAT IT POINTEDLY DOES NOT DO: it ignores `state_card` entirely - no
    peeking at the budget, the history or the convergence streak. That is
    the definition of the baseline arm. A "uniform policy, but it skips
    slots that failed twice" is not a control, it is an untested heuristic
    wearing the control's name, and every later measurement would be
    against a moving target.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        """Instance-owned, never the global `random` module.

        The global module's stream is shared with every other caller in the
        process - a test that seeds it, a library that draws from it, an
        import that happens to run first. A policy drawing from it would
        produce a different sequence of arms depending on what else the
        process did, which would make an ablation comparison meaningless
        and a failing run impossible to reproduce.
        """

    def select_slot(
        self, state_card: Mapping[str, Any], candidate_slots: Sequence[SlotName]
    ) -> SlotName:
        """One slot, drawn uniformly. Never anything outside `candidate_slots`.

        `random.Random.choice` rather than `randrange` + index, so the
        draw is over the offered sequence itself and there is no arithmetic
        between the RNG and the returned value that could put it out of
        range.
        """
        if not candidate_slots:
            # The Controller ends a stage before it gets here, so this is a
            # caller bug rather than a run condition. Named explicitly
            # because `random.choice` on an empty sequence raises
            # IndexError, which says nothing about what went wrong.
            raise ValueError("UniformPolicy was offered no candidate slots")
        return self._rng.choice(tuple(candidate_slots))


class FixedOrderPolicy:
    """Cycles through a fixed slot order. No randomness at all.

    Two jobs, both real.

    AS A TEST INSTRUMENT: a test that wants to assert "the Controller
    passed `weighting` to the generator" cannot do that against a random
    policy without either seeding it and hardcoding the draw (brittle - the
    expected value changes if the candidate set changes) or asserting only
    that the result is a member of the set (weak). Naming the order makes
    the expected value obvious from the fixture.

    AS A DEFAULT: it is a legitimate policy in its own right - round-robin
    over the arms, which spreads coverage as evenly as uniform does but
    without a seed to carry around and with a reproducible order rather
    than merely a reproducible distribution. Where a run wants "try each
    slot in turn" and nothing cleverer, this is that, and it is not a
    lesser thing than a random draw.

    SKIPPING, NOT FAILING: slots in `order` that are not currently on offer
    - blocked, or not part of this stage - are stepped over rather than
    returned or raised on. `order` is written once at construction while
    `candidate_slots` changes stage by stage and shrinks as slots get
    blocked, so an order that names a slot the current stage does not
    permit is the normal case, not an error.

    THE CURSOR ADVANCES ACROSS CALLS, and that is what makes it a cycle
    rather than a constant. It persists on the instance, so a fresh
    instance per run is what keeps two runs comparable - exactly the same
    caveat FakeExecutor documents about its own call-sequence dependence.
    """

    def __init__(self, order: Sequence[SlotName]) -> None:
        if not order:
            raise ValueError("FixedOrderPolicy needs at least one slot in its order")
        self.order: tuple[SlotName, ...] = tuple(order)
        self._cursor = 0

    def select_slot(
        self, state_card: Mapping[str, Any], candidate_slots: Sequence[SlotName]
    ) -> SlotName:
        """The next slot in `order` that is currently on offer.

        Walks at most one full lap from the cursor, so the scan terminates
        whether or not anything matches, and leaves the cursor one past
        whatever it returned - so the following call resumes rather than
        restarting. `state_card` is unread: this policy's whole point is
        that its sequence depends on nothing but its own construction.
        """
        offered = set(candidate_slots)
        for step in range(len(self.order)):
            index = (self._cursor + step) % len(self.order)
            slot = self.order[index]
            if slot in offered:
                self._cursor = (index + 1) % len(self.order)
                return slot
        # Raised rather than falling back to `candidate_slots[0]`. A silent
        # fallback would return a slot this policy was never configured to
        # pick, which is precisely the "policy secretly did something else"
        # failure the Controller's own validation exists to catch - and it
        # would be catching our own fallback rather than a real bug.
        raise ValueError(
            f"FixedOrderPolicy order {self.order} has no slot in common with "
            f"the offered candidates {tuple(candidate_slots)}"
        )


class CostAwareBanditPolicy:
    """Treats each slot as a bandit arm and buys improvement per token.

    UCB1 in shape, with the arm's realized improvement-per-unit-cost where
    UCB1 has a mean reward. The claim it is built to support is narrow and
    checkable: *more accepted improvement per token than uniform slot
    selection, over the same seeded configuration*. tests/test_controller.py
    runs it head-to-head against UniformPolicy for exactly that reason -
    see this module's docstring on why the baseline stays in the repo.

    WHERE THE NUMBERS COME FROM
    ---------------------------
    `state_card["slot_stats"]`, which is `controller.state.slot_stats`
    rendered JSON-safe: per slot, how many times it was pulled, how many of
    those the gate accepted, what they cost, every gate ruling it earned,
    and its current run of executor failures. build_state_card documents at
    length why that key was added to a card W4 also reads, and why a side
    channel would have been worse. A card without the key is treated as a
    card with no data - every arm untried - so this policy degrades to a
    deterministic SLOT_ORDER sweep rather than raising at a caller who
    handed it a bare dict.

    UNTRIED ARMS FIRST, IN SLOT_ORDER
    ---------------------------------
    Any offered slot with no attempts is selected before any tried slot,
    and with several untried the earliest in SLOT_ORDER wins. Ties are
    broken by position rather than by the RNG on purpose: the
    initialisation pass is the part of the run that decides what the whole
    rest of the search is scored against, and a reproducible order makes
    two runs of the same configuration comparable attempt-by-attempt rather
    than only in distribution. It costs nothing - with no data there is no
    basis to prefer one untried arm over another anyway.

    THE SCORE
    ---------
        value = mean(max(delta, 0) for that slot's rulings)   # improvement
        cost  = max(total_tokens / attempts, cost_floor_tokens)
        score = (value / cost) * 1000 + c * sqrt(ln(T) / n)

    where T is the total attempts over the arms currently on offer and n is
    this arm's. The exploitation term is quoted per 1000 tokens so it lands
    on a human scale (a strong arm here is ~0.02 primary for ~1200 tokens,
    so ~0.017 per 1k) instead of ~1e-5.

    IMPROVEMENT, NOT RAW DELTA - AND THIS IS A JUDGEMENT CALL
    --------------------------------------------------------
    `max(delta, 0)` clips regressions to zero before averaging. Written out
    so a reader can disagree with it.

    The argument for clipping: a candidate that scored -0.05 has told you
    that *this particular candidate* does not help. It has not told you the
    slot helps negatively by 0.05, because nothing about the slot forces
    the next hypothesis for it to resemble the last one - the generator can
    propose something entirely different in the same slot. Averaging raw
    deltas lets one catastrophic candidate bury an arm for the rest of the
    run, and the arm most likely to produce a catastrophic candidate is the
    high-variance structural one where the actual headroom is.

    The argument against, honestly: this throws away real information.
    Under a truly bad arm every candidate regresses, and clipping makes
    that look identical to an arm whose candidates merely do nothing. The
    circuit breaker does not help here either - a regression is not a
    failure - so a genuinely harmful arm is distinguished from an inert one
    only by the exploration term eventually spreading pulls elsewhere.

    We take that trade because the loss from burying a good arm early is
    unrecoverable within a run, while the loss from over-exploring an inert
    arm is bounded by the node budget. A reader who disagrees should change
    the one `max(delta, 0.0)` below and re-run the ablation.

    AN ARM WITH ATTEMPTS BUT NO RULINGS IS DELIBERATELY STILL IN PLAY.
    A slot whose candidates all failed, or were all misproposed, has an
    empty `deltas` and therefore value 0 - but it keeps its real cost and
    it keeps accruing an exploration bonus, so it will be retried. That is
    intentional: an arm can fail twice for reasons that have nothing to do
    with the arm (a flaky executor, a generator that wandered), and a
    policy that wrote an arm off on zero measurements would be blocking
    slots by the back door with none of the circuit breaker's evidentiary
    standard. Stopping that arm is the circuit breaker's job, on executor
    failures only, and it is the Controller that enforces it.

    KNOWN LIMITATION - THIS IS COMPUTE-COST-AWARE, NOT COST-AWARE
    ------------------------------------------------------------
    `tokens` reaches history from CandidateResult, which the executor
    fills. The generator's and the realizer's own LLM spend is real money
    and is reported by NO port: RealizerPort returns a SlotConfig and
    nothing else, GeneratorPort a HypothesisPayload. So an attempt that
    died before the executor - a slot mismatch, an unrealized hypothesis -
    is recorded at zero cost when it truly cost tokens, and every attempt
    that did reach the executor is undercounted by the proposal spend.
    Whoever writes the feasibility report must say "per unit of evaluation
    cost", not "per unit of cost". When a port reports proposal tokens,
    HistoryEntry is where the number goes and this class needs no change.

    `exploration_c` IS TUNED TO THIS BENCHMARK'S SCALE - HOW TO RE-DERIVE IT
    ------------------------------------------------------------------------
    The default is 0.005, not the textbook UCB1 value of 1.0. That is not a
    preference, it is arithmetic, and the arithmetic is written out here so
    a reader can redo it for another problem instead of inheriting a magic
    constant.

    The two terms of the score are not automatically commensurable, and
    nothing in the formula makes them so:

      - THE EXPLOITATION TERM is improvement per 1000 tokens (see
        TOKENS_PER_UNIT). Its size is a property of the BENCHMARK. Here an
        improvement worth having is around epsilon = 0.002 on the primary,
        and a candidate costs on the order of 1e3 tokens to evaluate, so
        the term lands around 0.002 per 1k tokens - order 1e-3, reaching
        1e-2 for an unusually strong arm.

      - THE UCB BONUS is `sqrt(ln T / n)`, which is a property of the
        COUNTS ALONE. It is order 1 for any small T and n - about 1.3 at
        T = 5, n = 1, and still about 0.4 after twenty pulls - and it stays
        order 1 whatever the metric, the cost model, or the units.

    So one side is 1e-3 and the other is 1e0, and `exploration_c` is the
    only thing standing between them. It has to be the same order as the
    exploitation term or one side decides every choice on its own: at
    c = 1.0 the bonus sits roughly two to three orders of magnitude above
    the signal, every arm's score is its exploration bonus, and the policy
    round-robins - a differently-spelled uniform, which is the one thing a
    bandit must not be. At c = 0 it never explores and locks onto whatever
    the initialisation pass happened to favour.

    0.005 was chosen BY MEASUREMENT, NOT BY TASTE, and the measurement is
    in the repo. tests/test_controller.py::
    test_the_bandit_pulls_the_good_arm_far_more_often_than_uniform runs this
    policy AS SHIPPED against UniformPolicy over the same seeded
    configuration and requires a wide margin on the good arm;
    ::test_at_the_textbook_exploration_constant_the_bandit_matches_uniform
    passes c = 1.0 explicitly and pins the failure, so the claim that the
    textbook value does not work here cannot quietly rot out of this
    docstring. Both must keep passing for this default to mean anything.

    RE-DERIVE IT FOR ANOTHER PROBLEM rather than porting it. The number
    encodes this benchmark's metric scale and this executor's cost
    profile, and either can move it by orders of magnitude. A metric whose
    meaningful deltas are O(1) rather than O(1e-3) - a raw count, a
    percentage, anything not a probability-scale ranking metric - or a cost
    profile in the millions of tokens rather than the thousands, would need
    a different constant. The recipe: take the improvement a run would be
    pleased with, divide by the tokens a candidate costs, multiply by 1000,
    and start there. That is the scale of the exploitation term, and the
    exploration constant belongs at the same order.

    DETERMINISM. The RNG is instance-owned, never the global module, and is
    consulted only for a residual tie SLOT_ORDER cannot break. Given the
    same seed and the same sequence of cards, two instances make the same
    sequence of choices.
    """

    def __init__(
        self,
        seed: int = 0,
        # 0.005, not textbook UCB1's 1.0. The exploitation term is
        # improvement-per-1k-tokens, order 1e-3 on this benchmark, while
        # sqrt(ln T / n) is order 1 on any benchmark - so a constant of 1.0
        # buries the signal under the bonus and the policy round-robins.
        # Tuned to THIS problem's scale and measured by the ablation test;
        # see the class docstring to re-derive it for another one.
        exploration_c: float = 0.005,
        cost_floor_tokens: float = 1.0,
    ) -> None:
        if cost_floor_tokens <= 0.0:
            # The floor's whole job is to keep the cost denominator away
            # from zero; a non-positive floor would either reintroduce the
            # division by zero or flip the sign of every score.
            raise ValueError(
                f"cost_floor_tokens must be positive, got {cost_floor_tokens!r}"
            )
        if exploration_c < 0.0:
            # A negative constant would turn the exploration bonus into a
            # penalty on under-sampled arms, which is not a conservative
            # bandit - it is one that locks onto whatever it tried first.
            raise ValueError(
                f"exploration_c must be non-negative, got {exploration_c!r}"
            )
        self.seed = seed
        self.exploration_c = exploration_c
        self.cost_floor_tokens = cost_floor_tokens
        self._rng = random.Random(seed)
        """Instance-owned, never the global `random` module - the same
        contract UniformPolicy documents. Consulted only as a last
        tie-break."""

    def select_slot(
        self, state_card: Mapping[str, Any], candidate_slots: Sequence[SlotName]
    ) -> SlotName:
        """One offered slot: an untried arm if there is one, else the best score."""
        if not candidate_slots:
            # The Controller ends a stage before it gets here, so this is a
            # caller bug rather than a run condition - named explicitly for
            # the same reason UniformPolicy names it.
            raise ValueError("CostAwareBanditPolicy was offered no candidate slots")

        stats = self._slot_stats(state_card)
        offered = tuple(candidate_slots)

        untried = [slot for slot in offered if self._attempts(stats, slot) <= 0]
        if untried:
            # Position, not a draw: see the class docstring on why the
            # initialisation pass is deliberately reproducible.
            return min(untried, key=_order_index)

        # Total pulls over the arms actually in play for THIS decision,
        # rather than over every slot the run has ever touched. The
        # comparison is between the arms on offer, and a stage that permits
        # four of six should not have its exploration bonus inflated by
        # pulls on the two it cannot choose.
        total_attempts = sum(self._attempts(stats, slot) for slot in offered)

        best: Optional[SlotName] = None
        best_key: Optional[tuple[float, int]] = None
        tied: list[SlotName] = []
        for slot in offered:
            score = self._score(stats, slot, total_attempts)
            # Negated position so that "larger key wins" also means
            # "earlier in SLOT_ORDER wins" on an exact score tie.
            key = (score, -_order_index(slot))
            if best_key is None or key > best_key:
                best, best_key, tied = slot, key, [slot]
            elif key == best_key:
                tied.append(slot)

        if len(tied) > 1:
            # Reachable only for two arms that tie on score AND share a
            # position - i.e. both absent from SLOT_ORDER. Distinct known
            # slots never get here, which is the point: the RNG is the
            # backstop, not the tie-break.
            return self._rng.choice(tuple(tied))
        assert best is not None  # `offered` is non-empty, checked above
        return best

    # -- scoring -------------------------------------------------------

    def _score(
        self, stats: Mapping[str, Any], slot: SlotName, total_attempts: int
    ) -> float:
        """Expected improvement per 1k tokens, plus the UCB exploration bonus."""
        attempts = self._attempts(stats, slot)
        if attempts <= 0:
            # Unreachable from select_slot (untried arms are returned
            # before scoring) but guarded anyway: every division below
            # would be by zero, and a policy is the wrong place to learn
            # that from a traceback.
            return 0.0

        entry = stats.get(slot) or {}
        deltas = entry.get("deltas") or ()
        improvements = [max(float(d), 0.0) for d in deltas]
        value = sum(improvements) / len(improvements) if improvements else 0.0

        mean_tokens = float(entry.get("total_tokens") or 0) / attempts
        cost = max(mean_tokens, self.cost_floor_tokens)
        exploit = (value / cost) * TOKENS_PER_UNIT

        # ln(1) is 0, so a single total pull yields no bonus rather than a
        # negative one; ln of anything smaller is undefined, and reaching
        # here with total_attempts < 1 would mean `attempts` lied.
        explore = 0.0
        if self.exploration_c > 0.0 and total_attempts > 1:
            explore = self.exploration_c * math.sqrt(
                math.log(total_attempts) / attempts
            )
        return exploit + explore

    # -- card reading --------------------------------------------------

    @staticmethod
    def _slot_stats(state_card: Mapping[str, Any]) -> Mapping[str, Any]:
        """The card's `slot_stats`, or an empty mapping.

        Absent or malformed is read as "no data", never as an error. A
        policy is handed the card by the Controller and has no way to fix a
        bad one; refusing to choose would stop a run over a missing
        summary, whereas treating every arm as untried degrades to the
        deterministic sweep this class starts with anyway.
        """
        stats = state_card.get("slot_stats")
        return stats if isinstance(stats, Mapping) else {}

    @staticmethod
    def _attempts(stats: Mapping[str, Any], slot: SlotName) -> int:
        """How many times this arm was pulled. Absent slot means zero.

        `slot_stats` omits untried slots entirely rather than zero-filling
        them, so an absent key and a zero-attempt entry mean the same thing
        here and are collapsed deliberately.
        """
        entry = stats.get(slot)
        if not isinstance(entry, Mapping):
            return 0
        try:
            return int(entry.get("attempts") or 0)
        except (TypeError, ValueError):
            return 0
