"""The Controller's stage vocabulary and its immutable run state.

WHY THIS IS NOT IN contracts.py
-------------------------------
contracts.py is the frozen, cross-team file: the four packages agree on it
and changing it is a coordination event. The stage list is the opposite of
that — it is the Controller's own progression, and it *will* change as the
search policy matures (splitting a stage, adding a warm-up, reordering).
Putting it in contracts would mean every reorder of W2's internal plan
forced a shared-file change on three other people.

Nothing outside controller/ needs the enum anyway. `contracts.EventKind`
already carries STAGE_CHANGE, and every stage lands in the journal as the
plain string in its payload — so W3's report renderer reads
`payload["stage"]`, a string, and never imports this module. That keeps
the dependency arrow pointing out of controller/ and never in.

WHY THE STAGES ARE IN THIS ORDER
--------------------------------
Structural moves first, hyperparameter tuning last, and that ordering is
load-bearing rather than aesthetic. The organizers' convergence rule ends
the run after three consecutive iterations that improve the primary metric
by less than epsilon = 0.002. Tuning reliably produces small gains. So a
run that tunes early spends its first iterations making sub-epsilon
progress and *converges itself to a halt* before it ever tries the
high-variance structural changes where the real headroom is (the published
baseline sits ~30% of the way to the oracle ceiling — the remaining 0.27 is
not going to come from a learning-rate sweep).

Hence: reproduce the baseline, then swing hard (STAGE_1_STRUCTURAL), then
combine what worked (STAGE_2_COMBINE), and only then tune
(STAGE_3_TUNE), where a run of small gains is a legitimate signal that
there is nothing left rather than a self-inflicted early stop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, NamedTuple, Optional

from contracts import (
    Budget,
    BudgetCounter,
    BUDGET_COUNTER_ORDER,
    CandidateResult,
    PipelineConfig,
    SlotName,
)

__all__ = [
    "STAGE_CARD_HISTORY",
    "STAGE_ORDER",
    "HistoryEntry",
    "RunState",
    "Stage",
    "build_state_card",
    "next_stage",
]


class Stage(str, Enum):
    """Where the run currently is in its plan.

    A str-valued Enum, matching contracts.EventKind's style, so the value
    drops straight into a JSON journal payload with no custom encoder and
    round-trips via `Stage(value)`.
    """

    INIT = "init"
    REPRODUCE_BASELINE = "reproduce_baseline"
    STAGE_1_STRUCTURAL = "stage_1_structural"
    STAGE_2_COMBINE = "stage_2_combine"
    STAGE_3_TUNE = "stage_3_tune"
    FINALIZE = "finalize"
    DONE = "done"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.INIT,
    Stage.REPRODUCE_BASELINE,
    Stage.STAGE_1_STRUCTURAL,
    Stage.STAGE_2_COMBINE,
    Stage.STAGE_3_TUNE,
    Stage.FINALIZE,
)
"""The legal linear progression, working stages only.

DONE is deliberately absent: it is the terminal state the Controller lands
in after FINALIZE, not a stage that does work. `next_stage` walks
STAGE_ORDER + DONE, so DONE is reachable but never iterated over as a
stage with candidates to attempt.
"""

_PROGRESSION: tuple[Stage, ...] = STAGE_ORDER + (Stage.DONE,)


def next_stage(current: Stage) -> Optional[Stage]:
    """The stage that legally follows `current`, or None at the end.

    This is the *only* way the Controller is allowed to advance, which is
    what makes illegal transitions inexpressible: there is no
    `state.stage = Stage.FINALIZE` shortcut that skips the middle, because
    RunState is frozen and this helper is the sole source of a successor.
    A skipped stage would be invisible in the journal — the STAGE_CHANGE
    events would simply not mention it — so the guard has to be structural
    rather than a reviewer noticing.
    """
    index = _PROGRESSION.index(current)
    if index + 1 >= len(_PROGRESSION):
        return None
    return _PROGRESSION[index + 1]


class HistoryEntry(NamedTuple):
    """One attempted candidate, as the state card summarises it.

    A NamedTuple so it is genuinely a `(config_id, primary, accepted)`
    triple — indexable and JSON-serialisable as an array — while still
    being readable at the call site. `primary` is None for a candidate
    that failed before producing scores.
    """

    config_id: str
    primary: Optional[float]
    accepted: bool


STATE_CARD_HISTORY = 5
"""How many recent attempts the state card carries.

Bounded because the card is handed to an LLM: an unbounded history would
grow the prompt without bound over a long run and would eventually crowd
out the parts that matter. RunState keeps the *full* history for the
audit trail; only the card is truncated.
"""


@dataclass(frozen=True)
class RunState:
    """Everything the loop knows, at one instant.

    Frozen, and derived only through the `with_*` methods below.

    WHY IMMUTABLE: the journal is the audit trail, and its whole value is
    that an event records what was true when it was written. With a
    mutable state object every event would hold a reference to one
    ever-changing thing, and "what was the incumbent when this DECISION
    was logged?" would be unanswerable after the fact — you would read
    back the state as it ended, not as it was. Deriving a new RunState per
    transition makes each event's context permanently pinned, and makes
    the whole loop replayable by folding the same transitions again.

    `iteration` and `node` are both tracked, and they are not the same
    number: `iteration` counts committed revisions (candidates that were
    accepted and became the incumbent) while `node` counts every
    evaluation attempted, accepted or not. contracts.JournalEvent logs
    both for exactly this reason — the convergence rule's "iteration" is
    still undefined between the two, so the log records each and the
    question can be settled later without re-running anything.
    """

    run_id: str
    stage: Stage
    iteration: int = 0
    node: int = 0
    incumbent: Optional[CandidateResult] = None
    incumbent_config: Optional[PipelineConfig] = None
    budget: Budget = Budget()
    blocked_slots: frozenset[SlotName] = frozenset()
    history: tuple[HistoryEntry, ...] = ()

    # -- pure derivations ---------------------------------------------

    def with_stage(self, stage: Stage) -> RunState:
        """Move to `stage`. Callers must source it from `next_stage`."""
        return replace(self, stage=stage)

    def with_node_started(self) -> RunState:
        """Increment `node` at the *start* of an attempt.

        Deliberately before the work rather than after, so every event
        emitted for a candidate carries that candidate's own node number.
        Incrementing afterwards would leave the whole attempt logged under
        the previous candidate's number, which is exactly the kind of
        off-by-one that makes a log untrustworthy.
        """
        return replace(self, node=self.node + 1)

    def with_incumbent(
        self, result: CandidateResult, config: PipelineConfig
    ) -> RunState:
        """Adopt a candidate as the new incumbent. Touches no counters."""
        return replace(self, incumbent=result, incumbent_config=config)

    def with_outcome(
        self, config_id: str, primary: Optional[float], accepted: bool
    ) -> RunState:
        """Record an attempt's result: history always, `iteration` only on
        acceptance."""
        return replace(
            self,
            iteration=self.iteration + (1 if accepted else 0),
            history=self.history + (HistoryEntry(config_id, primary, accepted),),
        )

    def with_spend(self, result: CandidateResult) -> RunState:
        """Charge one evaluation's cost against the budget.

        Cost is taken from the CandidateResult even when it FAILED — a
        candidate that died halfway still burned the wall-clock and tokens
        it took to get there, and zeroing that out would understate the
        run's true cost, which is itself a graded axis.
        """
        return replace(
            self,
            budget=Budget(
                wall_seconds=_charge(self.budget.wall_seconds, result.wall_seconds),
                tokens=_charge(
                    self.budget.tokens, float(result.tokens_in + result.tokens_out)
                ),
                evaluations=_charge(self.budget.evaluations, 1.0),
                gpu_seconds=_charge(self.budget.gpu_seconds, result.gpu_seconds),
            ),
        )

    def with_blocked_slot(self, slot: SlotName) -> RunState:
        """Mark a slot as off-limits for further proposals.

        Not called by the current Controller — slot blocking arrives with
        the repair policy in a later PR. It lives here now so the state
        card can already report `blocked_slots`, giving W4 a stable key to
        code against rather than one that appears later.
        """
        return replace(self, blocked_slots=self.blocked_slots | {slot})


def _charge(counter: BudgetCounter, amount: float) -> BudgetCounter:
    """One counter, advanced. Pure; the original is untouched."""
    return replace(counter, consumed=counter.consumed + amount)


def build_state_card(state: RunState) -> dict[str, Any]:
    """The compact summary handed to `GeneratorPort.propose`.

    THIS IS AN INTERFACE, NOT AN INTERNAL DETAIL. It is the *only* input
    W4's hypothesis generator receives, so its key set is a contract in
    spirit even though it cannot live in contracts.py (that file holds no
    behaviour, and this is a derivation). Renaming a key here silently
    breaks a component in another package with no type error to catch it.
    Treat additions as cheap and renames as a cross-team change.

    Deliberately excluded: raw conversation history, and whole
    CandidateResult objects. The generator needs to know where the run
    stands, not to re-derive it — and everything here must survive
    `json.dumps`, which a CandidateResult (nested Metrics, enum members)
    does not.

    Keys:
        run_id              str
        stage               str  (Stage value, not the enum)
        iteration           int  committed revisions
        node                int  evaluations attempted
        incumbent_config_id str | None
        incumbent_primary   float | None
        recent_history      list of {config_id, primary, accepted}
        blocked_slots       sorted list of slot names
        budget_remaining    {counter name: float | None}
    """
    incumbent_primary: Optional[float] = None
    if state.incumbent is not None and state.incumbent.val:
        incumbent_primary = sum(
            m.primary for m in state.incumbent.val.values()
        ) / len(state.incumbent.val)

    return {
        "run_id": state.run_id,
        "stage": state.stage.value,
        "iteration": state.iteration,
        "node": state.node,
        "incumbent_config_id": (
            state.incumbent_config.config_id
            if state.incumbent_config is not None
            else None
        ),
        "incumbent_primary": incumbent_primary,
        "recent_history": [
            entry._asdict() for entry in state.history[-STATE_CARD_HISTORY:]
        ],
        "blocked_slots": sorted(state.blocked_slots),
        "budget_remaining": {
            name: _remaining_or_none(getattr(state.budget, name))
            for name in BUDGET_COUNTER_ORDER
        },
    }


def _remaining_or_none(counter: BudgetCounter) -> Optional[float]:
    """Headroom, or None when the counter is unmetered.

    An unmetered counter has `limit = inf`, and `inf` is not valid JSON
    (RFC 8259 has no infinity literal). Python's json module emits a
    non-standard `Infinity` token by default, which a strict parser on the
    other side of this interface would reject. None says "no limit" in a
    form every JSON parser accepts.
    """
    if counter.limit == float("inf"):
        return None
    return counter.remaining
