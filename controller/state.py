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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, NamedTuple, Optional

from contracts import (
    Budget,
    BudgetCounter,
    BUDGET_COUNTER_ORDER,
    CandidateResult,
    PipelineConfig,
    SLOT_ORDER,
    SlotName,
)
from controller import convergence

__all__ = [
    "STAGE_ORDER",
    "STAGE_SLOTS",
    "STATE_CARD_HISTORY",
    "HistoryEntry",
    "RunState",
    "SlotStats",
    "Stage",
    "build_state_card",
    "next_stage",
    "slot_stats",
    "slot_stats_as_json",
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


_STRUCTURAL_SLOTS: tuple[SlotName, ...] = (
    "features",
    "weighting",
    "model",
    "objective",
)
"""The four slots the organizers' published FM baseline leaves on the table.

It consumes five raw categorical fields with no engineered features
("features"), weights every interaction equally ("weighting"), fits a
factorization machine ("model"), and optimizes plain logloss
("objective"). Those are the four places a genuinely different method can
go, and the baseline sits ~30% of the way to the oracle ceiling largely
because it takes the default in each of them.

`data_view` and `calibration` are deliberately absent from the structural
stages: the first fixes what data is even in scope (changing it mid-search
would make earlier candidates incomparable), and the second is a monotone
post-hoc transform - a refinement of a good model, not a route to one.
Both are still reachable in STAGE_3_TUNE.
"""


STAGE_SLOTS: dict[Stage, tuple[SlotName, ...]] = {
    Stage.INIT: (),
    Stage.REPRODUCE_BASELINE: (),
    Stage.STAGE_1_STRUCTURAL: _STRUCTURAL_SLOTS,
    # The same four. STAGE_2_COMBINE does not open new ground; it
    # recombines structural changes that were accepted in stage one, so
    # its arms are exactly stage one's arms.
    Stage.STAGE_2_COMBINE: _STRUCTURAL_SLOTS,
    # Everything. Tuning applies anywhere - there are hyperparameters in
    # the data view and the calibrator as much as in the model - so the
    # last stage is the only one with no structural restriction.
    Stage.STAGE_3_TUNE: tuple(SLOT_ORDER),
    Stage.FINALIZE: (),
    Stage.DONE: (),
}
"""Which slots each stage is allowed to attack. Empty means "proposes nothing".

WHY THE STAGE CONSTRAINS THE POLICY, RATHER THAN THE POLICY OVERRIDING
THE STAGE
---------------------------------------------------------------------
The stage ordering is not a filing system, it is protection against a
specific failure mode, and it is documented at the top of this module:
the organizers' rule ends the run after three consecutive iterations that
improve the primary by less than epsilon = 0.002, and tuning reliably
produces gains under epsilon. A run that tunes early converges itself to
a halt before it has tried anything structural.

A search policy optimizing observed reward per token would walk straight
into that. Hyperparameter moves are cheap and land small positive deltas,
so early on they look like the best arms available - and picking them is
precisely what forfeits the run. The policy cannot see this, because the
cost of an early tuning move is not paid by that move; it is paid later,
by the structural moves that never get attempted.

So the constraint is structural rather than advisory. The Controller
intersects STAGE_SLOTS[stage] with the un-blocked slots and offers the
policy only that set, then validates what comes back. A policy is free to
be as greedy as it likes inside a stage; it is not free to spend stage one
on hyperparameters.

The empty entries are entries, not omissions. INIT and FINALIZE mark
boundaries, REPRODUCE_BASELINE evaluates a fixed published config, and
DONE is terminal - none of the four propose anything. Listing them
explicitly means a Stage added later fails the import-time guard below
rather than raising KeyError deep inside the first attempt that reaches it.
"""


# Import-time guards, in the same spirit as controller.py's BASELINE_SLOTS
# check. A raise rather than an assert, because `python -O` strips asserts
# and a stage with no entry would otherwise surface as a KeyError several
# layers into a run.
if set(STAGE_SLOTS) != set(Stage):
    raise RuntimeError(
        "STAGE_SLOTS must have an entry for every Stage; missing "
        f"{set(Stage) - set(STAGE_SLOTS)}"
    )
_unknown_slots = {
    slot for slots in STAGE_SLOTS.values() for slot in slots
} - set(SLOT_ORDER)
if _unknown_slots:
    raise RuntimeError(
        f"STAGE_SLOTS names slots absent from contracts.SLOT_ORDER: {_unknown_slots}"
    )


class HistoryEntry(NamedTuple):
    """One attempted candidate, as the state card summarises it.

    A NamedTuple so the first three fields are still genuinely a
    `(config_id, primary, accepted)` triple — indexable and
    JSON-serialisable as an array — while remaining readable at the call
    site. `primary` is None for a candidate that failed before producing
    scores.

    WHY THE VERDICT FIELDS ARE RETAINED HERE: the gate's `delta` is a
    *paired* per-seed difference, and it is the only measurement that
    properly accounts for sigma (~0.0008) sitting a factor of 2.5 below
    epsilon (0.002). It used to be computed, logged into the DECISION
    event, and then thrown away — which left convergence with nothing but
    differenced absolute primaries, whose noise is sqrt(2)*sigma ~= 0.0011,
    over half of epsilon. A rule built on those alone fires on noise.
    Keeping delta/ci95/significant is what makes the noise-aware internal
    rule possible at all.

    All three are None where no gate ruling exists: the baseline adoption
    (nothing to compare against) and any failed or unrealized candidate.
    """

    config_id: str
    primary: Optional[float]
    accepted: bool
    delta: Optional[float] = None
    ci95: Optional[tuple[float, float]] = None
    significant: Optional[bool] = None

    target_slot: Optional[SlotName] = None
    """Which slot this attempt attacked - the arm the policy pulled.

    None only where no slot was chosen: the REPRODUCE_BASELINE evaluation,
    which runs a fixed published config that no policy selected. Every
    attempt in a search stage records a slot, including the ones that
    failed, were rejected, or never got realized.

    Without this the history is a flat list of config_ids and there is no
    way to ask "how has `weighting` been doing" after the fact - the arm
    that produced each outcome is simply not written down anywhere.
    """

    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    tokens: int = 0
    """Cost of this ONE attempt, retained per attempt.

    WHY THIS IS KEPT HERE WHEN `with_spend` ALREADY CHARGES THE BUDGET.
    The two are not redundant, they answer different questions. The Budget
    counters answer "how much has the run spent in total" - they are a
    running sum, and a sum cannot be decomposed after the fact. A per-slot
    policy has to answer "how much has `model` cost me, and what did I get
    for it", which means attributing every charge to the arm that incurred
    it. There is exactly one moment at which that attribution is known -
    when the attempt completes - and if it is not written down then, it is
    gone. Reconstructing it later from the aggregate counters is not hard,
    it is impossible.

    FAILED AND UNREALIZED ATTEMPTS CARRY THEIR COST TOO, and this is the
    part worth being careful about. A candidate that OOM'd after twenty
    minutes burned twenty minutes. If its entry recorded zero, a
    cost-aware policy would compute the average cost of the slot that
    produced it as *lower* than a slot whose candidates all succeed - so
    the more fragile a slot is, the cheaper it would look, and the harder
    the policy would push on it. That is exactly backwards, and it is the
    same reasoning contracts.CandidateResult uses to populate its own cost
    fields on failure.

    (One honest gap: a hypothesis the realizer could not turn into code
    never reached the executor, so its recorded cost is genuinely zero
    here. The realizer's own token spend is real but no port reports it
    yet - RealizerPort returns a SlotConfig and nothing else. When that
    changes, this is where the number goes.)

    `tokens` is the single summed figure (in + out) rather than the pair,
    matching what `with_spend` charges the budget and what the EVAL_RESULT
    payload logs, so the per-slot totals and the run total are commensurable.
    """

    executor_failed: bool = False
    """True only for an attempt the EXECUTOR ran and reported FAILED.

    THE CIRCUIT BREAKER'S ONLY INPUT, and the reason it is a stored field
    rather than something derived at read time.

    Four distinct outcomes arrive here as `accepted=False`, and they are
    not the same event:

      (a) the generator proposed a slot other than the one it was asked
          for                                    -> MISPROPOSED_CONFIG_ID
      (b) the realizer could not turn a hypothesis into code
                                                 -> UNREALIZED_CONFIG_ID
      (c) the executor ran the candidate and it failed  -> THIS FLAG
      (d) the gate ruled against it               -> a clean rejection

    Only (c) is evidence that spending more evaluations on this slot is
    wasteful. (a) and (b) are contract breaches by another workstream's
    collaborator - blocking `weighting` because W4's generator kept naming
    the wrong slot would delete a good arm over somebody else's bug - and
    (d) is a working arm honestly reporting no improvement, which is the
    signal the bandit exists to consume rather than a reason to stop.

    WHY A FIELD RATHER THAN A config_id COMPARISON. The alternative was to
    recognise (a) and (b) by testing `config_id` against
    MISPROPOSED_CONFIG_ID and UNREALIZED_CONFIG_ID. Two problems. First,
    those constants live in controller/controller.py, which imports THIS
    module - reading them here would close an import cycle, so state.py
    would have to carry its own copy of the two string literals and the
    circuit breaker would silently start blocking on contract breaches the
    day either constant was edited on one side only. Second, a sentinel
    comparison infers the outcome from a stand-in value chosen for a
    different purpose (giving `config_id` something printable), whereas the
    Controller knows the answer outright at the point it writes the entry.

    A bool rather than an enum because the question the breaker asks is
    exactly binary. Defaulting to False keeps every existing call site
    correct without change: only the executor-failure path passes True.
    """


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
        """Move to `stage`. Callers must source it from `next_stage`.

        BLOCKED SLOTS ARE CLEARED HERE. Blocking is stage-scoped, and this
        is the transition that scopes it.

        The circuit breaker blocks a slot so the agent stops re-debugging
        one bad idea for the rest of the stage it went wrong in. That is a
        budget argument, not a verdict about the slot: the evidence for
        blocking is k consecutive executor failures, and an executor
        failure can just as easily be an OOM on a busy machine, a transient
        library fault or one badly-realized candidate as it can be a slot
        that is genuinely a dead end. Carrying a block forward for the rest
        of the run would let a transient fault permanently delete an arm -
        with no unblock path anywhere, since `with_blocked_slot` only ever
        adds - and the later stages are exactly where a slot deserves
        another look: STAGE_3_TUNE attacks a different question with the
        same arms, on top of an incumbent that has since moved.

        So a block costs the slot the remainder of one stage and no more.
        The full history survives regardless, so `slot_stats` still shows
        the failures and the bandit still scores the arm accordingly - a
        slot that failed three times in stage one comes back into stage two
        eligible but with a poor record, which is the right amount of
        memory to carry.
        """
        return replace(self, stage=stage, blocked_slots=frozenset())

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
        self,
        config_id: str,
        primary: Optional[float],
        accepted: bool,
        delta: Optional[float] = None,
        ci95: Optional[tuple[float, float]] = None,
        significant: Optional[bool] = None,
        target_slot: Optional[SlotName] = None,
        wall_seconds: float = 0.0,
        gpu_seconds: float = 0.0,
        tokens: int = 0,
        executor_failed: bool = False,
    ) -> RunState:
        """Record an attempt's result: history always, `iteration` only on
        acceptance.

        The verdict fields default to None so the baseline adoption and
        failed/unrealized attempts — none of which have a gate ruling —
        record honestly rather than with a fabricated zero.

        `target_slot` and the three cost figures default the same way and
        for the same reason: the baseline has no slot and an attempt that
        never reached the executor has no measured spend, so both record
        their absence rather than a made-up value. Callers that DO have
        those numbers must pass them for every outcome, acceptance and
        failure alike - see HistoryEntry's field docs for why a zero on a
        failed attempt would actively mislead the search policy.

        `executor_failed` defaults to False so only the one path that means
        it - the executor ran this candidate and reported FAILED - has to
        say so. See HistoryEntry.executor_failed for why the other three
        non-accepted outcomes must not set it.

        Still pure: derives a new RunState and touches nothing on this one.
        """
        return replace(
            self,
            iteration=self.iteration + (1 if accepted else 0),
            history=self.history
            + (
                HistoryEntry(
                    config_id=config_id,
                    primary=primary,
                    accepted=accepted,
                    delta=delta,
                    ci95=ci95,
                    significant=significant,
                    target_slot=target_slot,
                    wall_seconds=wall_seconds,
                    gpu_seconds=gpu_seconds,
                    tokens=tokens,
                    executor_failed=executor_failed,
                ),
            ),
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

    @property
    def committed_revisions(self) -> tuple[HistoryEntry, ...]:
        """Accepted attempts only, OLDEST FIRST.

        The window convergence reads. Oldest-first so that `[-N:]` is the
        most recent N revisions, matching how both rules are phrased ("the
        last N iterations") and how `_improvements` differences
        consecutive pairs.

        Rejected, failed and unrealized attempts are excluded by
        construction: an iteration is a committed revision, so a dead end
        is not one. Every entry here therefore has a non-None `primary`,
        which is what lets the convergence rules skip None-handling.
        """
        return tuple(entry for entry in self.history if entry.accepted)

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


@dataclass(frozen=True)
class SlotStats:
    """What one slot has cost and returned, aggregated over the whole run.

    One arm's record, in the vocabulary a bandit needs: how often it was
    pulled, how often that paid off, what it cost, and the distribution of
    what it returned.

    Frozen and plain-valued, like everything else in this module, so a
    snapshot can be logged into a journal payload and still mean what it
    meant when it was written.
    """

    attempts: int = 0
    """Every attempt on this slot, INCLUDING failures and unrealized
    hypotheses. This is the denominator of a cost-per-attempt figure, and
    a denominator that quietly dropped the expensive failures would flatter
    exactly the slots that deserve it least."""

    accepted: int = 0
    """Attempts the gate accepted. The numerator of the arm's hit rate."""

    total_tokens: int = 0
    total_wall_seconds: float = 0.0

    deltas: tuple[float, ...] = ()
    """Gate-measured deltas, oldest first, for attempts that got a ruling.

    Shorter than `attempts` whenever this slot produced a failure or an
    unrealized hypothesis - those have no delta, and this records that by
    being absent rather than by contributing a zero. A zero would read as
    "measured, and it did nothing", which is a different and much stronger
    claim than "never measured".
    """

    consecutive_failures: int = 0
    """Executor failures at the TAIL of this slot's attempts. The circuit
    breaker's counter.

    Counts only `HistoryEntry.executor_failed` entries - see that field for
    why a generator slot-mismatch, a realizer exhaustion and a clean gate
    rejection all leave this alone. Any other outcome on this slot resets
    it to zero, so it measures a current run of failures rather than a
    lifetime total: a slot that failed twice, was fixed, and has since
    worked is not one attempt away from being blocked.

    WHY IT IS FOLDED HERE RATHER THAN DERIVED FROM THE OTHER FIELDS. It
    cannot be. Every other field on this class is order-free - a count or a
    sum - and could be recomputed from an unordered bag of entries.
    "Consecutive" is a statement about the ORDER of the attempts, and
    `slot_stats` is the only place that still sees it: by the time the fold
    returns, the sequence is gone. Deriving it afterwards from `attempts`
    and `accepted` is not merely awkward, it is impossible.
    """


def slot_stats(state: RunState) -> dict[SlotName, SlotStats]:
    """Per-slot aggregation over `state.history`. Pure; a fold, not a cache.

    Recomputed from history on each call rather than maintained
    incrementally on RunState. History is append-only and short (bounded by
    the run's node count, which is tens), so the fold is free, and keeping
    a running tally on a frozen state object would mean two representations
    of the same fact that can disagree - the class of bug this module's
    immutability is meant to make impossible.

    Entries with `target_slot is None` are skipped. That is the baseline
    evaluation, which no policy chose; charging it to an arm would credit
    or blame a slot for a decision it had nothing to do with.

    Only slots that have actually been attempted appear. An absent key
    means "no data", which a policy must handle anyway - an untried arm and
    an arm tried once to no effect are different situations, and a
    zero-filled entry would erase the difference.

    THE CONSUMER IS THE NEXT PR. The cost-aware bandit reads this to score
    arms on delta-per-1k-tokens; UniformPolicy ignores it entirely, which
    is fine and expected. Building the bookkeeping alongside the seam is
    what keeps that PR to the policy itself instead of bundling a state
    refactor with it.
    """
    stats: dict[SlotName, SlotStats] = {}
    for entry in state.history:
        slot = entry.target_slot
        if slot is None:
            continue
        current = stats.get(slot, SlotStats())
        stats[slot] = SlotStats(
            attempts=current.attempts + 1,
            accepted=current.accepted + (1 if entry.accepted else 0),
            total_tokens=current.total_tokens + entry.tokens,
            total_wall_seconds=current.total_wall_seconds + entry.wall_seconds,
            deltas=(
                current.deltas
                if entry.delta is None
                else current.deltas + (entry.delta,)
            ),
            # Folded in the pass, because this is the only place the
            # ORDER of the attempts is still visible. Incremented on an
            # executor failure, reset to zero by literally anything else
            # on this slot.
            consecutive_failures=(
                current.consecutive_failures + 1 if entry.executor_failed else 0
            ),
        )
    return stats


def slot_stats_as_json(stats: Mapping[SlotName, SlotStats]) -> dict[str, Any]:
    """`slot_stats` output in the shape it will have after a JSON hop.

    The same job `_history_as_json` does for one HistoryEntry, and for the
    same reason: `deltas` is a tuple on the way in and a list on the way
    out, since json has only one sequence type. Rendering here means what
    the Controller hands the policy in-process is byte-identical to what a
    collaborator on the far side of a serialisation boundary receives, and
    keeps `json.loads(json.dumps(card)) == card` true.

    Public rather than underscored because it is the one place that fixes
    the wire shape a policy reads. A policy hand-rolling its own dict of
    per-slot numbers in a test would be asserting against a shape nothing
    guarantees.

    Slots come out sorted by name so two runs with the same history
    produce the same key order in the journal and in any diff of it.
    """
    return {
        slot: {
            "attempts": entry.attempts,
            "accepted": entry.accepted,
            "total_tokens": entry.total_tokens,
            "total_wall_seconds": entry.total_wall_seconds,
            "deltas": list(entry.deltas),
            "consecutive_failures": entry.consecutive_failures,
        }
        for slot, entry in sorted(stats.items())
    }


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

    THE SELECTED SLOT IS NOT A KEY HERE. DELIBERATE - READ THIS.
    ------------------------------------------------------------
    Slot selection moved to the Controller, and the obvious-looking move
    was to announce the choice in this card. It is not done, for two
    reasons.

    First, ordering. The policy consumes this card in order to *make* the
    choice, so at the moment the card is built there is no choice to put
    in it. Adding the key would mean building the card twice - once for
    the policy without it, once for the generator with it - and the two
    would then differ, so "what did the run look like when the slot was
    picked" and "what did the generator see" would stop being the same
    question. One card per attempt, handed to both, keeps them honest.

    Second, and more important: a slot named in this dict would be exactly
    the silent, ignorable directive that GeneratorPort.propose's docstring
    explains at length is the wrong shape for this constraint. The slot is
    a parameter on `propose`, where it can be checked against what comes
    back. Putting a second copy in the card would create a channel through
    which the constraint could be honoured or not with no way to tell.

    *** FLAGGED: ONE NEW TOP-LEVEL KEY, `slot_stats`. W4 READS THIS. ***
    ------------------------------------------------------------------
    The card now carries `slot_stats` - the whole of `slot_stats(state)`,
    rendered JSON-safe. It is an ADDITION, not a rename: all ten previous
    keys are present with identical meaning, so nothing coded against the
    old shape breaks. Announced loudly anyway, because this file's contract
    is "additions are cheap, renames are a cross-team change" and the
    cheapness of an addition is not a licence to make one quietly.

    WHY IT GOES ON THE CARD, WHEN THE PREFERENCE IS TO LEAVE THE CARD
    ALONE. CostAwareBanditPolicy scores arms on realized improvement per
    token, so it needs per-slot history. The card is the only channel
    PolicyPort has: `select_slot(state_card, candidate_slots)`, and the
    other two options are both worse.

      - Widening PolicyPort to take a RunState would hand every policy the
        entire mutable-by-derivation world and make the port a
        controller-internal type, which is the seam we deliberately kept
        narrow so a policy stays testable as a pure function.
      - A side channel - an `observe(stats)` hook the Controller calls
        before `select_slot` - would create a SECOND account of what the
        run looks like, alongside the card. Controller._attempt goes out of
        its way to build one card and hand it to both the policy and the
        generator precisely so that "what did the run look like when the
        slot was picked" and "what did the generator know" cannot become
        two questions with two answers. A second channel reintroduces
        exactly that, and it would be the channel carrying the numbers the
        search decision actually turned on.

    So one card, one derivation, both readers. It is also genuinely useful
    to a generator, which can now see which arms have been paying off
    rather than re-deriving that from five truncated history entries.

    Note the asymmetry with the selected slot below: that is excluded
    because it does not EXIST when the card is built. Per-slot history does
    exist, and is the same for both readers.

    THE SELECTED SLOT IS STILL NOT A KEY HERE.

    ONE NESTED CHANGE, FLAGGED BECAUSE W4 READS IT: each entry in
    `recent_history` now carries four additional keys - `target_slot`
    (str | None), `wall_seconds` (float), `gpu_seconds` (float) and
    `tokens` (int) - because HistoryEntry gained those fields and this
    renders whatever HistoryEntry holds. Additions, not renames: every key
    that was there before is still there with the same meaning, so nothing
    reading the old shape breaks. They are genuinely useful to a generator,
    which can now see which slots recent attempts went to and what they
    cost rather than inferring it from config_ids it cannot decode.

    Keys:
        run_id              str
        stage               str  (Stage value, not the enum)
        iteration           int  committed revisions
        node                int  evaluations attempted
        incumbent_config_id str | None
        incumbent_primary   float | None
        recent_history      list of {config_id, primary, accepted, delta,
                             ci95, significant, target_slot, wall_seconds,
                             gpu_seconds, tokens, executor_failed}
        slot_stats          {slot name: {attempts, accepted, total_tokens,
                             total_wall_seconds, deltas, consecutive_failures}}
                             — attempted slots only; an absent slot means
                             "no data", never "zero"
        blocked_slots       sorted list of slot names
        budget_remaining    {counter name: float | None}
        convergence         {iterations_considered, epsilon, n_required,
                             flat_streak, converged, by_rule}
    """
    committed = state.committed_revisions
    status = convergence.assess(committed)

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
            _history_as_json(entry) for entry in state.history[-STATE_CARD_HISTORY:]
        ],
        # NOT truncated, unlike recent_history. This is an aggregate over
        # six slots at most, so it is bounded by the slot vocabulary rather
        # than by the run length, and truncating it would defeat the point:
        # a bandit scoring an arm on a five-attempt window would forget
        # everything it learned in the stage before.
        "slot_stats": slot_stats_as_json(slot_stats(state)),
        "blocked_slots": sorted(state.blocked_slots),
        "budget_remaining": {
            name: _remaining_or_none(getattr(state.budget, name))
            for name in BUDGET_COUNTER_ORDER
        },
        # How close the run is to stopping. Actionable context for the
        # generator: "one more flat iteration ends this" is a reason to
        # propose something bolder rather than another safe tweak.
        "convergence": {
            "iterations_considered": status.iterations_considered,
            "epsilon": status.epsilon,
            "n_required": status.n_required,
            "flat_streak": convergence.flat_streak(committed),
            "converged": status.converged,
            "by_rule": status.by_rule,
        },
    }


def _history_as_json(entry: HistoryEntry) -> dict[str, Any]:
    """One history entry, in the shape it will have after a JSON hop.

    `ci95` is a tuple on the way in and a list on the way out — json has
    only one sequence type. Converting here means what the Controller hands
    the generator is byte-identical to what a generator on the far side of
    a serialisation boundary receives, so a card cannot pass a
    round-trip assertion in-process and then arrive subtly different in
    production. It also keeps `json.loads(json.dumps(card)) == card` true,
    which is the cheapest possible check that this interface is portable.
    """
    fields = entry._asdict()
    if fields["ci95"] is not None:
        fields["ci95"] = list(fields["ci95"])
    return fields


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
