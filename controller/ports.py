"""Interface seams the Controller calls out through.

WHY THESE LIVE HERE AND NOT IN contracts.py
-------------------------------------------
contracts.py opens with a HARD RULE: "contracts carry data, never
behaviour." Everything in it is a frozen dataclass, an enum, or a pure
helper that derives a value from fields already on the object. A Protocol
describes *behaviour* — the calls one component makes on another — so by
that rule it cannot go there, and this module is where it goes instead.

There is a second, practical reason, and it is the one that matters day
to day. These Protocols describe what the **Controller needs**, not what
the whole team has agreed to. Keeping them inside controller/ means
adjusting one is a local edit in W2 rather than a cross-team event that
obliges three other people to re-read a file we have all agreed is
frozen. The *data* crossing these seams is still contracts types, so the
shared vocabulary is unchanged — only the verbs live here.

STRUCTURAL, NOT NOMINAL
-----------------------
Nothing has to inherit from these. A class satisfies a Protocol by having
the right methods, so W1/W3/W4 can write their components without
importing controller/ at all — which is the point, since the dependency
arrow must never run from another package into mine. The Protocols exist
so the Controller can be type-checked and so its test doubles
(controller/fakes.py) are provably substitutable for the real thing.

`@runtime_checkable` is applied to every one so tests can assert
`isinstance(double, SomePort)`. Be aware that this only ever checks that
the named methods *exist* — it does not check signatures or types. It
catches a renamed or missing method, not a wrong argument list; a static
checker is what catches the rest.

NO CACHE PORT, DELIBERATELY
---------------------------
harness/cache.py is still `get(*args, **kwargs)` / `put(*args, **kwargs)`
with no real signature, and more importantly the Controller never touches
the cache directly — the executor does, underneath ExecutorPort.run.
Inventing a port for a collaborator we do not call would be fiction with
no consumer to keep it honest.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from contracts import (
    CandidateResult,
    HypothesisPayload,
    JournalEvent,
    PipelineConfig,
    SlotConfig,
    SlotName,
    Verdict,
)

__all__ = [
    "ExecutorPort",
    "GatePort",
    "GeneratorExhausted",
    "GeneratorPort",
    "JournalPort",
    "PolicyContractError",
    "PolicyPort",
    "PortExhausted",
    "RealizerExhausted",
    "RealizerPort",
]


# ---------------------------------------------------------------------------
# Port-level exceptions
#
# WHY THESE LIVE BESIDE THE PROTOCOLS: any implementation of a port may run
# out of work or refuse a request - a scripted double runs off the end of its
# script, a real generator decides it has nothing left worth trying, a
# realizer cannot turn a hypothesis into runnable code. The Controller has to
# catch that, and it must be able to do so WITHOUT importing from any
# particular implementation - least of all from the test doubles, which is
# exactly the dependency this hierarchy exists to remove.
#
# Declaring them next to the Protocols also completes the contract: a port is
# its method signatures AND the exceptions it is permitted to raise. An
# implementor reading this file sees both halves in one place.
# ---------------------------------------------------------------------------


class PortExhausted(RuntimeError):
    """Base: a port has no further work to offer for this request.

    Subclasses RuntimeError rather than Exception so that existing broad
    handlers behave as they did, and because that is what this is - a
    runtime condition of a collaborator, not a programming error in the
    Controller.

    Not raised directly. Catch this to mean "some port gave up"; catch a
    subclass to distinguish which one and how much it costs.
    """


class GeneratorExhausted(PortExhausted):
    """The generator has no further hypothesis to offer.

    A normal end to a run rather than a crash: the Controller finalises
    through its ordinary path, so the journal still receives FINALIZE and
    RUN_END and a replay can tell the run finished rather than died.
    """


class RealizerExhausted(PortExhausted):
    """The realizer cannot turn this hypothesis into a SlotConfig.

    Deliberately NOT fatal to the run. One hypothesis that cannot be
    realised is a localized failure, and absorbing exactly that is what
    the architecture is for - so the Controller logs it, counts the node
    and moves on to the next candidate.
    """


class PolicyContractError(RuntimeError):
    """A PolicyPort implementation returned a slot it was not offered.

    Deliberately NOT a PortExhausted. Every exception above means "a
    collaborator has nothing left to give", which is a runtime condition
    the Controller absorbs. This one means "our own code is wrong", and
    the two must not be catchable as one thing: a handler written to
    shrug off an exhausted generator must not also shrug off a policy bug.

    WHY THIS IS FATAL WHILE A MISBEHAVING GENERATOR IS NOT. The generator
    has an LLM behind it. An LLM that ignores an instruction is an
    expected operating condition, so the Controller logs it, discards the
    candidate and carries on. A policy is ordinary deterministic code we
    wrote; if it returns a slot outside `candidate_slots` there is a bug
    in the search policy itself, and every subsequent number the run
    produces would be attributed to the wrong arm. Failing loudly at the
    point of the bug beats producing a plausible-looking run built on it.

    Raised BY the Controller ABOUT a policy, rather than by the policy
    itself - but it is documented here because it is part of PolicyPort's
    contract, and an implementor reading this file needs to know that
    breaking that contract stops the run.
    """


@runtime_checkable
class ExecutorPort(Protocol):
    """Runs one candidate across seeds and reports what happened (W3)."""

    def run(self, config: PipelineConfig, seeds: Sequence[int]) -> CandidateResult:
        """Evaluate `config` once per seed in `seeds`.

        Must return a CandidateResult even when the candidate blows up —
        with `status=Status.FAILED` and a classified `error_class` —
        rather than raising. The Controller decides what to do about a
        failure (repair, block the slot, move on); an exception escaping
        here would take the whole run down with it and lose the journal,
        which is the one artifact that makes the run auditable.

        `seeds` is a Sequence rather than a set because the order is
        recorded and re-used: the noise gate pairs candidate and incumbent
        on matching seeds, so both sides must be evaluated on the same
        seeds, and a caller passing a list must get those exact keys back
        in `CandidateResult.val`.
        """
        ...


@runtime_checkable
class GatePort(Protocol):
    """Decides whether a candidate genuinely beats the incumbent (W1).

    KNOWN DISCREPANCY WITH harness/gate.py — DELIBERATE.
    As of this writing harness/gate.py exposes only
    `passes_gate(*args, **kwargs)`, which has no body and returns None.
    Nothing anywhere in the repo returns a `Verdict`. So this Protocol is
    **the shape the Controller needs**, not a description of a shape that
    exists: it is a standing request to W1, written down in code where it
    cannot be forgotten.

    It is deliberately NOT adapted to fit the current stub. Bending the
    port to match `passes_gate`'s untyped `*args` would erase the request
    and leave the Controller consuming a bare bool — which discards
    `delta` and `ci95`, the two numbers convergence tracking is built on.
    The mismatch is known, is tracked, and should be resolved by
    harness/gate.py moving to this shape.
    """

    def compare(self, candidate: CandidateResult, incumbent: CandidateResult) -> Verdict:
        """Paired per-seed comparison of candidate against incumbent.

        Paired, on matching seeds, because seed-to-seed noise on this
        benchmark (sigma ~0.0008 on the primary) is close to the
        acceptance threshold the organizers publish (epsilon = 0.002,
        about 2.5 sigma). Comparing two means throws away the pairing that
        makes a real effect of that size detectable at all.
        """
        ...


@runtime_checkable
class PolicyPort(Protocol):
    """Chooses WHICH SLOT the next candidate attacks. The search policy.

    WHY THIS IS A SEAM OF ITS OWN, AND WHY IT IS OURS AND NOT W4's
    --------------------------------------------------------------
    Picking the slot is a search decision - an explore/exploit tradeoff
    over arms whose costs and payoffs the Controller is the only component
    that can measure. It used to sit with the generator by default: the
    Controller asked for a hypothesis, read `payload.target_slot` back and
    spliced into whatever the LLM had named. That handed the search policy
    to a model that cannot see the budget, cannot see which slots have
    already failed, and has no memory of what it tried last time.

    The generator's job is to propose a *method* and cite it; deciding
    where in the pipeline to spend the next evaluation is ours.

    IMPLEMENTATIONS LIVE IN controller/policy.py, NOT IN fakes.py.
    UniformPolicy and FixedOrderPolicy are real components - the first is
    the ablation baseline the cost-aware bandit will be measured against,
    the second is the no-randomness default. Neither is a test double.

    DETERMINISM IS PART OF THE CONTRACT
    -----------------------------------
    An implementation must be deterministic given its own seed and its own
    call history, and must draw randomness from an instance-owned
    `random.Random` rather than the global `random` module. Search
    behaviour has to be reproducible: an ablation that compares two
    policies is meaningless if replaying either one produces a different
    sequence of arms, and a global RNG makes the sequence depend on
    whatever else in the process happened to draw from it.
    """

    def select_slot(
        self, state_card: Mapping[str, Any], candidate_slots: Sequence[SlotName]
    ) -> SlotName:
        """Pick one slot from `candidate_slots` for this attempt.

        MUST return a member of `candidate_slots`. That set is already
        filtered - it is this stage's permitted slots minus the ones the
        run has blocked - so anything outside it is either a stage
        violation or a re-attempt on a slot known to be broken. The
        Controller validates the return value and raises
        `PolicyContractError` on a miss rather than trusting it; see that
        exception for why this one is fatal when a misbehaving generator
        is not.

        `candidate_slots` is never empty. The Controller ends the stage
        before calling a policy with nothing to choose from, so an
        implementation does not have to invent an answer for that case.

        `state_card` is a Mapping rather than a dict for the same reason it
        is one on GeneratorPort.propose: it is a read-only view of the
        Controller's state, and a collaborator has no business mutating it
        on the way past. It is the SAME card object the generator receives
        for this attempt, so a policy and a generator can never disagree
        about what the run looked like when the slot was chosen.
        """
        ...


@runtime_checkable
class GeneratorPort(Protocol):
    """Proposes the next thing to try (W4). One of only two LLM calls.

    IMPLEMENTED BY W4, AND methods/ IS CURRENTLY EMPTY. That is precisely
    why `propose` is being resignatured now: today the change costs one
    edit to this file and one to a test double, and every week it waits it
    costs a coordination round with another workstream instead. A seam is
    cheapest to move before anyone stands on it.
    """

    def propose(
        self, state_card: Mapping[str, Any], target_slot: SlotName
    ) -> HypothesisPayload:
        """Propose one change to `target_slot`, given the run so far.

        WHY THE SLOT IS A PARAMETER AND NOT A KEY IN THE STATE CARD
        ----------------------------------------------------------
        Both would "tell" the generator which slot to work on, and that is
        where the resemblance ends. A directive buried in a dict is a
        suggestion: it relies on the model noticing the key, understanding
        it as binding, and choosing to honour it - and when it does not,
        nothing anywhere surfaces that. The run simply proceeds, attacking
        a slot the search policy did not pick, and the cost of that
        evaluation is attributed to the wrong arm for the rest of the run.
        That is a silent failure, and a silent failure inside a search loop
        is the worst kind: it degrades results without producing a symptom.

        As a parameter it is an argument at a call site. It is visible in a
        stack trace, it is checkable against what came back, and a static
        checker can see it. The constraint stops being something we hope
        for and becomes something we assert.

        THE RETURNED PAYLOAD'S `target_slot` MUST EQUAL `target_slot`.
        The Controller compares them and treats a mismatch as a contract
        violation: it logs an ERROR classified `ErrorClass.CONTRACT`,
        counts the node and moves to the next candidate. It does NOT obey
        the payload's choice. An LLM that wandered off the requested slot
        has produced a hypothesis about something nobody asked about, and
        splicing it in would let the model quietly take back the search
        decision this parameter exists to remove from it.

        `state_card` is a Mapping rather than a concrete dict so the
        Controller can hand over a read-only view: the generator is the
        one component with an LLM behind it, and it has no business
        mutating the Controller's state on the way past.

        The returned `expected_gain` is advisory only — see
        HypothesisPayload's docstring. The Controller may use it to order
        the queue; it must never let it influence acceptance.
        """
        ...


@runtime_checkable
class RealizerPort(Protocol):
    """Turns a hypothesis into runnable slot code (W4). LLM call #2.

    The second half of the two-model split that makes the project's cost
    story reportable: a strong model proposes *what* to change and why
    (GeneratorPort), and a cheap model writes the code that does it. Two
    roles, two models, two token budgets, separately accountable in the
    run's cost breakdown - which is only possible if they are two seams
    rather than one.
    """

    def realize(self, hypothesis: HypothesisPayload) -> SlotConfig:
        """Produce a SlotConfig for `hypothesis.target_slot` - that slot ONLY.

        The realizer returns one slot's configuration, never a whole
        pipeline. Splicing it into the incumbent is the Controller's job,
        because only the Controller knows what the incumbent currently is,
        and it is the component answerable for candidate lineage.

        May raise RealizerExhausted when the hypothesis cannot be turned
        into runnable code. That ends the candidate, never the run.
        """
        ...


@runtime_checkable
class JournalPort(Protocol):
    """Append-only event log with replay (W3).

    This is what makes crash-resume and an auditable intervention count
    possible, so the Controller writes through it for every decision, not
    only the interesting ones.
    """

    def append(self, event: JournalEvent) -> None:
        """Record one event. Should be durable before returning — a
        resume is only as good as what actually reached disk."""
        ...

    def replay(self, run_id: str) -> Iterator[JournalEvent]:
        """Yield that run's events in the order they were appended.

        Returns an Iterator rather than a list because a real journal is a
        file that may be large and may be torn at the tail; streaming lets
        the reader stop at the last intact event (see
        contracts.JournalDecodeError) instead of failing to materialise
        the whole log.
        """
        ...
