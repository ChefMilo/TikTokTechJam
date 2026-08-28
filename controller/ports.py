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
    Verdict,
)

__all__ = [
    "ExecutorPort",
    "GatePort",
    "GeneratorPort",
    "JournalPort",
]


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
class GeneratorPort(Protocol):
    """Proposes the next thing to try (W4). One of only two LLM calls."""

    def propose(self, state_card: Mapping[str, Any]) -> HypothesisPayload:
        """Given a compact summary of the run so far, propose one change.

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
