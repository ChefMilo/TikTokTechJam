"""Deterministic test doubles for the Controller's ports.

These are permanent test instruments, not scaffolding to delete once the
real components land. Their value is that they have **known ground
truth**: with a real executor you can only check that the Controller did
something plausible, whereas here you know in advance what the right
answer is and can count how often the Controller gets it wrong. That
property does not expire when W3 ships.

Everything here is free of I/O, network and wall-clock dependence, and
uses a per-instance `random.Random` — never the global `random` module,
never numpy. A test that sleeps, writes files, or draws from shared
global state is a test that flakes at 3am and teaches you nothing when it
does.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Callable, Optional

from contracts import (
    CandidateResult,
    ErrorClass,
    EventKind,
    HypothesisPayload,
    JournalEvent,
    Metrics,
    PipelineConfig,
    SlotConfig,
    Status,
    Verdict,
)

# The doubles raise the port-level exceptions, not private ones of their own.
# That is the whole point of the hierarchy: the Controller catches the port
# contract, so swapping a double for W4's real component changes nothing
# about how failure is handled.
from controller.ports import GeneratorExhausted, RealizerExhausted

__all__ = [
    "BASELINE_GAUC",
    "BASELINE_NDCG5",
    "BASELINE_PRIMARY",
    "BASELINE_SIGMA",
    "AlwaysAcceptGate",
    "AlwaysRejectGate",
    "DeterministicRealizer",
    "FakeExecutor",
    "FailureHook",
    "InMemoryJournal",
    "ScriptExhaustedError",
    "ScriptedGenerator",
    "ScriptedRealizer",
    "mean_primary",
    "metrics_from_delta",
]


# ---------------------------------------------------------------------------
# Calibration constants
#
# Sourced from the organizers' published FM baseline, which we reproduced
# ourselves: see tests/test_rungs.py (FM_VALID_PUBLISHED) and
# vendor/kuairand-starter-kit/baseline_scores.json. The key names below
# match exactly what harness/metrics.py:evaluate puts into Metrics.values
# ("GAUC" and f"nDCG@{k}", so "nDCG@5" at the default k) — the fakes are
# only useful if the shape they emit is the shape the real harness emits.
# ---------------------------------------------------------------------------

BASELINE_GAUC = 0.6674
BASELINE_NDCG5 = 0.5357
BASELINE_PRIMARY = (BASELINE_GAUC + BASELINE_NDCG5) / 2.0
"""0.60155 — the unweighted mean of the two metrics above, which is how
`Metrics.primary` is defined and what the reproduced FM baseline scored
(published as 0.6016)."""

BASELINE_SIGMA = 0.0008
"""Measured seed-to-seed standard deviation of the primary metric, over
seeds 0-4 of the official FM baseline.

Worth internalising: the organizers' acceptance threshold is
epsilon = 0.002, only about 2.5 sigma. The search is operating inside a
margin where noise and signal are the same order of magnitude, which is
exactly why these fakes are built around a known-null ground truth.
"""

METRIC_KEY_GAUC = "GAUC"
METRIC_KEY_NDCG = "nDCG@5"

# Synthetic cost accounting. Fixed numbers, never a clock reading, so that
# a Controller budget test is reproducible. 40s is the vendor README's
# stated FM runtime (CPU, single core).
FAKE_WALL_SECONDS_PER_SEED = 40.0
FAKE_TOKENS_IN = 900
FAKE_TOKENS_OUT = 300

FailureHook = Callable[[PipelineConfig], Optional[tuple[ErrorClass, str]]]
"""Return None to let a config evaluate normally, or
`(error_class, error_excerpt)` to make it fail."""


def metrics_from_delta(delta: float) -> Metrics:
    """Baseline metrics shifted by `delta` on the primary.

    The same shift is applied to both component metrics, which makes
    `Metrics.primary` come out as exactly `BASELINE_PRIMARY + delta` —
    since primary is their unweighted mean. That exactness is deliberate:
    a test can then reason about the primary directly without having to
    model how the two components move relative to each other, which is
    not something these fakes are trying to simulate.
    """
    return Metrics(
        values={
            METRIC_KEY_GAUC: BASELINE_GAUC + delta,
            METRIC_KEY_NDCG: BASELINE_NDCG5 + delta,
        }
    )


def mean_primary(per_seed: Mapping[int, Metrics]) -> float:
    """Mean primary across seeds. Display/heuristic use only.

    Never the basis of an acceptance decision — that goes through the
    gate, which needs the per-seed pairing this average discards. Returns
    0.0 for an empty mapping (a FAILED candidate) so callers do not have
    to special-case it.
    """
    if not per_seed:
        return 0.0
    return sum(m.primary for m in per_seed.values()) / len(per_seed)


def _paired_delta(
    candidate: Mapping[int, Metrics], incumbent: Mapping[int, Metrics]
) -> float:
    """Mean of per-seed differences over seeds present in BOTH.

    Paired rather than a difference of means, matching what a real gate
    must do. Returns 0.0 when the two share no seeds.
    """
    shared = sorted(set(candidate) & set(incumbent))
    if not shared:
        return 0.0
    return sum(candidate[s].primary - incumbent[s].primary for s in shared) / len(shared)


# ---------------------------------------------------------------------------
# ExecutorPort double
# ---------------------------------------------------------------------------


class FakeExecutor:
    """An executor whose candidates are, by construction, all the same.

    THE POINT OF THIS FAKE
    ----------------------
    With the default `true_effect=0.0`, ground truth is: *no candidate is
    genuinely better than any other — every observed difference is noise*.
    Scores are drawn i.i.d. around the FM baseline with the real measured
    sigma, and the draw does not depend on the config at all.

    That makes it the strongest available test of acceptance logic,
    because the right answer is known in advance: **every acceptance the
    Controller makes against this fake is a false positive, and therefore
    countable.** A run that accepts 30 of 100 candidates has a broken
    gate or a broken loop, and you learn that without needing a GPU, a
    dataset, or a single second of training. Nothing measured against a
    real executor can tell you that, because there you never know whether
    a gain was real.

    `true_effect` lets a test flip the experiment around: set it to a
    known non-zero size and the same machinery measures false *negatives*
    — how often a genuine improvement of that size is missed.

    DETERMINISM AND ITS ONE SHARP EDGE
    ----------------------------------
    Draws come from a per-instance `random.Random(seed)` stream, so two
    instances built with the same seed and driven through the same
    sequence of calls produce byte-identical results. The edge: the stream
    advances with every call, so results depend on the *call sequence*,
    not on the config. Ask for the same config twice and you get two
    different answers — which is realistic (that is what re-running is)
    but means this fake deliberately does not model a cache. It also
    means a test that reorders its `run()` calls will see different
    numbers; build a fresh FakeExecutor per test rather than sharing one.

    `random.normalvariate` is used rather than `random.gauss` because
    gauss historically cached a spare value between calls, making results
    depend on call parity — a needless subtlety in something whose whole
    job is to be predictable.
    """

    def __init__(
        self,
        seed: int = 0,
        true_effect: float = 0.0,
        sigma: float = BASELINE_SIGMA,
        fail_on: Optional[FailureHook] = None,
    ) -> None:
        self.seed = seed
        self.true_effect = true_effect
        self.sigma = sigma
        self._fail_on = fail_on
        self._rng = random.Random(seed)
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        """(config_id, seeds) for every run() call, so a test can assert on
        what the Controller actually asked for — e.g. that it evaluated
        candidate and incumbent on the *same* seeds."""

    def run(self, config: PipelineConfig, seeds: Sequence[int]) -> CandidateResult:
        requested = tuple(seeds)
        config_id = config.config_id
        self.calls.append((config_id, requested))

        failure = self._fail_on(config) if self._fail_on is not None else None
        if failure is not None:
            error_class, excerpt = failure
            # val/backtest left empty: a candidate that failed produced no
            # scores. They stay required-but-empty rather than absent,
            # because CandidateResult has no way to express "not run".
            return CandidateResult(
                config_id=config_id,
                status=Status.FAILED,
                val={},
                backtest={},
                error_class=error_class,
                error_excerpt=excerpt,
                wall_seconds=FAKE_WALL_SECONDS_PER_SEED * len(requested),
                tokens_in=FAKE_TOKENS_IN,
                tokens_out=FAKE_TOKENS_OUT,
            )

        val: dict[int, Metrics] = {}
        backtest: dict[int, Metrics] = {}
        for s in requested:
            val[s] = metrics_from_delta(self._draw())
            # Drawn independently, so val and backtest are not perfectly
            # correlated. A real backtest is a different split, and a
            # candidate can win on one and lose on the other — which is
            # precisely the overfitting signal the backtest exists to
            # catch. A fake that returned the same numbers for both would
            # make that check untestable.
            backtest[s] = metrics_from_delta(self._draw())

        return CandidateResult(
            config_id=config_id,
            status=Status.OK,
            val=val,
            backtest=backtest,
            wall_seconds=FAKE_WALL_SECONDS_PER_SEED * len(requested),
            tokens_in=FAKE_TOKENS_IN,
            tokens_out=FAKE_TOKENS_OUT,
        )

    def _draw(self) -> float:
        """One shift off the baseline: the true effect plus fresh noise."""
        return self.true_effect + self._rng.normalvariate(0.0, self.sigma)


# ---------------------------------------------------------------------------
# GeneratorPort double
# ---------------------------------------------------------------------------


class ScriptExhaustedError(GeneratorExhausted):
    """Raised when a ScriptedGenerator is asked for one hypothesis too many.

    WHY IT SUBCLASSES GeneratorExhausted AND STILL LIVES HERE: the name is
    fake-specific and belongs with the fake, but the Controller must never
    import it. Reparenting onto the port-level exception lets the
    Controller catch `ports.GeneratorExhausted` and handle a scripted
    double and W4's real generator through one code path, which removed a
    production-code dependency on this module.

    It remains a RuntimeError by inheritance, so any handler written
    against the old base still catches it.
    """


class ScriptedGenerator:
    """Yields a fixed, ordered list of hypotheses. No LLM, no randomness.

    With a scripted generator, any deviation in Controller behaviour is a
    bug in the Controller — never model variance. That is the entire
    reason this exists: it removes the only nondeterministic component in
    the loop so the rest can be tested as ordinary software.

    EXHAUSTION: it does NOT cycle. Running off the end of the script means
    the Controller iterated more times than the test intended, and
    silently wrapping around would hide that — worse, a Controller looping
    until the generator stops would never stop. So exhaustion raises.

    It raises `ScriptExhaustedError` rather than `StopIteration`
    specifically: under PEP 479 a StopIteration escaping into a
    surrounding generator is converted to an opaque RuntimeError, which
    would turn a clear "your script was too short" into a confusing
    traceback far from the cause.
    """

    def __init__(self, script: Sequence[HypothesisPayload]) -> None:
        self._script = tuple(script)
        self._index = 0
        self.state_cards: list[dict[str, Any]] = []
        """A copy of every state_card handed in, so a test can assert on
        what the Controller believed about the run when it asked."""

    def propose(self, state_card: Mapping[str, Any]) -> HypothesisPayload:
        self.state_cards.append(dict(state_card))
        if self._index >= len(self._script):
            raise ScriptExhaustedError(
                f"ScriptedGenerator exhausted after {len(self._script)} "
                f"hypotheses; the Controller asked for one more"
            )
        payload = self._script[self._index]
        self._index += 1
        return payload

    @property
    def remaining(self) -> int:
        """Hypotheses left in the script."""
        return len(self._script) - self._index


# ---------------------------------------------------------------------------
# RealizerPort doubles
# ---------------------------------------------------------------------------


class DeterministicRealizer:
    """Realizes a hypothesis into a SlotConfig with no script to maintain.

    The default realizer double, and the one most tests should use. A test
    that only cares about the loop should not have to keep a hypothesis
    script and a config script in lockstep - an off-by-one between the two
    is a test bug that looks exactly like a Controller bug, and costs an
    afternoon to tell apart.

    WHAT GOES INTO THE CONFIG, AND WHAT POINTEDLY DOES NOT: `impl` comes
    from `citation.library_entry`, the payload's pointer into the method
    library, and `params` carries only `citation.key`. Identity-bearing
    fields only. `expected_gain` and `expected_cost_s` are excluded even
    though they sit right there on the payload, because they are advisory
    forecasts - folding a forecast into `params` folds it into the content
    hash, so two identical proposals that merely disagreed about how much
    they would help would produce two different config_ids and quietly
    defeat both caching and dedup.

    Emits no code_blob: this models a registry-backed realization, where
    the implementation already exists in the library and only needs
    selecting. Use ScriptedRealizer when a test needs freeform code.
    """

    def realize(self, hypothesis: HypothesisPayload) -> SlotConfig:
        return SlotConfig(
            impl=hypothesis.citation.library_entry,
            params={"method_key": hypothesis.citation.key},
        )


class ScriptedRealizer:
    """Returns a fixed, ordered list of SlotConfigs. For exact control.

    Does not cycle, for the same reason ScriptedGenerator does not:
    wrapping around would hide a Controller that iterated more times than
    the test intended. Exhaustion raises `RealizerExhausted` - the
    port-level exception - so a test exercises the same handling path the
    Controller will use when W4's real realizer gives up on a hypothesis.
    """

    def __init__(self, script: Sequence[SlotConfig]) -> None:
        self._script = tuple(script)
        self._index = 0
        self.calls: list[HypothesisPayload] = []
        """Every hypothesis handed in, in order, so a test can assert on
        what the Controller actually asked to have realized."""

    def realize(self, hypothesis: HypothesisPayload) -> SlotConfig:
        self.calls.append(hypothesis)
        if self._index >= len(self._script):
            raise RealizerExhausted(
                f"ScriptedRealizer exhausted after {len(self._script)} "
                f"configs; the Controller asked for one more"
            )
        config = self._script[self._index]
        self._index += 1
        return config

    @property
    def remaining(self) -> int:
        """SlotConfigs left in the script."""
        return len(self._script) - self._index


# ---------------------------------------------------------------------------
# JournalPort double
# ---------------------------------------------------------------------------


class InMemoryJournal:
    """Journal that keeps events in a list. Never touches disk.

    Append order is the source of truth, deliberately: `replay` returns
    events in the order they arrived rather than sorting by `ts` or
    `(iteration, node)`. A real journal is an append-only file, so that is
    what the file would give back, and a fake that quietly sorted would
    hide a Controller that emitted events out of order.
    """

    def __init__(self) -> None:
        self._events: list[JournalEvent] = []

    def append(self, event: JournalEvent) -> None:
        self._events.append(event)

    def replay(self, run_id: str) -> Iterator[JournalEvent]:
        """Yield only this run's events, in append order."""
        return iter([e for e in self._events if e.run_id == run_id])

    @property
    def events(self) -> tuple[JournalEvent, ...]:
        """Everything recorded, across all runs. A tuple so a test cannot
        accidentally mutate the journal it is asserting on."""
        return tuple(self._events)

    def events_of_kind(
        self, kind: EventKind, run_id: Optional[str] = None
    ) -> tuple[JournalEvent, ...]:
        """Recorded events of one kind, optionally limited to one run.

        Exists because most journal assertions are of the form "exactly
        three DECISION events were logged, and each carried a delta" —
        writing that filter inline in every test invites subtle
        inconsistencies between them.
        """
        return tuple(
            e
            for e in self._events
            if e.kind is kind and (run_id is None or e.run_id == run_id)
        )


# ---------------------------------------------------------------------------
# GatePort doubles
# ---------------------------------------------------------------------------


class _ConstantGate:
    """Shared body for the two boundary gates.

    These are boundary instruments, not statistics. `delta` and
    `backtest_delta` are computed honestly from the per-seed pairing so
    the journal payloads look real, but `ci95` is a fixed-width interval
    around the delta — NOT a bootstrap CI, and nothing should read it as
    one. Their job is to pin the Controller's two extreme paths: what
    happens when everything is accepted, and when nothing is.
    """

    _ACCEPT: bool = False
    _REASON: str = ""

    def compare(self, candidate: CandidateResult, incumbent: CandidateResult) -> Verdict:
        delta = _paired_delta(candidate.val, incumbent.val)
        backtest_delta = _paired_delta(candidate.backtest, incumbent.backtest)
        # 1.96 sigma: nominal-width placeholder so the tuple is plausible.
        half_width = 1.96 * BASELINE_SIGMA
        return Verdict(
            accept=self._ACCEPT,
            delta=delta,
            ci95=(delta - half_width, delta + half_width),
            # Required float, never None — CandidateResult.backtest is
            # itself required, so there is no "no backtest" case to model.
            backtest_delta=backtest_delta,
            reason=self._REASON,
        )


class AlwaysAcceptGate(_ConstantGate):
    """Accepts everything. Drives the Controller's accept path, and — run
    against a default FakeExecutor — produces a run in which every single
    acceptance is provably a false positive."""

    _ACCEPT = True
    _REASON = "AlwaysAcceptGate: accepts unconditionally (test double)"


class AlwaysRejectGate(_ConstantGate):
    """Rejects everything. Drives the reject path, and is the fixture for
    checking that the Controller still terminates, still journals, and
    still converges when nothing ever improves."""

    _ACCEPT = False
    _REASON = "AlwaysRejectGate: rejects unconditionally (test double)"
