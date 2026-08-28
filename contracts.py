"""Frozen cross-package interfaces for the autonomous ML research agent.

This module is the single source of truth for every data structure and
protocol that crosses a package boundary. The four workstreams --
``harness/`` (W1), ``controller/`` (W2), ``executor/`` (W3) and
``methods/`` (W4) -- import from here and never from each other.

Design rules that this module exists to enforce:

* **Stdlib only.** No pydantic, no attrs, no third-party validation. A
  small dependency surface is a scored criterion for the project, and
  every type here is a plain dataclass, enum or Protocol.
* **Value objects are frozen.** ``Metrics``, ``SlotConfig``, ``Verdict``
  and ``HypothesisPayload`` cannot be mutated after construction, so a
  result cannot be edited in flight between the executor and the gate.
  (Note that ``frozen=True`` prevents attribute rebinding, not mutation
  of a ``dict`` *inside* a field. Treat dict fields as read-only by
  convention.)
* **Content addressing over identity.** Candidates, slots and cache keys
  are identified by the sha256 of their canonical JSON, never by object
  identity or insertion order, so two processes agree on what has
  already been computed.

Every non-obvious decision below is documented with *why*, because three
other people code against this file without asking.

Stability: this is contracts v1. Adding an optional field with a default
or a new enum member is a minor bump; renaming, reordering or removing
anything is a major bump and requires all four workstreams to agree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

CONTRACTS_VERSION = "1.0.0"
"""Bumped when this module changes. Log it in the RUN_START journal event
so an archived run can be replayed against the contracts it was produced
under."""

HASH_LENGTH = 16
"""Truncation length for all content hashes.

16 hex chars is 64 bits. At the scale of a hackathon run (thousands of
candidates, not billions) collision probability is negligible, and short
hashes keep journal lines and cache filenames readable.
"""


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Coerce the few non-JSON types we permit inside ``params``.

    Deliberately narrow: anything else raises ``TypeError`` rather than
    falling back to ``repr()``. A ``repr()`` fallback would embed memory
    addresses, which would make hashes unstable across processes and
    silently break the artifact cache -- the exact failure this module
    exists to prevent.
    """
    if isinstance(obj, StrEnum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(
        f"{type(obj).__name__} is not JSON-serialisable; slot params must be "
        "built from str/int/float/bool/None/list/dict (plus StrEnum, set, Path)"
    )


def canonical_json(payload: Any) -> str:
    """Serialise ``payload`` to the one canonical string form we hash.

    ``sort_keys=True`` applies recursively, so dict insertion order can
    never change the hash -- two controllers that built the same config
    in a different order must agree on its identity.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def stable_hash(payload: Any, length: int = HASH_LENGTH) -> str:
    """Content hash of ``payload``: canonical JSON -> sha256 -> truncate."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:length]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """One evaluation's scores.

    ``primary`` is the single authoritative number the agent optimises and
    the only one the noise gate compares. ``metric_name`` records which
    metric that is.

    Why not typed ``gauc`` / ``ndcg`` fields: the organisers have not
    confirmed which metric is authoritative. Hardcoding fields would mean
    a schema migration across all four packages the day they announce it.
    With this shape, switching the objective is a one-line change to
    whatever the harness puts in ``metric_name``, and everything already
    computed stays valid because it is all preserved in ``secondary``.

    Always populate ``secondary`` with everything you computed, even the
    metrics you are not optimising -- it costs nothing at eval time and it
    is what lets the final report show that the winning candidate did not
    quietly regress some other axis.
    """

    primary: float
    metric_name: str
    secondary: dict[str, float] = field(default_factory=dict)

    def __hash__(self) -> int:
        # The auto-generated __hash__ would raise TypeError on the
        # `secondary` dict. Hash by content instead so Metrics can be used
        # in sets and as dict keys.
        return hash(stable_hash(self.as_dict()))

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view, for journal payloads and reports."""
        return {
            "primary": self.primary,
            "metric_name": self.metric_name,
            "secondary": dict(self.secondary),
        }


# ---------------------------------------------------------------------------
# Pipeline slots
# ---------------------------------------------------------------------------


class SlotName(StrEnum):
    """The contracted stages of the pipeline DAG.

    A ``StrEnum`` rather than a bare ``Enum`` so members serialise to
    their own value in journal lines and cache keys with no custom
    encoder, and round-trip via ``SlotName(value)``.
    """

    DATA_VIEW = "data_view"
    FEATURE_BLOCKS = "feature_blocks"
    SAMPLE_WEIGHTING = "sample_weighting"
    MODEL = "model"
    OBJECTIVE = "objective"
    CALIBRATION = "calibration"
    BLEND = "blend"
    CUSTOM = "custom"
    """Escape hatch for freeform agent-written code that still honours the
    slot contract but does not fit a named stage. Treated as depending on
    every other slot (see ``PipelineConfig.upstream_chain``)."""


PIPELINE_ORDER: tuple[SlotName, ...] = (
    SlotName.DATA_VIEW,
    SlotName.FEATURE_BLOCKS,
    SlotName.SAMPLE_WEIGHTING,
    SlotName.MODEL,
    SlotName.OBJECTIVE,
    SlotName.CALIBRATION,
    SlotName.BLEND,
)
"""Topological order of the pipeline DAG.

``CUSTOM`` is deliberately absent: its position is unknown, so it is
handled conservatively as depending on everything.
"""


@dataclass(frozen=True)
class SlotConfig:
    """One slot's chosen implementation and its hyperparameters.

    ``impl`` is a library identifier ("fm", "lightgbm", "isotonic") that
    W4's method library resolves to code. ``params`` must be built from
    JSON-native types -- that is what makes ``content_hash`` reproducible
    across processes.
    """

    slot: SlotName
    impl: str
    params: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        # As with Metrics: hash by content, because `params` is a dict.
        return hash(self.content_hash())

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view; also the exact payload that is hashed."""
        return {
            "slot": self.slot.value,
            "impl": self.impl,
            "params": dict(self.params),
        }

    def content_hash(self) -> str:
        """Stable hash of this slot alone, ignoring its upstream context.

        Use ``PipelineConfig.slot_hash`` for cache keys -- a slot's output
        depends on everything feeding it, so this hash on its own is not a
        valid cache key.
        """
        return stable_hash(self.as_dict())


@dataclass
class PipelineConfig:
    """A full candidate solution: one ``SlotConfig`` per occupied slot.

    Not frozen, because the controller builds a candidate by copying an
    incumbent and applying a one- or two-slot diff. Identity is content
    based (``config_id``), so mutability does not compromise caching --
    the id simply follows the content.

    ``data_version`` participates in every hash. A change to the
    underlying dataset or split definition must invalidate every cached
    artifact, and folding it into the keys makes that automatic rather
    than a thing someone has to remember to do.
    """

    slots: dict[SlotName, SlotConfig] = field(default_factory=dict)
    data_version: str = "unset"

    @property
    def config_id(self) -> str:
        """Stable content hash of the whole candidate.

        A computed property rather than a stored field so it can never
        drift from the content it claims to identify. Copy it into
        ``CandidateResult.config_id`` when a run completes; that snapshot
        is what the journal and the cache refer to afterwards.
        """
        return stable_hash(
            {
                "data_version": self.data_version,
                "slots": {
                    slot.value: cfg.content_hash() for slot, cfg in self.slots.items()
                },
            }
        )

    def upstream_chain(self, slot: SlotName) -> tuple[SlotName, ...]:
        """Occupied slots that feed ``slot``, in DAG order, ending at it.

        ``CUSTOM`` is treated as depending on every occupied named slot,
        because we cannot know where agent-written code sits in the DAG.
        Over-invalidating its cache entries is cheap; serving a stale one
        would silently corrupt a result.
        """
        if slot is SlotName.CUSTOM:
            chain = [s for s in PIPELINE_ORDER if s in self.slots]
            if SlotName.CUSTOM in self.slots:
                chain.append(SlotName.CUSTOM)
            return tuple(chain)
        cutoff = PIPELINE_ORDER.index(slot) + 1
        return tuple(s for s in PIPELINE_ORDER[:cutoff] if s in self.slots)

    def slot_hash(self, slot: SlotName) -> str:
        """Artifact cache key for ``slot``'s output in this pipeline.

        Combines ``data_version`` with the content hash of every upstream
        slot in DAG order, then this slot's own. Changing a hyperparameter
        of an upstream slot changes this key; changing a *downstream* slot
        does not, which is what lets the search reuse an expensive feature
        matrix while it sweeps models on top of it.

        **The seed is deliberately not an input.** Feature matrices and
        data views are seed-independent, so one cached artifact serves
        every seed -- and since we evaluate on multiple seeds per
        candidate, folding the seed in here would multiply the most
        expensive part of the run by the seed count for no benefit. The
        seed-dependent outputs (prediction vectors) are keyed separately
        by ``(config_id, seed)`` -- see ``prediction_key``.

        Raises ``KeyError`` if the slot is not occupied: there is no
        meaningful cache key for a stage that does not exist.
        """
        if slot not in self.slots:
            raise KeyError(f"slot {slot.value!r} is not present in this PipelineConfig")
        return stable_hash(
            {
                "data_version": self.data_version,
                "chain": [
                    [s.value, self.slots[s].content_hash()]
                    for s in self.upstream_chain(slot)
                ],
            }
        )

    def with_slot(self, config: SlotConfig) -> PipelineConfig:
        """Copy of this pipeline with one slot replaced.

        The primitive the controller's diff operator is built from;
        returning a copy keeps the incumbent untouched.
        """
        slots = dict(self.slots)
        slots[config.slot] = config
        return PipelineConfig(slots=slots, data_version=self.data_version)


def prediction_key(config_id: str, seed: int) -> str:
    """Cache key for one candidate's saved prediction vector at one seed.

    The counterpart to ``PipelineConfig.slot_hash``: seed-*dependent*
    outputs are keyed here, seed-*independent* ones there. Keeping the two
    key spaces separate is what allows a single cached feature matrix to
    serve every seed.
    """
    return stable_hash({"config_id": config_id, "seed": seed})


# ---------------------------------------------------------------------------
# Run outcomes
# ---------------------------------------------------------------------------


class RunStatus(StrEnum):
    """Terminal state of one candidate evaluation."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Never attempted -- e.g. budget ran out, or the slot was blocked
    after repeated failures. Distinct from FAILED so the report does not
    count a deliberate skip as a robustness failure."""


class ErrorClass(StrEnum):
    """Closed taxonomy of evaluation failures.

    Each member maps 1:1 to an entry in W3's repair policy table. The set
    is deliberately closed and exhaustive: a closed set is what makes that
    table total rather than best-effort, so there is always a defined
    response. Anything unrecognised must be classified ``UNKNOWN`` (which
    has its own policy -- do not repair, log and block the slot) rather
    than prompting a new member mid-run.
    """

    NONE = "none"
    """No error. The correct value on a successful run."""

    SYNTAX = "syntax"
    """Generated code does not parse or import."""

    CONTRACT = "contract"
    """Code ran but violated the slot's interface (wrong return type,
    missing method, wrong column names)."""

    SHAPE_MISMATCH = "shape_mismatch"
    """Array/frame dimensions disagree between stages."""

    OOM = "oom"
    TIMEOUT = "timeout"

    NAN_LOSS = "nan_loss"
    """Training diverged: loss became NaN or inf."""

    DEGENERATE = "degenerate"
    """Ran cleanly but the output is useless -- constant predictions, all
    one class, zero variance. Caught by the harness, not by an exception,
    which is why it needs its own class."""

    DEPENDENCY = "dependency"
    """Required a package that is not installed or not permitted."""

    UNKNOWN = "unknown"


@dataclass
class CandidateResult:
    """Everything one candidate evaluation produced. Returned by W3.

    This is the unit the controller reasons over and the journal records.
    """

    config_id: str
    status: RunStatus

    val: dict[int, Metrics] = field(default_factory=dict)
    """Validation metrics keyed **by seed**.

    Not a pre-averaged float, and this is load-bearing. The noise gate
    performs a *paired* per-seed comparison: for each seed it takes
    candidate minus incumbent on that same seed, and builds a confidence
    interval over those differences. Pairing cancels the shared
    seed-to-seed variance, which on this benchmark is larger than the
    effects we are chasing. Averaging first destroys the pairing and the
    gate can no longer tell a real 0.3% gain from seed noise -- so it
    would accept changes that are worth nothing.

    Store every seed you ran. Never collapse this before the gate sees it.
    """

    backtest: Metrics | None = None
    """Rolling-origin forward split: fit on an early window, score on a
    later one, both carved from training data only.

    A second, temporally honest read on a candidate that never touches the
    validation split, so it is independent evidence rather than a second
    chance at the same test. Used to catch candidates that win on
    validation by exploiting its particular time slice. ``None`` when the
    backtest was not run.
    """

    error_class: ErrorClass = ErrorClass.NONE
    traceback: str | None = None
    """Full traceback text when ``status`` is FAILED, for the repair
    prompt and the report. ``None`` on success."""

    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0

    tokens: dict[str, int] = field(default_factory=dict)
    """LLM token counts broken down by role -- "hypothesis", "realize",
    "repair". Broken down rather than totalled because the report needs to
    show where the token budget actually went, and repair tokens spent is
    a direct measure of how much robustness cost us.
    """

    pred_paths: dict[int, str] = field(default_factory=dict)
    """Per-seed paths to saved float32 ``.npy`` prediction vectors.

    Persisted so that finalisation can blend the top candidates by
    combining stored predictions, with zero retraining. Keyed by seed to
    match ``val``; see ``prediction_key`` for the cache key convention.
    """

    def seeds(self) -> tuple[int, ...]:
        """Seeds this candidate was evaluated on, ascending."""
        return tuple(sorted(self.val))

    def mean_primary(self) -> float | None:
        """Mean of ``primary`` across seeds -- **display and logging only**.

        Never use this for an acceptance decision. Acceptance goes through
        the noise gate, which needs the per-seed pairing that this number
        throws away (see ``val``). It exists so a progress line or a report
        row can show one number. ``None`` when no seeds were evaluated.
        """
        if not self.val:
            return None
        return sum(m.primary for m in self.val.values()) / len(self.val)


@dataclass(frozen=True)
class Verdict:
    """The noise gate's ruling on candidate-vs-incumbent. Returned by W1.

    Note for W1: the existing ``harness.gate.passes_gate`` stub is named
    as though it returns a bool. It must return a ``Verdict`` instead. A
    bare bool discards ``delta`` and ``ci95``, which the controller needs
    for convergence tracking (are our accepted gains shrinking towards
    noise?) and the report needs to show *why* something was accepted.
    Resolving that rename is W1's task; this contract is the target.
    """

    accepted: bool
    delta: float
    """Mean paired difference in the primary metric, candidate minus
    incumbent. Positive means the candidate scored higher."""

    ci95: tuple[float, float]
    """95% confidence interval on ``delta``, as (low, high). The gate
    accepts when this interval excludes zero on the positive side --
    that is the whole point of the per-seed pairing."""

    n_seeds: int
    """Number of paired seeds behind ``delta``. Recorded because a verdict
    from 3 seeds and one from 10 are not equally trustworthy."""

    backtest_delta: float | None = None
    """Same difference measured on the backtest split, when available.
    Advisory corroboration: a positive validation delta with a negative
    backtest delta is the signature of overfitting to the val slice."""

    reason: str = ""
    """Human-readable justification, journalled verbatim and quoted in the
    report. Populate it even on acceptance."""


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisPayload:
    """A proposed change to one slot. Returned by W4's generator.

    One of exactly two points where an LLM is in the loop (the other is
    code realisation). Everything downstream of this is deterministic.
    """

    slot: SlotName
    method_id: str
    """Identifier into W4's YAML method library."""

    citation: str
    """Where the method comes from -- paper, or the library entry that
    describes it. Recorded so the report can attribute every accepted
    change to a real technique rather than to an unsourced guess."""

    rationale: str
    """Why the generator expects this to help *given the current state
    card*. Journalled before evaluation, so the log shows the reasoning
    that led to a run rather than a retrospective story about it."""

    expected_gain: float
    """Forecast improvement in the primary metric.

    **Advisory only -- it must never gate acceptance.** It is logged
    before the candidate runs purely so that afterwards we can score the
    generator's forecast calibration (predicted gain vs. measured delta).
    Letting it influence acceptance would launder an LLM's guess into a
    result: only the noise gate, on measured per-seed evidence, decides
    what is accepted. It is legitimate to use it to *order* the search
    queue -- that only changes what we try first, never what we keep.
    """

    proposed: SlotConfig


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


class EventKind(StrEnum):
    """The journal's closed event vocabulary.

    The journal is the append-only log that makes crash-resume and an
    auditable intervention count possible, so the vocabulary is fixed:
    a reader written today must be able to parse a log written later.
    """

    RUN_START = "run_start"
    STAGE_ENTER = "stage_enter"
    HYPOTHESIS = "hypothesis"
    EVAL_START = "eval_start"
    EVAL_END = "eval_end"
    DECISION = "decision"
    ERROR = "error"
    REPAIR_ATTEMPT = "repair_attempt"
    RECOVERY = "recovery"
    SLOT_BLOCKED = "slot_blocked"
    """A slot has failed too often and is excluded from further search."""

    BUDGET = "budget"
    INTERVENTION = "intervention"
    """A human touched the run. Counted directly off the journal, so the
    autonomy claim in the report is measured rather than asserted."""

    CHECKPOINT = "checkpoint"
    """Resumable state boundary. Replay stops at the last one on crash."""

    RUN_END = "run_end"


@dataclass
class JournalEvent:
    """One line of the append-only journal.

    Serialises to exactly one line of JSON so the journal is an ordinary
    JSONL file: appendable without rewriting, tailable while running, and
    replayable line by line after a crash.
    """

    kind: EventKind
    ts: float
    """Unix timestamp, seconds."""

    run_id: str
    seq: int
    """Monotonic per-run sequence number. Ordering must not depend on
    ``ts``, whose resolution can tie between two fast consecutive events."""

    payload: dict[str, Any] = field(default_factory=dict)
    """Kind-specific body. Must be JSON-object-safe: string keys, and
    values built from JSON-native types. In particular a dict keyed by
    seed (an int) will come back with string keys -- convert seeds to
    ``str`` on the way in, or store a list of pairs.
    """

    def to_json(self) -> str:
        """Serialise to a single JSON line (no trailing newline).

        Keys are sorted so two runs producing the same event produce
        byte-identical lines, which makes journals diffable.
        """
        return json.dumps(
            {
                "kind": self.kind.value,
                "ts": self.ts,
                "run_id": self.run_id,
                "seq": self.seq,
                "payload": self.payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )

    @classmethod
    def from_json(cls, line: str) -> JournalEvent:
        """Parse one journal line. Inverse of ``to_json``.

        Raises ``ValueError`` on a malformed or truncated line -- which is
        expected at the tail of a journal from a crashed run, and is the
        signal for replay to stop there rather than guess.
        """
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("journal line must be a JSON object")
        missing = {"kind", "ts", "run_id", "seq", "payload"} - raw.keys()
        if missing:
            raise ValueError(f"journal line missing fields: {sorted(missing)}")
        return cls(
            kind=EventKind(raw["kind"]),
            ts=float(raw["ts"]),
            run_id=str(raw["run_id"]),
            seq=int(raw["seq"]),
            payload=dict(raw["payload"]),
        )


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class BudgetCounter:
    """One resource: how much we may spend, and how much we have spent."""

    limit: float = float("inf")
    """``inf`` means unconstrained -- the right default for a resource the
    run does not actually meter (e.g. GPU seconds on a CPU-only box)."""

    consumed: float = 0.0

    @property
    def remaining(self) -> float:
        return self.limit - self.consumed

    @property
    def exhausted(self) -> bool:
        """True at or past the limit. ``>=``, not ``>``: a limit of zero
        means do not start."""
        return self.consumed >= self.limit


@dataclass
class Budget:
    """The run's spending envelope across all metered resources.

    Held by the controller, which checks it before starting any candidate.
    Exceeding a budget must end the run cleanly through a RUN_END event,
    never by being killed -- an unfinished journal cannot be reported on.
    """

    wall_seconds: BudgetCounter = field(default_factory=BudgetCounter)
    tokens: BudgetCounter = field(default_factory=BudgetCounter)
    evaluations: BudgetCounter = field(default_factory=BudgetCounter)
    gpu_seconds: BudgetCounter = field(default_factory=BudgetCounter)

    def counters(self) -> dict[str, BudgetCounter]:
        """Counters by name, in a fixed order for stable reporting."""
        return {
            "wall_seconds": self.wall_seconds,
            "tokens": self.tokens,
            "evaluations": self.evaluations,
            "gpu_seconds": self.gpu_seconds,
        }

    def tripped(self) -> tuple[str, ...]:
        """Names of every counter at or past its limit, in fixed order.

        Returns all of them rather than the first, so the BUDGET journal
        event and the report say exactly what ran out -- two limits can
        land on the same evaluation.
        """
        return tuple(name for name, c in self.counters().items() if c.exhausted)

    def exhausted(self) -> bool:
        """True when *any* counter is at or past its limit."""
        return bool(self.tripped())


# ---------------------------------------------------------------------------
# Protocols -- interface only, no implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """W1. Scores a candidate on one split at one seed."""

    def evaluate(self, config: PipelineConfig, split: str, seed: int) -> Metrics:
        """``split`` is one of "train", "val", "test", "backtest"."""
        ...


@runtime_checkable
class NoiseGate(Protocol):
    """W1. Decides whether a measured improvement is real."""

    def compare(self, candidate: CandidateResult, incumbent: CandidateResult) -> Verdict:
        """Paired per-seed comparison over the seeds present in both."""
        ...


@runtime_checkable
class ArtifactCache(Protocol):
    """W1. Content-addressed store keyed by ``slot_hash``/``prediction_key``."""

    def get(self, key: str) -> Any | None:
        """Cached artifact, or ``None`` on miss. Must never raise on miss."""
        ...

    def put(self, key: str, artifact: Any) -> None: ...


@runtime_checkable
class Executor(Protocol):
    """W3. Runs a candidate across seeds in a sandbox and reports back."""

    def run(self, config: PipelineConfig, seeds: list[int]) -> CandidateResult:
        """Must return a ``CandidateResult`` even on failure -- with
        ``status=FAILED`` and a classified ``error_class`` -- rather than
        raising. The controller decides what to do about a failure; a
        raised exception would take the whole run down with it."""
        ...


@runtime_checkable
class Journal(Protocol):
    """W3. Append-only event log with replay."""

    def append(self, event: JournalEvent) -> None:
        """Must be durable before returning -- a resume is only as good as
        what actually reached disk."""
        ...

    def replay(self, run_id: str) -> Iterator[JournalEvent]:
        """Events for ``run_id`` in ``seq`` order, stopping at the first
        malformed line (see ``JournalEvent.from_json``)."""
        ...


@runtime_checkable
class HypothesisGenerator(Protocol):
    """W4. LLM call #1: propose the next thing to try."""

    def propose(self, state_card: dict[str, Any]) -> HypothesisPayload:
        """``state_card`` is the controller's compact summary of the run so
        far -- incumbent scores, what has been tried, what is blocked."""
        ...


@runtime_checkable
class CodeRealizer(Protocol):
    """W4. LLM call #2: turn a hypothesis into a runnable slot config."""

    def realize(self, hypothesis: HypothesisPayload) -> SlotConfig: ...


__all__ = [
    "CONTRACTS_VERSION",
    "HASH_LENGTH",
    "PIPELINE_ORDER",
    "ArtifactCache",
    "Budget",
    "BudgetCounter",
    "CandidateResult",
    "CodeRealizer",
    "ErrorClass",
    "EventKind",
    "Evaluator",
    "Executor",
    "HypothesisGenerator",
    "HypothesisPayload",
    "Journal",
    "JournalEvent",
    "Metrics",
    "NoiseGate",
    "PipelineConfig",
    "RunStatus",
    "SlotConfig",
    "SlotName",
    "Verdict",
    "canonical_json",
    "prediction_key",
    "stable_hash",
]
