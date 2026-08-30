"""Shared interfaces between harness, controller, executor, and methods.

This is the single shared interface file for the project: all four
packages import from here, and never from each other's internals. It is
frozen once committed — changing a contract here is a cross-team event,
not a local refactor.

HARD RULE: contracts carry data, never behaviour. Everything below is a
frozen dataclass, an enum, or a pure helper method (hashing, averaging,
(de)serialization). There are no registries, no factories, and no
implementations here. If a function would *do* something beyond deriving
a value from the fields already on the object, it belongs in harness,
controller, executor, or methods — not here.

The three contracts:

  1. PipelineConfig  — identifies a candidate method (what to run).
  2. CandidateResult — the outcome of running one (what happened), plus
     the Verdict the noise gate renders over it.
  3. JournalEvent    — an append-only log record of anything the agent
     did or decided, for audit and post-hoc analysis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# PART 0: enums
# ---------------------------------------------------------------------------


class Status(str, Enum):
    """Drives the controller's accept/reject logic.

    Kept separate from ErrorClass on purpose: Status answers "what should
    the controller do with this candidate", ErrorClass answers "what
    should the executor do to fix it". Same event, different readers.
    """

    OK = "ok"
    OK_AFTER_REPAIR = "ok_after_repair"
    SCREENED_OUT = "screened_out"
    FAILED = "failed"


class ErrorClass(str, Enum):
    """Drives the executor's repair routing.

    DEGENERATE gets its own value rather than folding into FAILED because
    it needs different handling: a degenerate model (constant or
    near-constant scores) doesn't crash, so there's nothing to repair —
    it must be rejected outright rather than retried like a transient
    OOM/TIMEOUT would be.
    """

    NONE = "none"
    SYNTAX = "syntax"
    CONTRACT = "contract"
    OOM = "oom"
    TIMEOUT = "timeout"
    NAN_LOSS = "nan_loss"
    DEGENERATE = "degenerate"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# PART 1: PipelineConfig
# ---------------------------------------------------------------------------

SlotName = Literal[
    "data_view",
    "features",
    "weighting",
    "model",
    "objective",
    "calibration",
]

# Order matters: this is the walk order for slot_hash's cascading hash
# (see PipelineConfig.slot_hash below), and it is also the dependency
# order of the pipeline itself — each slot's output feeds the next.
SLOT_ORDER: list[SlotName] = [
    "data_view",
    "features",
    "weighting",
    "model",
    "objective",
    "calibration",
]


@dataclass(frozen=True)
class SlotConfig:
    """One stage of the pipeline (e.g. `model="lightgbm"`)."""

    impl: str  # registry key, e.g. "lightgbm"
    params: dict[str, Any] = field(default_factory=dict)
    # Only set when impl == "custom": freeform code an LLM emitted for
    # this slot. It is deliberately part of the hashed identity (see
    # canonical() below) rather than sitting alongside it, so a custom
    # implementation is content-addressed exactly like a registry impl.
    code_blob: Optional[str] = None

    def canonical(self) -> str:
        """Deterministic string form used as sha256 input by slot_hash.

        sort_keys + compact separators so the same logical config always
        serializes to the same bytes regardless of dict insertion order
        or incidental whitespace.
        """
        payload = {"impl": self.impl, "params": self.params, "code": self.code_blob}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PipelineConfig:
    """Identifies a candidate method: which impl+params fill each slot.

    Blending is deliberately NOT a slot here. It operates over many
    already-completed candidates (picking a weighted ensemble of finished
    runs), not over a single pipeline's data flow, so it lives in the
    controller's finalization step rather than in this per-candidate
    identity.
    """

    slots: dict[SlotName, SlotConfig]
    parent_id: Optional[str] = None
    # `seed` is deliberately EXCLUDED from slot_hash / config_id. The
    # noise gate runs one config across 3-5 seeds to estimate variance;
    # those runs must share a single cached feature matrix (which does
    # not depend on the model's random seed) while still producing
    # distinct, seed-specific results downstream. Concretely: the
    # feature cache is keyed on slot_hash alone, while prediction
    # artifacts are keyed on (config_id, seed). If seed were part of the
    # hash, every seed would rebuild features from scratch.
    seed: int = 0

    def slot_hash(self, upto: SlotName) -> str:
        """sha256 over canonical() of each slot, walking SLOT_ORDER and
        stopping after `upto`. First 12 hex chars.

        This is cascading on purpose: two candidates that differ only in
        `model` produce an identical slot_hash("weighting"), so the
        harness can reuse the already-built feature matrix instead of
        recomputing it. This upstream-reuse property is the main
        performance mechanism in the project — most search steps change
        one downstream slot at a time.
        """
        hasher = hashlib.sha256()
        for slot_name in SLOT_ORDER:
            hasher.update(self.slots[slot_name].canonical().encode("utf-8"))
            if slot_name == upto:
                break
        return hasher.hexdigest()[:12]

    @property
    def config_id(self) -> str:
        return self.slot_hash(SLOT_ORDER[-1])


# ---------------------------------------------------------------------------
# PART 2: CandidateResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """A named bag of metric values for a single (config, seed) run.

    `values` is a plain dict rather than named float fields (e.g.
    `auc: float`, `ndcg: float`) because the organizers' own
    documentation is internally inconsistent about which metrics are
    scored — the prose names one pair, the shipped evaluation script
    computes another. Staying agnostic here means the harness doesn't
    need a contract change once that's sorted out.
    """

    values: dict[str, float]

    @property
    def primary(self) -> float:
        """Unweighted mean across all reported metric values."""
        return sum(self.values.values()) / len(self.values)


@dataclass(frozen=True, kw_only=True)
class CandidateResult:
    """The outcome of running one PipelineConfig for one or more seeds.

    kw_only=True: several fields below have no sensible default
    (config_id, status, val, backtest) while others do, and grouping
    fields by *meaning* rather than by default-vs-required reads better
    here than reordering them to satisfy positional dataclass rules.
    kw_only sidesteps that tension by requiring keyword construction
    everywhere, which is arguably the right call anyway for a dataclass
    with this many same-typed fields (float, float, int, int, int...) —
    positional construction would be a foot-gun.
    """

    config_id: str
    status: Status
    # Per-seed dicts, and they must NEVER be pre-averaged into a single
    # float before reaching this object. The noise gate needs paired
    # comparisons on matched seeds (candidate A seed 3 vs candidate B
    # seed 3, etc.); seed-to-seed noise (sigma ~0.0008) sits close to the
    # acceptance threshold (0.002), and a paired test is far more
    # sensitive to a real effect than comparing two means would be.
    # Averaging here would destroy exactly the information the gate
    # needs and make it unable to tell signal from noise.
    val: dict[int, Metrics]
    # First-class, not Optional — see rationale below. Populated with
    # the same per-seed structure as `val` for the same reason.
    backtest: dict[int, Metrics]

    error_class: ErrorClass = ErrorClass.NONE
    error_excerpt: Optional[str] = None
    val_pred_path: Optional[str] = None
    test_pred_path: Optional[str] = None
    # Cost fields are populated even when status is FAILED. Total token
    # and GPU spend is itself a graded criterion for this project, and a
    # candidate that failed midway still consumed real budget getting
    # there — zeroing these out on failure would understate true cost.
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    repair_attempts: int = 0

    # `backtest` above is required, not Optional, because acceptance
    # requires a change to help on BOTH an internal forward split and
    # real validation. Making it optional would invite skipping it under
    # time pressure — exactly the moment overfitting to validation is
    # most likely and most dangerous. An empty dict (`{}`) is still a
    # legitimate value here — it's the "no backtest was run" case (screen
    # stage, or a candidate that failed before backtesting), kept
    # distinguishable from "ran and produced per-seed metrics" while the
    # field itself stays required so that state can't be quietly skipped
    # by passing None instead. This mirrors Verdict.backtest_delta below,
    # which is None for exactly the same reason at the gate's output.


@dataclass(frozen=True, kw_only=True)
class Verdict:
    """Returned by the noise gate (harness.gate), consumed by the
    controller to decide accept/reject.

    CONTRACT AMENDMENT (post W2 review): added `n_seeds`, and loosened
    `backtest_delta` to Optional. kw_only=True because backtest_delta now
    has a default while `reason` after it doesn't — same rationale as
    CandidateResult above for sidestepping dataclass field-ordering rules
    rather than reordering fields away from their conceptual grouping.
    """

    accept: bool
    delta: float
    ci95: tuple[float, float]
    # A 1-seed screen and a 5-seed confirm are otherwise indistinguishable
    # downstream from delta/ci95 alone. That distinction matters because
    # per-seed noise (sigma ~0.0008) sits close to the acceptance
    # threshold (0.002) — the controller and the journal both need to
    # know how much evidence backed a given verdict, not just what the
    # verdict was.
    n_seeds: int
    # None means "no backtest was run" (screen stage, or a candidate that
    # failed before backtesting ever happened) — distinct from 0.0, which
    # means the backtest ran and showed no change. Collapsing those two
    # into 0.0 would make an untested candidate indistinguishable from a
    # confirmed-neutral one.
    backtest_delta: Optional[float] = None
    reason: str


# ---------------------------------------------------------------------------
# PART 3: JournalEvent
# ---------------------------------------------------------------------------


class EventKind(str, Enum):
    RUN_START = "run_start"
    STAGE_CHANGE = "stage_change"
    HYPOTHESIS = "hypothesis"
    CODE_EMITTED = "code_emitted"
    EVAL_START = "eval_start"
    EVAL_RESULT = "eval_result"
    ERROR = "error"
    REPAIR_ATTEMPT = "repair_attempt"
    RECOVERY = "recovery"
    SLOT_BLOCKED = "slot_blocked"
    DECISION = "decision"
    CONVERGENCE_CHECK = "convergence_check"
    INTERVENTION = "intervention"
    BUDGET_WARNING = "budget_warning"
    FINALIZE = "finalize"

    RUN_END = "run_end"
    """Terminal event of a completed run, emitted however the run ended
    (normally, out of budget, or out of hypotheses).

    Crash-resume depends on replay being able to tell "the log ends
    because the run finished" from "the log ends because the process
    died". Without a terminal event those two are indistinguishable, and
    a resume would either re-run finished work or abandon a live one.
    Budget's own docstring already refers to ending "through a normal
    RUN_END event"; this is the member that reference needed."""


# Keys every journal line must carry. Kept beside JournalEvent's field list
# so the two cannot drift apart silently.
_REQUIRED_EVENT_KEYS: tuple[str, ...] = (
    "ts",
    "run_id",
    "iteration",
    "node",
    "kind",
    "payload",
)

# Journal lines can be long (a payload may embed a traceback or a code blob).
# Error messages quote only the first stretch, enough to identify the line in
# the file without dumping the whole thing into a log or a test failure.
_ERROR_LINE_EXCERPT = 120


class JournalDecodeError(ValueError):
    """Raised by `JournalEvent.from_jsonl` on a line it cannot decode.

    Why this exists rather than letting the underlying error escape: the
    journal is an append-only log whose whole purpose is crash-resume, and
    a run killed mid-write leaves a truncated final line. Replay must be
    able to recognise that case and stop cleanly at the last intact event.
    Previously the failure modes were inconsistent and un-catchable as a
    group — bad JSON raised `json.JSONDecodeError`, a missing field raised
    a bare `KeyError`, and an unrecognised kind raised a bare `ValueError`
    — so a resume loop had to catch three unrelated exception types and
    could not tell "this line is torn" from "this code has a bug".

    Subclasses `ValueError` so existing `except ValueError` handlers keep
    working. Carries `problem` and the offending `line` as attributes for
    callers that want to log or re-raise with more context.
    """

    def __init__(self, problem: str, line: str) -> None:
        self.problem = problem
        self.line = line
        excerpt = line[:_ERROR_LINE_EXCERPT]
        if len(line) > _ERROR_LINE_EXCERPT:
            excerpt += "..."
        super().__init__(f"{problem}; offending line: {excerpt!r}")


@dataclass(frozen=True)
class JournalEvent:
    """One append-only log record.

    Both `iteration` and `node` are logged, even though they move
    together most of the time, because it's still unresolved whether the
    convergence rule ("stop after N iterations without improvement")
    should count "iteration" as a committed revision (a config that was
    accepted and became the new parent) or as every evaluation attempted
    (including rejected ones). Logging both counters lets the log be
    re-rendered under either definition later without rerunning anything.
    """

    ts: str  # ISO8601 UTC
    run_id: str
    iteration: int  # committed revisions
    node: int  # raw evaluations
    kind: EventKind
    payload: dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "ts": self.ts,
                "run_id": self.run_id,
                "iteration": self.iteration,
                "node": self.node,
                "kind": self.kind.value,
                "payload": self.payload,
            },
            sort_keys=True,
        )

    @classmethod
    def from_jsonl(cls, line: str) -> "JournalEvent":
        """Parse one journal line. Inverse of `to_jsonl`.

        Raises `JournalDecodeError` — never a bare `KeyError` — for every
        malformed input, so a crash-resume replay can catch one exception
        type and stop at the last intact event. The accepted on-disk format
        is unchanged: any line that parsed before still parses to an
        identical object.
        """
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JournalDecodeError(f"not valid JSON ({exc.msg})", line) from exc

        if not isinstance(data, dict):
            raise JournalDecodeError(
                f"expected a JSON object, got {type(data).__name__}", line
            )

        missing = [key for key in _REQUIRED_EVENT_KEYS if key not in data]
        if missing:
            raise JournalDecodeError(
                f"missing required key(s): {', '.join(missing)}", line
            )

        try:
            kind = EventKind(data["kind"])
        except ValueError as exc:
            raise JournalDecodeError(
                f"unknown EventKind {data['kind']!r}", line
            ) from exc

        return cls(
            ts=data["ts"],
            run_id=data["run_id"],
            iteration=data["iteration"],
            node=data["node"],
            kind=kind,
            payload=data["payload"],
        )


# ---------------------------------------------------------------------------
# JournalEvent.payload shapes
#
# These are dicts, not dataclasses — the schema below is documented, not
# enforced, since payload shape varies by EventKind and forcing every
# variant through one dataclass (or a union of many) would add ceremony
# without adding safety here.
#
#   HYPOTHESIS:
#     AUTHORITATIVE SHAPE: the `HypothesisPayload` dataclass in PART 4
#     below. The dict sketch here is retained for readability alongside
#     the other payload kinds, but it is no longer the source of truth —
#     if the two ever disagree, the dataclass wins. Build the payload with
#     `dataclasses.asdict(hypothesis)` rather than hand-rolling the dict.
#     {
#       "target_slot": SlotName,
#       "rationale": str,
#       "citation": {"key": str, "url": str, "library_entry": str},
#       "expected_gain": float,
#       "expected_cost_s": float,
#       "predecessor_evidence": [...],
#     }
#     `expected_gain` is the agent's own forecast, logged BEFORE the
#     candidate is evaluated, so forecast calibration (predicted gain vs.
#     realized delta) can be reported after the fact.
#
#   REPAIR_ATTEMPT:
#     {
#       "error_class": ErrorClass,
#       "policy": str,
#       "attempt": int,
#       "change": str,
#       "outcome": str,
#     }
#
#   DECISION:
#     {
#       "verdict": bool,
#       "delta_primary": float,
#       "ci95": [float, float],
#       "backtest_delta": float,
#       "reason": str,
#     }
#
#   INTERVENTION:
#     {
#       "who": str,
#       "type": str,
#       "reason": str,
#       "iteration_affected": int,
#     }
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PART 4: HypothesisPayload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """Where a proposed method comes from.

    All three fields are required rather than optional because the final
    report has to attribute every accepted change to a real, checkable
    technique. An agent that cannot name a source for what it is doing is
    guessing, and a guess that happens to win is not a result we can
    defend to anyone reading the log afterwards.
    """

    key: str  # short bibliographic handle, e.g. "rendle2010fm"
    url: str  # resolvable link to the paper or documentation
    library_entry: str  # id of the matching entry in the methods/ YAML library


@dataclass(frozen=True)
class HypothesisPayload:
    """A proposed change to one slot, emitted by the hypothesis generator.

    Promoted from the documented dict sketch in the payload-shapes comment
    above into a real type. That shape was enforced by nothing, so a
    renamed or dropped key would have surfaced as a KeyError deep inside
    the controller, long after the generator that caused it had returned.
    Field names match the comment exactly, so `dataclasses.asdict()` of
    this object is a drop-in replacement for the hand-built dict and no
    journal reader has to change.

    `predecessor_evidence` is a tuple rather than a list because this is a
    frozen value object and a list field would still be mutable in place;
    it serializes to the same JSON array the comment specifies. It carries
    the config_ids of earlier candidates that motivated this proposal, and
    defaults to empty because the first hypothesis of a run has no
    predecessors.
    """

    target_slot: SlotName
    rationale: str
    citation: Citation

    expected_gain: float
    """Forecast improvement in the primary metric.

    ADVISORY ONLY — it must never gate acceptance. It is logged before the
    candidate is evaluated purely so that afterwards we can score the
    generator's forecast calibration (predicted gain vs. realized delta).
    The moment it influenced a decision it would stop being an honest
    measurement and become a self-fulfilling one: only the noise gate, on
    measured per-seed evidence, decides what is kept. Using it to *order*
    the search queue is fine — that changes what we try first, never what
    we accept.
    """

    expected_cost_s: float
    predecessor_evidence: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# PART 5: Budget
# ---------------------------------------------------------------------------


BUDGET_COUNTER_ORDER: tuple[str, ...] = (
    "wall_seconds",
    "tokens",
    "evaluations",
    "gpu_seconds",
)
"""Fixed reporting order for `Budget`'s counters.

Pinned here rather than derived from `dataclasses.fields()` so the order
in a BUDGET_WARNING journal payload stays stable even if the field
declaration order is ever reshuffled — a run archived today must render
the same way next week.
"""


@dataclass(frozen=True)
class BudgetCounter:
    """One metered resource: how much may be spent, and how much has been.

    `limit` defaults to infinity so a resource the run does not actually
    meter (GPU seconds on a CPU-only box) can be left alone rather than
    forcing every caller to invent a sentinel.
    """

    limit: float = float("inf")
    consumed: float = 0.0

    @property
    def remaining(self) -> float:
        """Headroom left. Goes negative once the limit is overshot, which
        is deliberate: an overshoot is worth seeing in the log rather than
        being clamped to zero and hidden."""
        return self.limit - self.consumed

    @property
    def exhausted(self) -> bool:
        """True at or past the limit.

        `>=`, not `>`: a limit of zero means do not start. Deciding this
        once, here, is the point — an off-by-one on this predicate is the
        difference between stopping cleanly and overrunning the budget the
        run was graded on.
        """
        return self.consumed >= self.limit


@dataclass(frozen=True)
class Budget:
    """The run's spending envelope across every metered resource.

    Held by the controller, which checks it before starting a candidate.
    Exceeding a budget must end the run through a normal RUN_END event,
    never by being killed — an unfinished journal cannot be reported on.

    Frozen like every other value object here, so spending is recorded by
    deriving a new Budget rather than mutating one:

        budget = dataclasses.replace(
            budget,
            evaluations=dataclasses.replace(
                budget.evaluations, consumed=budget.evaluations.consumed + 1
            ),
        )

    That is deliberately more verbose than `+= 1`. It means a Budget
    captured in a journal payload or passed to another component cannot be
    changed underneath the holder, so a BUDGET_WARNING event records what
    was actually true when it was written.
    """

    wall_seconds: BudgetCounter = field(default_factory=BudgetCounter)
    tokens: BudgetCounter = field(default_factory=BudgetCounter)
    evaluations: BudgetCounter = field(default_factory=BudgetCounter)
    gpu_seconds: BudgetCounter = field(default_factory=BudgetCounter)

    @property
    def tripped(self) -> tuple[str, ...]:
        """Names of every counter at or past its limit, in
        BUDGET_COUNTER_ORDER.

        Returns all of them rather than the first, because two limits can
        land on the same evaluation and a report that named only one would
        misattribute why the run stopped.
        """
        return tuple(
            name for name in BUDGET_COUNTER_ORDER if getattr(self, name).exhausted
        )

    @property
    def exhausted(self) -> bool:
        """True when ANY counter is at or past its limit."""
        return bool(self.tripped)
