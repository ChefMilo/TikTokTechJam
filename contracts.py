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
    # most likely and most dangerous.


@dataclass(frozen=True)
class Verdict:
    """Returned by the noise gate (harness.gate), consumed by the
    controller to decide accept/reject.
    """

    accept: bool
    delta: float
    ci95: tuple[float, float]
    backtest_delta: float
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
        data = json.loads(line)
        return cls(
            ts=data["ts"],
            run_id=data["run_id"],
            iteration=data["iteration"],
            node=data["node"],
            kind=EventKind(data["kind"]),
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
