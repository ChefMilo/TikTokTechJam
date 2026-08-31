"""W2-owned autonomy package: wiring that drives the REAL Controller.

Everything here is glue. It imports contracts, controller/, executor/,
harness/ and methods/ READ-ONLY and adds no new contracts, no new
EventKind, and no behaviour those packages do not already have.

The package exists so that "run the audited state machine against the
real executor" is a thing we own end to end, rather than something
approximated by a hand-rolled loop in a script.
"""

from autonomy.adapters import (
    DurableJournal,
    ExecutorAdapterError,
    MovesRealizer,
    RunCandidateExecutor,
    ScriptedMoves,
    SlotScriptedGenerator,
    resolve_fragment,
)
from autonomy.integrity import (
    DEFAULT_SOURCE_DIRS,
    CheckedExecutor,
    IntegrityMetadata,
    IntegrityMonitor,
    RelaunchClassification,
    classify_relaunch,
    code_fingerprint,
    git_state,
    launch_fingerprint,
)

__all__ = [
    "DEFAULT_SOURCE_DIRS",
    "CheckedExecutor",
    "DurableJournal",
    "ExecutorAdapterError",
    "IntegrityMetadata",
    "IntegrityMonitor",
    "MovesRealizer",
    "RelaunchClassification",
    "RunCandidateExecutor",
    "ScriptedMoves",
    "SlotScriptedGenerator",
    "classify_relaunch",
    "code_fingerprint",
    "git_state",
    "launch_fingerprint",
    "resolve_fragment",
]
