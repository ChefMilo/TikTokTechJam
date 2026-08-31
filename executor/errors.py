"""A minimal error taxonomy over contracts.ErrorClass — classify() maps a
caught exception to a class, POLICY maps a class to a repair strategy
name.

HONESTY NOTE: only ErrorClass.CONTRACT (policy "skip_unimplemented") is
actually exercised anywhere in this codebase today — every unimplemented
slot/impl combination in executor/realize.py raises NotImplementedError,
and that is the only failure mode scripts/run_agent.py's ten scripted
moves actually produce. SYNTAX, OOM, TIMEOUT, NAN_LOSS, and DEPENDENCY
are classified correctly if they occur (each has a real, narrow
detection rule below, not a guess), but nothing in this project has ever
raised a real SyntaxError, MemoryError, TimeoutError, NaN-loss, or
ImportError during a run — there is no retry loop, no lowered-LR resume,
no simplified-config resume, and no revert-and-block mechanism
implemented anywhere. Their POLICY entries are declared for completeness
(so the mapping exists on the day retry logic gets built) and are
UNREACHED code, not tested code. Do not read their presence as evidence
they were exercised — see SCOPE DISCIPLINE below.

SCOPE DISCIPLINE: no sandbox, no subprocess isolation, no memory caps,
no actual retry/repair loop. Building any of those now would be
unexercised code — this run only ever hits CONTRACT errors (six of the
ten scripted moves) and ordinary success/gate-rejection (the other
four), so that is all that's implemented.
"""

from __future__ import annotations

import re

from contracts import ErrorClass

_NAN_INF_RE = re.compile(r"\b(nan|inf)\b", re.IGNORECASE)
_OOM_TEXT_RE = re.compile(r"out of memory", re.IGNORECASE)


def classify(exc: BaseException, traceback_text: str) -> ErrorClass:
    """Classifies `exc` (with `traceback_text` = traceback.format_exc()
    from the same except block) into an ErrorClass.

    Order matters: NotImplementedError is checked before the generic
    text-based rules below it, since a str(exc) for a NotImplementedError
    naming a slot/impl (e.g. "no realization implemented for model impl
    'lightgbm'") could otherwise coincidentally match one of them.
    """
    if isinstance(exc, NotImplementedError):
        return ErrorClass.CONTRACT
    if isinstance(exc, SyntaxError):
        return ErrorClass.SYNTAX
    if isinstance(exc, MemoryError) or _OOM_TEXT_RE.search(traceback_text):
        return ErrorClass.OOM
    if isinstance(exc, TimeoutError):
        return ErrorClass.TIMEOUT
    # "in loss context": nan/inf alone is too broad (e.g. a stray "info"
    # substring or an unrelated numeric literal) — require "loss" to
    # appear in the same traceback too, so this only fires for what it's
    # named for: a diverged training loss, not any NaN anywhere.
    if _NAN_INF_RE.search(traceback_text) and "loss" in traceback_text.lower():
        return ErrorClass.NAN_LOSS
    if isinstance(exc, ImportError):  # covers ModuleNotFoundError too
        return ErrorClass.DEPENDENCY
    return ErrorClass.UNKNOWN


# Repair policy per class. Only "skip_unimplemented" (CONTRACT) is ever
# invoked by anything in this codebase — see module docstring. The rest
# are the honest, unimplemented "next step" for a class that hasn't
# occurred yet, not dead branches to be trusted as working.
POLICY: dict[ErrorClass, str] = {
    ErrorClass.CONTRACT: "skip_unimplemented",  # IMPLEMENTED — see executor/run.py, scripts/run_agent.py
    ErrorClass.NAN_LOSS: "lower_lr_retry",  # UNREACHED — no retry loop exists
    ErrorClass.OOM: "simplify_retry",  # UNREACHED — no retry loop exists
    ErrorClass.TIMEOUT: "simplify_retry",  # UNREACHED — no retry loop exists
    ErrorClass.SYNTAX: "revert_and_block",  # UNREACHED — nothing generates code to have a syntax error
    ErrorClass.DEPENDENCY: "revert_and_block",  # UNREACHED — no optional/generated imports exist yet
    ErrorClass.DEGENERATE: "revert_and_block",  # UNREACHED — no degenerate-output detector exists yet
    ErrorClass.UNKNOWN: "revert_and_block",  # the honest fallback for anything the rules above don't name
    ErrorClass.NONE: "revert_and_block",  # never looked up in practice — NONE means no error occurred
}


def policy_for(error_class: ErrorClass) -> str:
    """Returns POLICY[error_class], or "revert_and_block" for any class
    not in the table (there shouldn't be one, since POLICY covers every
    ErrorClass member, but a missing key should never raise here — a
    journal entry for an unrecognised policy is better than a crash
    while trying to record a different failure).
    """
    return POLICY.get(error_class, "revert_and_block")
