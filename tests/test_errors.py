"""Tests for executor.errors.classify/policy_for."""

import traceback

from contracts import ErrorClass
from executor.errors import classify, policy_for


def _traceback_text(exc: BaseException) -> str:
    try:
        raise exc
    except type(exc):
        return traceback.format_exc()


def test_not_implemented_error_classifies_as_contract():
    exc = NotImplementedError("no realization implemented for model impl 'lightgbm'")
    assert classify(exc, _traceback_text(exc)) == ErrorClass.CONTRACT


def test_syntax_error_classifies_as_syntax():
    exc = SyntaxError("invalid syntax")
    assert classify(exc, _traceback_text(exc)) == ErrorClass.SYNTAX


def test_memory_error_classifies_as_oom():
    exc = MemoryError("cannot allocate array")
    assert classify(exc, _traceback_text(exc)) == ErrorClass.OOM


def test_out_of_memory_text_classifies_as_oom_even_for_other_exception_types():
    exc = RuntimeError("CUDA error: out of memory")
    assert classify(exc, _traceback_text(exc)) == ErrorClass.OOM


def test_timeout_error_classifies_as_timeout():
    exc = TimeoutError("training exceeded budget")
    assert classify(exc, _traceback_text(exc)) == ErrorClass.TIMEOUT


def test_nan_loss_requires_both_nan_and_loss_in_traceback():
    exc = RuntimeError("loss became nan at epoch 3")
    assert classify(exc, _traceback_text(exc)) == ErrorClass.NAN_LOSS


def test_bare_nan_without_loss_context_is_not_nan_loss():
    exc = RuntimeError("nan encountered in unrelated array")
    assert classify(exc, _traceback_text(exc)) != ErrorClass.NAN_LOSS


def test_import_error_classifies_as_dependency():
    exc = ModuleNotFoundError("No module named 'lightgbm'")
    assert classify(exc, _traceback_text(exc)) == ErrorClass.DEPENDENCY


def test_unrecognized_exception_classifies_as_unknown():
    exc = ValueError("something unrelated went wrong")
    assert classify(exc, _traceback_text(exc)) == ErrorClass.UNKNOWN


def test_policy_for_contract_is_skip_unimplemented():
    assert policy_for(ErrorClass.CONTRACT) == "skip_unimplemented"


def test_policy_for_covers_every_error_class():
    for member in ErrorClass:
        assert isinstance(policy_for(member), str)
