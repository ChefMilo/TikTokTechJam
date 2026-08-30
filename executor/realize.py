"""Turns a PipelineConfig into trained-model predictions.

SCOPE DISCIPLINE: this is the minimum spine to get one real candidate
running end to end, not the full executor. No sandbox, no subprocess
isolation, no error taxonomy, no journal — see EXECUTOR_SURVEY.md for
what comes after this. Only `model.impl == "fm"` is realized today;
every other slot's impl is recorded on the config (for a correct
config_id and for provenance) but not yet dispatched on by anything
here, since no other move implementation exists yet either.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from contracts import PipelineConfig, SlotConfig, SlotName

REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_DIR = REPO_ROOT / "vendor" / "kuairand-starter-kit"

DEFAULT_SLOTS: dict[SlotName, SlotConfig] = {
    "data_view": SlotConfig(impl="full"),
    "features": SlotConfig(impl="baseline_5"),
    "weighting": SlotConfig(impl="none"),
    "model": SlotConfig(impl="fm", params={"k": 16, "lr": 0.001}),
    "objective": SlotConfig(impl="bce"),
    "calibration": SlotConfig(impl="none"),
}
"""A complete six-slot default reproducing the vendor's own FM baseline.

A hypothesis generator move (see methods/scripted.py) only ever proposes
a fragment for ONE slot; this fills in the other five so there is always
a complete, hashable PipelineConfig to run.
"""


def build_config(fragment: SlotConfig, target_slot: SlotName, seed: int) -> PipelineConfig:
    """Overlays `fragment` onto DEFAULT_SLOTS at `target_slot`. Every
    other slot keeps its default.
    """
    slots = dict(DEFAULT_SLOTS)
    slots[target_slot] = fragment
    return PipelineConfig(slots=slots, seed=seed)


def _load_vendor_baseline():
    """Imports vendor/kuairand-starter-kit/baseline.py by file path — same
    approach as harness/data.py, harness/metrics.py, and
    tests/test_rungs.py's baseline import. baseline.py's own top-level
    `from data import ...` / `from evaluate import ...` rely on Python's
    normal sys.path search, so the vendor directory is put on sys.path
    just long enough to exec it.
    """
    vendor_dir_str = str(_VENDOR_DIR)
    sys.path.insert(0, vendor_dir_str)
    try:
        spec = importlib.util.spec_from_file_location(
            "_vendor_kuairand_baseline", _VENDOR_DIR / "baseline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(vendor_dir_str)
    return module


_vendor = _load_vendor_baseline()


def _realize_fm(model_slot: SlotConfig, fit_rows: list[tuple], score_rows: list[tuple], seed: int):
    """Trains vendor's FM on `fit_rows`, early-stopping and scoring on
    `score_rows`, and returns (user_ids, labels, scores) aligned to
    `score_rows`, in `score_rows` order.

    WHAT WE FOUND ABOUT run_fm (investigated as requested, since we need
    to run this on backtest windows too, not just the official split):

    1. run_fm's dict KEYS are hardcoded ('train'/'valid'/'test' —
       vendor/kuairand-starter-kit/baseline.py does
       `Xtr,... = enc['train']`, `Xva,... = enc['valid']`,
       `Xte,... = enc['test']` verbatim) but the VALUES are not tied to
       any date range — it only ever touches whatever row lists the
       caller placed under those keys. So it CAN be reused for arbitrary
       fit/score row lists by relabeling: `fit_rows` under 'train',
       `score_rows` under 'valid'. It also unconditionally destructures
       `enc['test']`, so a 'test' entry must be present even though nothing
       reads its result; we reuse `score_rows` there rather than ever
       touching the real hidden test split (which `data.load("test")`
       would refuse to return anyway).

    2. THIS IS NOT WHY WE DON'T CALL run_fm() DIRECTLY, THOUGH. The real
       blocker: run_fm's return value only ever exposes vendor's own
       aggregated evaluate() dict —
       `return {'valid': evaluate(uva, yva, m.predict(Xva)), 'test': ...}`
       — the trained model `m` and the raw predictions `m.predict(Xva)`
       are local variables, never returned. We need the raw
       (user_ids, labels, scores) triple (harness.metrics.evaluate and
       cache.save_predictions both need it), which run_fm's signature
       cannot give us at all, regardless of the key-relabeling above. So
       this function reimplements run_fm's training loop inline —
       same hyperparameters, same early-stopping rule — using
       `encode()` and the `FM` class directly, stopping short of
       run_fm's own final `evaluate()` call so the raw scores survive.

    Using `score_rows` for BOTH early-stopping model selection AND final
    scoring is not a leak introduced by this reuse — it's the same
    pattern run_fm already uses for the official train/valid split
    (early-stop on 'valid', then score 'valid' again), which is standard
    practice. It's also exactly what the backtest is supposed to mirror:
    the same train-then-evaluate-on-a-later-window methodology real
    validation uses, just carved out of the training data itself.
    """
    k = model_slot.params.get("k", 16)
    lr = model_slot.params.get("lr", 0.001)
    epochs = model_slot.params.get("epochs", 40)
    bs = model_slot.params.get("bs", 8192)
    patience = model_slot.params.get("patience", 4)

    splits = {"train": fit_rows, "valid": score_rows, "test": score_rows}
    enc, dim = _vendor.encode(splits)
    x_train, y_train, _ = enc["train"]
    x_score, y_score, user_ids_score = enc["valid"]

    model = _vendor.FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary, best_state, bad = -1.0, None, 0
    for _ in range(1, epochs + 1):
        order = rng.permutation(len(y_train))
        for i in range(0, len(order), bs):
            batch = order[i : i + bs]
            model.step(x_train[batch], y_train[batch])
        score_result = _vendor.evaluate(user_ids_score, y_score, model.predict(x_score))
        if score_result["primary"] > best_primary + 1e-5:
            best_primary, bad = score_result["primary"], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    model.V, model.W, model.b = best_state

    scores = model.predict(x_score)
    return user_ids_score, y_score, scores


def realize(config: PipelineConfig, fit_rows: list[tuple], score_rows: list[tuple], seed: int):
    """Trains `config` on `fit_rows` and returns (user_ids, labels,
    scores) aligned to `score_rows`, in `score_rows` order.

    Dispatches on config.slots["model"].impl. Only "fm" is implemented
    today — any other impl raises NotImplementedError naming it. That is
    correct behaviour, not a gap to silently paper over: nothing else has
    a realization yet.
    """
    model_slot = config.slots["model"]
    if model_slot.impl == "fm":
        return _realize_fm(model_slot, fit_rows, score_rows, seed)
    raise NotImplementedError(
        f"executor.realize: no realization implemented for model impl {model_slot.impl!r}"
    )
