"""run_candidate(): executes ONE PipelineConfig fragment end to end and
returns a contracts.CandidateResult.

SCOPE DISCIPLINE: minimum spine only. No sandbox, no subprocess
isolation. A failure is classified (executor.errors.classify) but not
repaired — see executor/errors.py's module docstring for exactly which
classes/policies are actually exercised versus declared for later. See
EXECUTOR_SURVEY.md for what comes after.
"""

from __future__ import annotations

import time
import traceback
from typing import Optional, Sequence

from contracts import CandidateResult, Metrics, SlotConfig, SlotName, Status
from executor import errors as errors_module
from executor import realize as realize_module
from executor.journal import Journal
from harness import backtest, cache, data, metrics


def run_candidate(
    fragment: SlotConfig,
    target_slot: SlotName,
    seeds: Sequence[int] = (0, 1, 2),
    journal: Optional[Journal] = None,
) -> CandidateResult:
    """Realizes `fragment` (a proposed change to `target_slot`) across
    `seeds`: a validation pass (train on data.load("train"), score on
    data.load("val")) and a backtest pass (harness.backtest.split()'s
    fit/score windows), for each seed.

    `journal` is optional and defaults to None so existing callers (the
    I1 smoke test in particular) keep working unchanged. When given, one
    EVAL_START is logged before training and one EVAL_RESULT after
    (or ERROR on failure) — both sharing the same node number, computed
    once up front, so a reader can tell they describe the same attempt
    regardless of whether it was accepted.

    Never raises. A failure anywhere becomes
    CandidateResult(status=Status.FAILED, error_excerpt=...) — one bad
    candidate must not take the whole run down with it, and an exception
    escaping here would also lose whatever the caller wanted to journal
    about this attempt. `error_class` comes from executor.errors.classify
    — see that module's docstring for which classes are actually
    exercised versus declared for a repair loop that doesn't exist yet.
    """
    start = time.perf_counter()
    config_id = "unknown"
    node = journal.current_node + 1 if journal is not None else None
    eval_started = False
    try:
        train_rows = data.load("train")
        val_rows = data.load("val")
        fit_rows, score_rows = backtest.split()

        val_metrics: dict[int, Metrics] = {}
        backtest_metrics: dict[int, Metrics] = {}

        for seed in seeds:
            config = realize_module.build_config(fragment, target_slot, seed)
            if config_id == "unknown":
                # config_id excludes `seed` by construction (see
                # contracts.PipelineConfig.seed) — identical across every
                # seed in this loop, so the first one is as good as any.
                config_id = config.config_id
                if journal is not None:
                    journal.log_eval_start(config_id, node=node)
                    eval_started = True

            # --- validation pass ---
            user_ids, labels, scores = realize_module.realize(config, train_rows, val_rows, seed)
            val_metrics[seed] = metrics.evaluate(user_ids, labels, scores)

            # THE SINGLE MOST IMPORTANT LINE IN THIS FILE. Without it,
            # harness/gate.py silently falls back to a bootstrap over
            # just the 3-5 per-seed deltas, which has a ~12.5%
            # false-positive rate — and the run still completes and
            # looks completely normal. See harness/HANDOFF.md.
            cache.save_predictions(config_id, seed, "val", user_ids, labels, scores)

            # --- backtest pass ---
            bt_user_ids, bt_labels, bt_scores = realize_module.realize(
                config, fit_rows, score_rows, seed
            )
            backtest_metrics[seed] = metrics.evaluate(bt_user_ids, bt_labels, bt_scores)

            # Mirrors the val save above, same reason: without this the
            # gate has no cached backtest predictions to rank-average or
            # otherwise recombine downstream (e.g. an ensemble candidate),
            # and anything built on top degrades silently rather than
            # loudly — there is no "backtest_missing"-equivalent warning
            # for a caller who reads cache.load_predictions("backtest")
            # expecting it to be there and finds nothing.
            cache.save_predictions(config_id, seed, "backtest", bt_user_ids, bt_labels, bt_scores)

        wall_seconds = time.perf_counter() - start
        # No GPU, no LLM anywhere in this pipeline today — both are
        # genuinely 0.0/0, not unknown. Read from a CandidateResult built
        # below rather than hand-summed here, so the journal and the
        # returned object can never disagree about what this candidate cost.
        result = CandidateResult(
            config_id=config_id,
            status=Status.OK,
            val=val_metrics,
            backtest=backtest_metrics,
            # Informational only — harness.gate's real ground truth for
            # "are predictions cached?" is cache.exists(config_id, seed,
            # "val") per seed, not this field (see harness/gate.py's
            # _confirm). A single string can't represent one path per
            # seed, so this is the naming pattern, not a literal path.
            val_pred_path=f"artifacts/preds/{config_id}__<seed>__val.npz",
            wall_seconds=wall_seconds,
        )
        if journal is not None:
            journal.log_eval_result(
                config_id,
                val_metrics,
                wall_seconds,
                backtest_per_seed_metrics=backtest_metrics,
                target_slot=target_slot,
                fragment_impl=fragment.impl,
                fragment_params=fragment.params,
                gpu_seconds=result.gpu_seconds,
                tokens=result.tokens_in + result.tokens_out,
                node=node,
            )

        return result
    except Exception as exc:  # noqa: BLE001 - must never escape, see docstring
        wall_seconds = time.perf_counter() - start
        error_class = errors_module.classify(exc, traceback.format_exc())
        policy = errors_module.policy_for(error_class)
        if journal is not None:
            if not eval_started:
                # Nothing was logged yet for this attempt (the failure
                # happened before config_id was even known) — still worth
                # a node of its own so the failure shows up in the log.
                journal.log_eval_start(config_id, node=node)
            journal.log_error(error_class, repr(exc)[:2000], policy=policy, node=node)
        return CandidateResult(
            config_id=config_id,
            status=Status.FAILED,
            val={},
            backtest={},
            error_class=error_class,
            error_excerpt=repr(exc)[:2000],
            wall_seconds=wall_seconds,
        )
