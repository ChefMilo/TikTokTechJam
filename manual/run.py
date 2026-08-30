"""The manual ceiling's standalone train/predict/score loop.

Mirrors executor/run.py's structure — per seed, a validation pass and a
backtest pass, assembled into a contracts.CandidateResult — but reaches
the vendored FM directly instead of going through executor.realize. See
manual/__init__.py on why that boundary exists and what it costs.

WHAT IS COPIED AND WHAT IS NEW
------------------------------
The training loop below is a transcription of the vendor's own `run_fm`:
same epoch structure, same per-epoch reshuffle, same early-stopping rule
(best score-window primary, `patience` epochs without improvement, then
restore the best weights). It is transcribed rather than called because
`run_fm` returns only aggregated metric dicts — the trained model and its
raw predictions are local variables it never hands back — and this
pipeline needs the raw (user_ids, labels, scores) triple, both to score
through harness.metrics and to cache for the noise gate's user-level
bootstrap.

The only genuinely NEW logic in this module's path is manual/encode.py's
field-parameterized encoder. Everything else is the vendor's, moved.

SCORES ARE RAW LOGITS, NEVER SIGMOIDED. `FM.predict` returns logits, and
both scored metrics (GAUC, nDCG@5) are rank-based, so a monotone squash
would change nothing about the score while quietly making the vectors
incomparable with the executor's cached ones and with each other at blend
time. Left alone deliberately.

WHY THIS RAISES INSTEAD OF RETURNING A FAILED CandidateResult.
executor/run.py swallows every exception because the Controller must
survive one bad candidate. Nothing here is running a search: this is an
experiment a human launches and reads. A traceback naming the missing CSV
is strictly more useful to that human than a Status.FAILED object they
have to unpack, so failures propagate.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Optional, Sequence

import numpy as np

from contracts import CandidateResult, Metrics, Status
from harness import backtest, cache, data, metrics
from manual._vendor import vendor
from manual.encode import (
    BASELINE_FIELDS,
    CROSS_SPECS,
    CROSS_USER_COLUMNS,
    CROSS_VIDEO_COLUMNS,
    CrossSpec,
    FieldSpec,
    SideTables,
    encode,
)

MANUAL_BASELINE_CONFIG_ID = "manual_baseline_fm_k16"
"""Stable, human-readable id for the manual baseline's cached predictions.

DELIBERATELY NOT the executor's `bce19171850a`. That id is a content hash
over a PipelineConfig, and matching it would mean reconstructing W3's
exact DEFAULT_SLOTS — coupling this package to a W3 constant it is not
allowed to import, and one that changes whenever a slot default changes.

Owning our own id costs one extra baseline training run and buys a
matched-seed comparison that is entirely ours: unit 2's variants are
gate-compared against THIS baseline, on these seeds, with both sides'
predictions cached under ids this package controls. A hash would also be
opaque in `artifacts/preds/`; this is legible at a glance.
"""

MANUAL_CROSSES_CONFIG_ID = "manual_crosses_v1"
"""Id for the crosses variant's cached predictions.

Versioned (`_v1`) because the cross SET is part of what is being measured:
if CROSS_SPECS changes, the cached vectors under this id no longer describe
the same candidate, and a comparison that silently reused them would be
comparing two different features under one name. Bump the suffix when the
set changes.
"""

BASELINE_HYPERPARAMS: dict[str, Any] = {
    "k": 16,
    "lr": 0.001,
    "epochs": 40,
    "bs": 8192,
    "patience": 4,
}
"""The organizers' published FM settings, verbatim from the vendor's
`run_fm` defaults. This is the incumbent the ceiling is measured against,
so it is reproduced exactly rather than re-tuned — a "baseline" that
quietly used better hyperparameters would understate the headroom it
exists to measure.
"""


def train_and_score(
    fit_rows: list[tuple],
    score_rows: list[tuple],
    seed: int,
    fields: Sequence[FieldSpec] = BASELINE_FIELDS,
    hyperparams: Optional[dict[str, Any]] = None,
    crosses: Sequence[CrossSpec] = (),
    side_tables: Optional[SideTables] = None,
):
    """Trains an FM on `fit_rows`, scores `score_rows`, returns the triple.

    Returns `(user_ids, labels, scores)` aligned to `score_rows`, in
    `score_rows` order — the shape harness.metrics.evaluate and
    harness.cache.save_predictions both consume.

    `score_rows` is used for BOTH early-stopping selection and final
    scoring. That is the vendor's own pattern (run_fm early-stops on
    'valid' then reports 'valid'), and it is what the backtest is meant to
    mirror; it is not a leak introduced here.

    Seeding is two-channel and both channels matter: `seed` initialises
    the FM's embedding matrix AND drives the per-epoch shuffle, through
    two independent generators, exactly as the vendor does.
    """
    settings = dict(BASELINE_HYPERPARAMS)
    settings.update(hyperparams or {})
    epochs = int(settings["epochs"])
    batch_size = int(settings["bs"])
    patience = int(settings["patience"])

    if epochs < 1:
        # The best-weight restore below reads `best_state`, which only
        # exists after an epoch has been scored. Named explicitly rather
        # than left to fail as `TypeError: cannot unpack None`.
        raise ValueError(f"epochs must be at least 1, got {epochs}")

    splits = {"train": fit_rows, "score": score_rows}
    enc, dim = encode(
        splits,
        fields=fields,
        train_key="train",
        crosses=crosses,
        side_tables=side_tables,
    )
    x_fit, y_fit, _ = enc["train"]
    x_score, y_score, user_ids_score = enc["score"]

    model = vendor.FM(dim, k=settings["k"], lr=settings["lr"], seed=seed)
    rng = np.random.default_rng(seed)

    best_primary, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        order = rng.permutation(len(y_fit))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            model.step(x_fit[batch], y_fit[batch])

        scored = metrics.evaluate(user_ids_score, y_score, model.predict(x_score))
        # `+ 1e-5` is the vendor's own improvement threshold: it stops a
        # run of floating-point-noise "improvements" from resetting the
        # patience counter forever.
        if scored.primary > best_primary + 1e-5:
            best_primary, bad = scored.primary, 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break

    model.V, model.W, model.b = best_state
    return user_ids_score, y_score, model.predict(x_score)


def load_side_tables() -> SideTables:
    """The user and video side-attribute lookups the crosses read.

    Goes through harness.data.load_side_features(), which is the only
    sanctioned door to those two CSVs — reading them here directly would
    duplicate a harness responsibility and put a second, unguarded file
    path in this package.

    Only the columns CROSS_SPECS actually names are retained; see
    SideTables.from_frames.
    """
    user_frame, video_frame = data.load_side_features()
    return SideTables.from_frames(
        user_frame, video_frame, CROSS_USER_COLUMNS, CROSS_VIDEO_COLUMNS
    )


def run_variant(
    config_id: str,
    seeds: Sequence[int] = (0, 1, 2),
    fields: Sequence[FieldSpec] = BASELINE_FIELDS,
    hyperparams: Optional[dict[str, Any]] = None,
    crosses: Sequence[CrossSpec] = (),
    side_tables: Optional[SideTables] = None,
) -> CandidateResult:
    """Runs one variant across `seeds` and returns its result.

    Per seed: a validation pass (fit on data.load("train"), score
    data.load("val")) whose predictions are CACHED, and a backtest pass
    over harness.backtest.split()'s fit/score windows, which are not.

    WHY VALIDATION PREDICTIONS ARE CACHED AND BACKTEST ONES ARE NOT.
    harness.gate's strong user-level bootstrap reads cached VALIDATION
    predictions for both sides on every matched seed, and silently
    degrades to a much weaker seed-level bootstrap (~12.5% false-positive
    rate) when any are missing. The backtest contributes only a scalar
    `backtest_delta`, derived from the per-seed Metrics that are already
    in the returned object, so caching those vectors would cost disk and
    buy nothing. Same split of responsibilities executor/run.py makes.

    The row loads happen ONCE, outside the seed loop: they are the same
    rows for every seed, and harness.data memoizes the raw frame anyway.
    """
    started = time.perf_counter()

    train_rows = data.load("train")
    val_rows = data.load("val")
    fit_rows, score_rows = backtest.split()

    val_metrics: dict[int, Metrics] = {}
    backtest_metrics: dict[int, Metrics] = {}

    for seed in seeds:
        user_ids, labels, scores = train_and_score(
            train_rows,
            val_rows,
            seed,
            fields=fields,
            hyperparams=hyperparams,
            crosses=crosses,
            side_tables=side_tables,
        )
        val_metrics[seed] = metrics.evaluate(user_ids, labels, scores)
        cache.save_predictions(config_id, seed, "val", user_ids, labels, scores)

        bt_user_ids, bt_labels, bt_scores = train_and_score(
            fit_rows,
            score_rows,
            seed,
            fields=fields,
            hyperparams=hyperparams,
            crosses=crosses,
            side_tables=side_tables,
        )
        backtest_metrics[seed] = metrics.evaluate(bt_user_ids, bt_labels, bt_scores)

    return CandidateResult(
        config_id=config_id,
        status=Status.OK,
        val=val_metrics,
        backtest=backtest_metrics,
        # Informational only, and a pattern rather than a literal path —
        # one string cannot name one file per seed. harness.gate's real
        # test for "are predictions cached" is cache.exists() per seed.
        val_pred_path=f"artifacts/preds/{config_id}__<seed>__val.npz",
        wall_seconds=time.perf_counter() - started,
    )


def run_baseline(
    seeds: Sequence[int] = (0, 1, 2),
    fields: Sequence[FieldSpec] = BASELINE_FIELDS,
    hyperparams: Optional[dict[str, Any]] = None,
    config_id: str = MANUAL_BASELINE_CONFIG_ID,
) -> CandidateResult:
    """The organizers' five fields and nothing else. The incumbent."""
    return run_variant(
        config_id=config_id, seeds=seeds, fields=fields, hyperparams=hyperparams
    )


def run_crosses(
    seeds: Sequence[int] = (0, 1, 2),
    fields: Sequence[FieldSpec] = BASELINE_FIELDS,
    hyperparams: Optional[dict[str, Any]] = None,
    config_id: str = MANUAL_CROSSES_CONFIG_ID,
    crosses: Sequence[CrossSpec] = CROSS_SPECS,
    side_tables: Optional[SideTables] = None,
) -> CandidateResult:
    """The baseline five PLUS three user-side x item-side cross columns.

    Strictly additive: the five baseline columns keep the ids they had, and
    the crosses widen X and the embedding table. So the only difference
    between this and the incumbent is the three new columns — which is what
    makes the gate's verdict attributable to them and nothing else.

    The raw side features are deliberately NOT added as their own columns;
    see CrossSpec's docstring on why that experiment is already settled.
    """
    if side_tables is None:
        side_tables = load_side_tables()
    return run_variant(
        config_id=config_id,
        seeds=seeds,
        fields=fields,
        hyperparams=hyperparams,
        crosses=crosses,
        side_tables=side_tables,
    )


def baseline_incumbent(
    seeds: Sequence[int] = (0, 1, 2),
    hyperparams: Optional[dict[str, Any]] = None,
) -> tuple[CandidateResult, str]:
    """The baseline CandidateResult to compare a variant against.

    Returns `(result, provenance)` — the provenance string is printed, so
    a reader of the run log can always tell whether the incumbent's numbers
    were re-measured or reconstructed.

    THE VALIDATION HALF COMES FROM CACHE when every seed's predictions are
    on disk: the vectors are the model's actual output, so re-scoring them
    through harness.metrics reproduces the exact Metrics the original run
    reported, for the price of an evaluate() call instead of a 75-second
    training pass.

    THE BACKTEST HALF CANNOT, and this is worth being explicit about.
    run_variant caches validation predictions only — harness.gate's
    user-level bootstrap needs those and nothing else — so there are no
    backtest vectors to re-score. But `gate.compare` computes
    `backtest_delta` from both sides' `.backtest` Metrics, and an incumbent
    with an empty backtest yields `backtest_delta=None`, which the gate
    reports as `backtest_missing` and refuses to accept on. Reconstructing
    the val half and leaving the backtest half empty would therefore
    produce a guaranteed-reject verdict that says nothing about the
    candidate.

    So the backtest passes are re-run — three trainings rather than six.
    Hardcoding the previously observed numbers instead was never an option:
    a measurement that is typed in rather than measured is not a
    measurement.
    """
    cached = all(cache.exists(MANUAL_BASELINE_CONFIG_ID, seed, "val") for seed in seeds)
    if not cached:
        result = run_baseline(seeds=seeds, hyperparams=hyperparams)
        return result, "full re-run (cached validation predictions were absent)"

    started = time.perf_counter()
    val_metrics: dict[int, Metrics] = {}
    for seed in seeds:
        user_ids, labels, scores = cache.load_predictions(
            MANUAL_BASELINE_CONFIG_ID, seed, "val"
        )
        val_metrics[seed] = metrics.evaluate(user_ids, labels, scores)

    fit_rows, score_rows = backtest.split()
    backtest_metrics: dict[int, Metrics] = {}
    for seed in seeds:
        bt_user_ids, bt_labels, bt_scores = train_and_score(
            fit_rows, score_rows, seed, hyperparams=hyperparams
        )
        backtest_metrics[seed] = metrics.evaluate(bt_user_ids, bt_labels, bt_scores)

    result = CandidateResult(
        config_id=MANUAL_BASELINE_CONFIG_ID,
        status=Status.OK,
        val=val_metrics,
        backtest=backtest_metrics,
        val_pred_path=(
            f"artifacts/preds/{MANUAL_BASELINE_CONFIG_ID}__<seed>__val.npz"
        ),
        wall_seconds=time.perf_counter() - started,
    )
    return result, (
        "validation re-scored from cached predictions; backtest re-run "
        "(unit 1 caches validation vectors only)"
    )


VARIANTS = {"baseline": run_baseline, "crosses": run_crosses}
"""Variant name -> runner. Unit 3 adds "blend".

A dict rather than an if/elif so adding a variant is one entry and the
CLI's `choices` stays in sync automatically.
"""


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Split out from main() so the argument contract is testable without
    running a multi-minute training job."""
    parser = argparse.ArgumentParser(
        prog="python -m manual.run",
        description=(
            "Run the manual ceiling pipeline and print its report. Requires "
            "the KuaiRand CSVs under data/."
        ),
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=sorted(VARIANTS),
        help="which manual pipeline to run (default: baseline)",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help=(
            "comma-separated seeds (default: 0,1,2). Three or more is what "
            "harness.gate needs for a CONFIRM-stage verdict."
        ),
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        choices=["baseline"],
        help=(
            "after running, gate.compare the result against this incumbent "
            "and print the Verdict. The incumbent's validation numbers come "
            "from cached predictions when available."
        ),
    )
    return parser.parse_args(argv)


def parse_seeds(raw: str) -> tuple[int, ...]:
    """"0,1,2" -> (0, 1, 2). Rejects an empty list loudly.

    A run with no seeds would train nothing, return an empty
    CandidateResult, and print a report full of zeros — which looks like a
    result rather than like a mistake.
    """
    seeds = tuple(int(part) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError(f"no seeds parsed from {raw!r}; expected e.g. '0,1,2'")
    return seeds


def main(argv: Optional[Sequence[str]] = None) -> CandidateResult:
    from manual import report

    args = _parse_args(argv)
    seeds = parse_seeds(args.seeds)

    print(f"manual ceiling: variant={args.variant!r} seeds={seeds}")
    result = VARIANTS[args.variant](seeds=seeds)
    report.print_candidate_report(result, label=f"manual {args.variant}")

    if args.compare_to is not None:
        if args.compare_to == args.variant:
            raise ValueError(
                f"--compare-to {args.compare_to!r} matches --variant; a "
                "candidate cannot be its own incumbent"
            )
        incumbent, provenance = baseline_incumbent(seeds=seeds)
        print()
        print(f"incumbent ({args.compare_to}): {provenance}")
        report.print_candidate_report(
            incumbent, label=f"manual {args.compare_to} (incumbent)"
        )
        report.print_comparison(
            result, incumbent, label=f"{args.variant} vs {args.compare_to}"
        )

    return result


if __name__ == "__main__":
    main()
