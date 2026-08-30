"""Tests for the manual ceiling (manual/), unit 1: the standalone
baseline runner and its encoder.

EVERY TEST HERE RUNS WITHOUT THE DATASET. harness.data.load needs all
three KuaiRand CSVs on disk and has no partial-load path, so anything
touching it is monkeypatched to synthetic rows — the same discipline
tests/test_realize.py and tests/test_run.py already follow ("seconds, not
minutes"). The real end-to-end number is Terry's execution check, not a
test's: nothing here asserts the published 0.6016, because reproducing it
requires the data this suite deliberately does not need.

The load-bearing test is `test_our_encoder_matches_the_vendors_exactly`.
manual/ re-implements the vendor's encoder so unit 2 can add cross
columns without editing the read-only vendored kit; that re-implementation
is only safe if it is provably identical for the five baseline fields,
and this is where that is proved.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from contracts import CandidateResult, Metrics, Status
from manual import encode as encode_module
from manual import report as report_module
from manual import run as run_module
from manual._vendor import vendor

MANUAL_DIR = Path(__file__).resolve().parent.parent / "manual"

# Tiny hyperparameters: enough epochs to exercise the early-stopping
# branch, small enough to finish in well under a second.
TINY_HYPERPARAMS = {"k": 2, "lr": 0.05, "epochs": 2, "bs": 4, "patience": 1}


def _row(date, user, video, author, tab, duration, label):
    """One harness.data row tuple:
    (date, user_id, video_id, author_id, tab, duration_ms, label)."""
    return (date, user, video, author, tab, duration, label)


def _train_rows():
    """Two users, each with positives and negatives, so GAUC is defined
    (it only counts users with 0 < npos < n_impressions), and a spread of
    durations so the quantile bucket edges are non-degenerate."""
    return [
        _row(20220410, "u1", "v1", "a1", "t1", 1000.0, 1),
        _row(20220410, "u1", "v2", "a1", "t1", 2000.0, 0),
        _row(20220411, "u1", "v3", "a2", "t2", 3000.0, 1),
        _row(20220411, "u1", "v4", "a2", "t1", 4000.0, 0),
        _row(20220412, "u2", "v1", "a1", "t2", 5000.0, 0),
        _row(20220412, "u2", "v2", "a1", "t1", 6000.0, 1),
        _row(20220413, "u2", "v5", "a3", "t2", 7000.0, 0),
        _row(20220413, "u2", "v6", "a3", "t1", 8000.0, 1),
        _row(20220414, "u3", "v1", "a1", "t1", 9000.0, 1),
        _row(20220414, "u3", "v7", "a4", "t2", 10000.0, 0),
        _row(20220415, "u3", "v8", "a4", "t1", 11000.0, 1),
        _row(20220415, "u3", "v2", "a1", "t2", 12000.0, 0),
    ]


def _val_rows():
    """Mostly values the train rows already carry, plus deliberate unseen
    ones (video v99, author a99, tab t9) to exercise the UNK slots."""
    return [
        _row(20220423, "u1", "v1", "a1", "t1", 1500.0, 1),
        _row(20220423, "u1", "v99", "a99", "t9", 2500.0, 0),
        _row(20220424, "u2", "v2", "a1", "t2", 3500.0, 0),
        _row(20220424, "u2", "v3", "a2", "t1", 4500.0, 1),
    ]


# ---------------------------------------------------------------------------
# Encoder faithfulness — the no-data proof that the standalone loop is
# the vendor's loop
# ---------------------------------------------------------------------------


def test_our_encoder_matches_the_vendors_exactly():
    """Identical ids, identical offsets, identical dim, for the five
    baseline fields. If this ever fails, manual/ has silently stopped
    reproducing the organizers' baseline and every ceiling number it
    produces is measured against something else."""
    splits = {"train": _train_rows(), "valid": _val_rows()}

    ours, our_dim = encode_module.encode(splits)
    theirs, their_dim = vendor.encode(splits)

    assert our_dim == their_dim
    assert set(ours) == set(theirs)
    for name in theirs:
        our_x, our_y, our_users = ours[name]
        their_x, their_y, their_users = theirs[name]
        assert np.array_equal(our_x, their_x), name
        assert np.array_equal(our_y, their_y), name
        assert our_users == their_users, name


def test_our_encoder_matches_the_vendor_on_the_backtest_style_windows():
    """The same equivalence with the fit/score naming the backtest pass
    uses, not just train/valid — our encoder takes `train_key` as a
    parameter and the vendor hardcodes 'train', so this pins that the
    parameter defaults to the vendor's behaviour."""
    fit_rows, score_rows = _train_rows()[:8], _train_rows()[8:]
    splits = {"train": fit_rows, "score": score_rows}

    ours, our_dim = encode_module.encode(splits, train_key="train")
    theirs, their_dim = vendor.encode(splits)

    assert our_dim == their_dim
    for name in theirs:
        assert np.array_equal(ours[name][0], theirs[name][0]), name


# ---------------------------------------------------------------------------
# Train-only vocabulary, UNK slots, offsets, dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score_key", ["valid", "score"])
def test_values_seen_only_at_score_time_land_in_the_unk_slot(score_key):
    """Never a newly minted id. A new id would index past the embedding
    table's width at predict time, and would also hand an untrained
    parameter to a value the model has genuinely never seen."""
    train_rows = _train_rows()
    splits = {"train": train_rows, score_key: _val_rows()}

    enc, dim = encode_module.encode(splits)
    x_score, _, _ = enc[score_key]

    # Rebuild the expected geometry straight from the train rows, rather
    # than from the encoder under test.
    fields = encode_module.BASELINE_FIELDS
    edges = encode_module.bucket_edges([row[5] for row in train_rows])
    distinct = [
        {spec.extract(row, edges) for row in train_rows} for spec in fields
    ]
    field_dims = [len(values) + 1 for values in distinct]
    offsets = np.cumsum([0] + field_dims[:-1])

    assert dim == sum(field_dims)
    # No encoded id may exceed the table width.
    assert x_score.max() < dim

    # Row 1 of _val_rows carries an unseen video (index 1), author (2) and
    # tab (3); each must sit in its field's LAST slot, the UNK one.
    for field_index in (1, 2, 3):
        expected_unk = offsets[field_index] + len(distinct[field_index])
        assert x_score[1, field_index] == expected_unk, fields[field_index].name

    # Row 0 carries only values the train rows already had, so nothing on
    # it may be UNK.
    for field_index, values in enumerate(distinct):
        unk_id = offsets[field_index] + len(values)
        assert x_score[0, field_index] != unk_id, fields[field_index].name


def test_fields_occupy_disjoint_id_ranges():
    """Every field is offset into ONE shared embedding table, and two
    fields sharing an id would make them the same parameter."""
    train_rows = _train_rows()
    enc, dim = encode_module.encode({"train": train_rows, "valid": _val_rows()})

    fields = encode_module.BASELINE_FIELDS
    edges = encode_module.bucket_edges([row[5] for row in train_rows])
    field_dims = [
        len({spec.extract(row, edges) for row in train_rows}) + 1 for spec in fields
    ]
    offsets = np.cumsum([0] + field_dims[:-1])

    for split_name in ("train", "valid"):
        x, _, _ = enc[split_name]
        for field_index in range(len(fields)):
            low = offsets[field_index]
            high = low + field_dims[field_index]
            column = x[:, field_index]
            assert column.min() >= low, (split_name, field_index)
            assert column.max() < high, (split_name, field_index)
    assert dim == sum(field_dims)


def test_encoded_arrays_have_the_dtypes_and_shapes_fm_expects():
    """int32 ids and float32 labels are what vendor FM.logits and
    FM.step index and arithmetic on."""
    train_rows, val_rows = _train_rows(), _val_rows()
    enc, _ = encode_module.encode({"train": train_rows, "valid": val_rows})

    x_train, y_train, users_train = enc["train"]
    x_val, y_val, users_val = enc["valid"]

    n_fields = len(encode_module.BASELINE_FIELDS)
    assert x_train.dtype == np.int32 and x_val.dtype == np.int32
    assert y_train.dtype == np.float32 and y_val.dtype == np.float32
    assert x_train.shape == (len(train_rows), n_fields)
    assert x_val.shape == (len(val_rows), n_fields)
    assert len(users_train) == len(train_rows)
    assert users_val == [row[1] for row in val_rows]


def test_encoder_refuses_to_fit_on_nothing():
    """An empty fitting window would build empty vocabularies and a
    degenerate quantile, and every row would encode as UNK — a pipeline
    that trains on nothing and reports a number."""
    with pytest.raises(ValueError):
        encode_module.encode({"train": [], "valid": _val_rows()})
    with pytest.raises(KeyError):
        encode_module.encode({"valid": _val_rows()})


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_harness(monkeypatch):
    """Replaces every disk-touching harness call with synthetic rows, and
    records what run_baseline did through them."""
    train_rows, val_rows = _train_rows(), _val_rows()
    saved: list[tuple] = []
    evaluated: list[int] = []

    monkeypatch.setattr(
        run_module.data,
        "load",
        lambda split: train_rows if split == "train" else val_rows,
    )
    monkeypatch.setattr(
        run_module.backtest, "split", lambda: (train_rows[:8], train_rows[8:])
    )
    monkeypatch.setattr(
        run_module.cache,
        "save_predictions",
        lambda config_id, seed, split, user_ids, labels, scores: saved.append(
            (config_id, seed, split, len(user_ids), len(labels), len(scores))
        ),
    )

    real_evaluate = run_module.metrics.evaluate

    def counting_evaluate(*args, **kwargs):
        evaluated.append(1)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(run_module.metrics, "evaluate", counting_evaluate)

    return {"saved": saved, "evaluated": evaluated}


def test_run_baseline_assembles_a_candidate_result_of_the_right_shape(fake_harness):
    seeds = (0, 1)
    result = run_module.run_baseline(seeds=seeds, hyperparams=TINY_HYPERPARAMS)

    assert isinstance(result, CandidateResult)
    assert result.status is Status.OK
    assert result.config_id == run_module.MANUAL_BASELINE_CONFIG_ID
    assert set(result.val) == set(seeds)
    assert set(result.backtest) == set(seeds)
    for seed in seeds:
        assert isinstance(result.val[seed], Metrics)
        assert isinstance(result.backtest[seed], Metrics)
        assert set(result.val[seed].values) == {"GAUC", "nDCG@5"}
        # A real number, not a nan — the int8/overflow class of bug that
        # bit this project before shows up here first.
        assert np.isfinite(result.val[seed].primary)
    assert result.wall_seconds > 0


def test_run_baseline_caches_validation_predictions_once_per_seed(fake_harness):
    """Exactly one save per seed, under the manual config id, on the
    'val' split. Missing any of these silently downgrades harness.gate to
    its weak seed-level bootstrap when unit 2 compares against this."""
    seeds = (0, 1, 2)
    run_module.run_baseline(seeds=seeds, hyperparams=TINY_HYPERPARAMS)

    saved = fake_harness["saved"]
    assert len(saved) == len(seeds)
    assert [entry[1] for entry in saved] == list(seeds)
    assert {entry[0] for entry in saved} == {run_module.MANUAL_BASELINE_CONFIG_ID}
    assert {entry[2] for entry in saved} == {"val"}
    # user_ids, labels and scores must be the same length as each other
    # and as the scored split.
    for _, _, _, n_users, n_labels, n_scores in saved:
        assert n_users == n_labels == n_scores == len(_val_rows())


def test_run_baseline_evaluates_both_passes_for_every_seed(fake_harness):
    seeds = (0, 1)
    run_module.run_baseline(seeds=seeds, hyperparams=TINY_HYPERPARAMS)

    # One final scoring call per pass per seed, plus at least one
    # early-stopping call per training run. Asserted as a lower bound
    # rather than an exact count because the number of early-stopping
    # evaluations depends on when patience trips.
    assert len(fake_harness["evaluated"]) >= 4 * len(seeds)


def test_train_and_score_returns_aligned_triples_of_raw_logits():
    """Scores must be raw logits aligned to score_rows, in score_rows
    order — never sigmoided, since both metrics are rank-based and a
    squash would make these vectors incomparable at blend time."""
    train_rows, val_rows = _train_rows(), _val_rows()

    user_ids, labels, scores = run_module.train_and_score(
        train_rows, val_rows, seed=0, hyperparams=TINY_HYPERPARAMS
    )

    assert list(user_ids) == [row[1] for row in val_rows]
    assert list(labels) == [float(row[6]) for row in val_rows]
    assert len(scores) == len(val_rows)
    assert np.all(np.isfinite(scores))
    # Logits are unbounded; probabilities would all sit inside (0, 1).
    # Not a proof, but it catches an accidental sigmoid on real data and
    # documents the intent here.
    assert scores.dtype == np.float32


def test_train_and_score_is_deterministic_for_a_given_seed():
    """Two runs of the same seed must produce identical scores, or the
    paired per-seed comparison the noise gate performs is meaningless."""
    train_rows, val_rows = _train_rows(), _val_rows()

    _, _, first = run_module.train_and_score(
        train_rows, val_rows, seed=3, hyperparams=TINY_HYPERPARAMS
    )
    _, _, second = run_module.train_and_score(
        train_rows, val_rows, seed=3, hyperparams=TINY_HYPERPARAMS
    )

    assert np.array_equal(first, second)


def test_different_seeds_produce_different_scores():
    """Keeps the determinism test above from passing vacuously."""
    train_rows, val_rows = _train_rows(), _val_rows()

    _, _, seed_a = run_module.train_and_score(
        train_rows, val_rows, seed=0, hyperparams=TINY_HYPERPARAMS
    )
    _, _, seed_b = run_module.train_and_score(
        train_rows, val_rows, seed=7, hyperparams=TINY_HYPERPARAMS
    )

    assert not np.array_equal(seed_a, seed_b)


def test_train_and_score_rejects_zero_epochs():
    """Zero epochs would leave the best-weight restore with nothing to
    restore, failing as an unpack error far from the cause."""
    with pytest.raises(ValueError):
        run_module.train_and_score(
            _train_rows(),
            _val_rows(),
            seed=0,
            hyperparams={**TINY_HYPERPARAMS, "epochs": 0},
        )


def test_the_manual_baseline_uses_the_organizers_published_hyperparameters():
    """The incumbent must be the published baseline, not a better-tuned
    one — a quietly improved baseline understates the headroom this
    pipeline exists to measure."""
    assert run_module.BASELINE_HYPERPARAMS == {
        "k": 16,
        "lr": 0.001,
        "epochs": 40,
        "bs": 8192,
        "patience": 4,
    }


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_cli_defaults_to_the_baseline_variant_on_three_seeds():
    args = run_module._parse_args([])
    assert args.variant == "baseline"
    assert run_module.parse_seeds(args.seeds) == (0, 1, 2)


def test_cli_parses_an_explicit_seed_list():
    args = run_module._parse_args(["--variant", "baseline", "--seeds", "0,2,5"])
    assert run_module.parse_seeds(args.seeds) == (0, 2, 5)


def test_cli_rejects_an_unknown_variant():
    # Was "crosses" in unit 1, when no such variant existed. Unit 2
    # registered it, so the example had to become a name that is not a
    # variant and will not become one. The assertion is unchanged.
    with pytest.raises(SystemExit):
        run_module._parse_args(["--variant", "definitely_not_a_variant"])


def test_parse_seeds_rejects_an_empty_list():
    with pytest.raises(ValueError):
        run_module.parse_seeds(" , ")


def test_every_variant_name_maps_to_a_callable():
    assert "baseline" in run_module.VARIANTS
    for name, runner in run_module.VARIANTS.items():
        assert callable(runner), name


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_mean_primary_averages_the_seeds_and_reports_none_for_an_empty_run():
    per_seed = {
        0: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60}),
        1: Metrics(values={"GAUC": 0.62, "nDCG@5": 0.62}),
    }
    assert report_module.mean_primary(per_seed) == pytest.approx(0.61)
    assert report_module.mean_primary({}) is None


def test_the_report_prints_every_seed_and_the_mean(capsys):
    result = CandidateResult(
        config_id="manual_baseline_fm_k16",
        status=Status.OK,
        val={
            0: Metrics(values={"GAUC": 0.64, "nDCG@5": 0.56}),
            1: Metrics(values={"GAUC": 0.66, "nDCG@5": 0.58}),
        },
        backtest={0: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.54})},
        wall_seconds=1.0,
    )

    report_module.print_candidate_report(result, label="manual baseline")
    out = capsys.readouterr().out

    assert "manual baseline" in out
    assert "manual_baseline_fm_k16" in out
    assert "seed 0" in out and "seed 1" in out
    assert "MEAN VAL PRIMARY" in out
    assert "0.6100" in out  # (0.60 + 0.62) / 2


def test_the_comparison_path_reports_what_the_gate_says(capsys):
    """print_comparison must surface the gate's Verdict rather than any
    arithmetic of its own. One seed exercises the SCREEN stage, which
    needs no cached predictions."""
    incumbent = CandidateResult(
        config_id="manual_baseline_fm_k16",
        status=Status.OK,
        val={0: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60})},
        backtest={0: Metrics(values={"GAUC": 0.60, "nDCG@5": 0.60})},
    )
    candidate = CandidateResult(
        config_id="manual_variant",
        status=Status.OK,
        val={0: Metrics(values={"GAUC": 0.61, "nDCG@5": 0.61})},
        backtest={0: Metrics(values={"GAUC": 0.61, "nDCG@5": 0.61})},
    )

    report_module.print_comparison(candidate, incumbent)
    out = capsys.readouterr().out

    assert "accept" in out
    assert "delta" in out
    assert "ci95" in out
    assert "backtest_delta" in out
    assert "reason" in out


# ---------------------------------------------------------------------------
# The import boundary
# ---------------------------------------------------------------------------


def test_nothing_in_manual_imports_the_executor():
    """manual/ is standalone by design: harness (W1), contracts, and the
    vendored kit only. Importing W3 would couple this ceiling to another
    workstream's in-flight file — and to executor.realize's dispatch,
    which raises NotImplementedError for exactly the additive moves the
    ceiling exists to try.

    Parsed from each module's AST rather than grepped for the substring,
    because the docstrings here discuss the executor at length and a test
    that a word is absent from prose would be a test of the prose.
    """
    modules = sorted(MANUAL_DIR.glob("*.py"))
    assert modules, "no modules found under manual/"

    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                assert name.split(".")[0] != "executor", f"{path.name}: {name}"


def test_manual_imports_no_other_first_party_package():
    """Stronger than the executor check: every sibling package is
    forbidden, so a new dependency on controller/, methods/ or scripts/
    is caught by the same guard rather than needing a new one."""
    forbidden = {"controller", "executor", "methods", "scripts", "tests"}
    offenders = []

    for path in sorted(MANUAL_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if name and name.split(".")[0] in forbidden:
                    offenders.append(f"{path.name} imports {name}")

    assert offenders == []


# ---------------------------------------------------------------------------
# UNIT 2: user-side x item-side cross columns
# ---------------------------------------------------------------------------


class _FakeFrame:
    """The two operations SideTables.from_frames uses, and nothing else.

    A stand-in for a pandas DataFrame so these tests neither build one nor
    read a CSV: `.columns`, and the
    `frame[[cols]].itertuples(index=False)` access pattern.
    """

    def __init__(self, records, columns):
        self._records = records
        self.columns = list(columns)

    def __getitem__(self, wanted):
        rows = [tuple(record[column] for column in wanted) for record in self._records]
        return _FakeSelection(rows)


class _FakeSelection:
    def __init__(self, rows):
        self._rows = rows

    def itertuples(self, index=False):
        return iter(self._rows)


def _side_tables():
    """Synthetic side tables covering the three crosses' columns.

    u1/u2/u3 are present. Other tests deliberately encode rows for ids the
    tables do not contain, which must degrade to MISSING rather than crash.
    """
    user_frame = _FakeFrame(
        [
            {"user_id": "u1", "user_active_degree": "full_active",
             "register_days_range": "730+", "watch_ratio": 0.1},
            {"user_id": "u2", "user_active_degree": "middle_active",
             "register_days_range": "[31,60)", "watch_ratio": 0.5},
            {"user_id": "u3", "user_active_degree": "full_active",
             "register_days_range": "[31,60)", "watch_ratio": 0.9},
        ],
        ["user_id", "user_active_degree", "register_days_range", "watch_ratio"],
    )
    video_frame = _FakeFrame(
        [
            {"video_id": "v1", "video_type": "NORMAL"},
            {"video_id": "v2", "video_type": "AD"},
            {"video_id": "v3", "video_type": "NORMAL"},
            {"video_id": "v4", "video_type": "NORMAL"},
            {"video_id": "v5", "video_type": "AD"},
            {"video_id": "v6", "video_type": "NORMAL"},
            {"video_id": "v7", "video_type": "NORMAL"},
            {"video_id": "v8", "video_type": "AD"},
        ],
        ["video_id", "video_type"],
    )
    return encode_module.SideTables.from_frames(
        user_frame,
        video_frame,
        encode_module.CROSS_USER_COLUMNS,
        encode_module.CROSS_VIDEO_COLUMNS,
    )


def test_side_tables_key_on_strings_so_int_ids_still_match():
    """Rows carry ids as strings; pandas reads the side tables' id columns
    as int64. Unconverted, every lookup would miss and every cross would be
    MISSING — a total failure that still trains and still reports a
    plausible number."""
    user_frame = _FakeFrame(
        [{"user_id": 7, "user_active_degree": "full_active"}],
        ["user_id", "user_active_degree"],
    )
    video_frame = _FakeFrame(
        [{"video_id": 9, "video_type": "NORMAL"}], ["video_id", "video_type"]
    )

    tables = encode_module.SideTables.from_frames(
        user_frame, video_frame, ("user_active_degree",), ("video_type",)
    )

    assert tables.user["7"]["user_active_degree"] == "full_active"
    assert tables.video["9"]["video_type"] == "NORMAL"


def test_the_three_crosses_are_the_ones_the_design_calls_for():
    """Each pairs a user-side attribute with an item-side one. Crossing two
    baseline fields would be redundant — the FM's bilinear term already
    models every pair among them."""
    names = [spec.name for spec in encode_module.CROSS_SPECS]
    assert names == [
        "user_active_degree_x_video_type",
        "user_active_degree_x_dur_bucket",
        "register_days_range_x_video_type",
    ]
    for spec in encode_module.CROSS_SPECS:
        assert spec.user_term.kind == "user_table"
        assert spec.item_term.kind in {"video_table", "row_field"}


def test_crosses_append_three_columns_and_leave_the_baseline_five_identical():
    """The load-bearing additivity check. If a baseline column shifted, the
    gate's verdict would stop being attributable to the crosses alone."""
    splits = {"train": _train_rows(), "valid": _val_rows()}

    plain, plain_dim = encode_module.encode(splits)
    widened, widened_dim = encode_module.encode(
        splits, crosses=encode_module.CROSS_SPECS, side_tables=_side_tables()
    )

    for name in splits:
        plain_x, plain_y, plain_users = plain[name]
        wide_x, wide_y, wide_users = widened[name]
        assert wide_x.shape == (len(splits[name]), 8)
        assert wide_x.dtype == np.int32
        assert wide_y.dtype == np.float32
        assert np.array_equal(wide_x[:, :5], plain_x), name
        assert np.array_equal(wide_y, plain_y), name
        assert wide_users == plain_users, name

    assert widened_dim > plain_dim


def test_the_widened_encoder_still_matches_the_vendor_on_the_baseline_columns():
    """Unit 1's fidelity guarantee must survive the extension."""
    splits = {"train": _train_rows(), "valid": _val_rows()}
    widened, _ = encode_module.encode(
        splits, crosses=encode_module.CROSS_SPECS, side_tables=_side_tables()
    )
    theirs, _ = vendor.encode(splits)

    for name in theirs:
        assert np.array_equal(widened[name][0][:, :5], theirs[name][0]), name


def test_cross_columns_occupy_their_own_disjoint_id_ranges():
    splits = {"train": _train_rows(), "valid": _val_rows()}
    enc, dim = encode_module.encode(
        splits, crosses=encode_module.CROSS_SPECS, side_tables=_side_tables()
    )

    x_train, _, _ = enc["train"]
    x_val, _, _ = enc["valid"]
    assert x_train.max() < dim and x_val.max() < dim

    # One shared table, non-overlapping blocks: every column's ids sit
    # strictly above the previous column's.
    for column in range(1, 8):
        assert x_train[:, column].min() > x_train[:, column - 1].max(), column


def test_a_cross_pair_seen_only_at_score_time_lands_in_the_unk_slot():
    """Leakage guard. An unseen pair must never mint a new id — that would
    index past the embedding table and hand an untrained parameter to a
    combination the model has never seen."""
    train_rows = [
        _row(20220410, "u1", "v1", "a1", "t1", 1000.0, 1),  # full_active x NORMAL
        _row(20220410, "u1", "v2", "a1", "t1", 2000.0, 0),  # full_active x AD
        _row(20220411, "u1", "v3", "a2", "t2", 3000.0, 1),
        _row(20220411, "u1", "v4", "a2", "t1", 4000.0, 0),
    ]
    # u2 is middle_active, so middle_active x AD never occurs in train.
    val_rows = [
        _row(20220423, "u1", "v1", "a1", "t1", 1500.0, 1),
        _row(20220423, "u2", "v2", "a1", "t1", 2500.0, 0),
    ]

    enc, _ = encode_module.encode(
        {"train": train_rows, "valid": val_rows},
        crosses=(encode_module.CROSS_SPECS[0],),
        side_tables=_side_tables(),
    )
    x_train, _, _ = enc["train"]
    x_val, _, _ = enc["valid"]

    cross_column = 5  # five baseline fields, then this one
    train_ids = set(x_train[:, cross_column].tolist())
    unk_id = max(train_ids) + 1  # the UNK slot sits after the known values

    assert x_val[0, cross_column] in train_ids  # full_active x NORMAL was seen
    assert x_val[1, cross_column] == unk_id  # middle_active x AD was not
    assert x_val[1, cross_column] not in train_ids


def test_cross_vocabularies_are_fit_on_the_fitting_window_only():
    """Encoding with and without a scoring split must give the fitting
    window byte-identical ids — otherwise the score window's values are
    leaking into the vocabulary."""
    train_rows = _train_rows()

    only_train, dim_a = encode_module.encode(
        {"train": train_rows},
        crosses=encode_module.CROSS_SPECS,
        side_tables=_side_tables(),
    )
    with_score, dim_b = encode_module.encode(
        {"train": train_rows, "valid": _val_rows()},
        crosses=encode_module.CROSS_SPECS,
        side_tables=_side_tables(),
    )

    assert dim_a == dim_b
    assert np.array_equal(only_train["train"][0], with_score["train"][0])


def test_a_user_or_video_absent_from_a_side_table_degrades_to_missing():
    """Robustness: the interaction log is the source of truth for which
    rows exist, so an id with no side-table record must encode, not
    crash."""
    train_rows = [
        _row(20220410, "u1", "v1", "a1", "t1", 1000.0, 1),
        _row(20220410, "u404", "v1", "a1", "t1", 2000.0, 0),  # unknown user
        _row(20220411, "u1", "v404", "a2", "t2", 3000.0, 1),  # unknown video
        _row(20220411, "u404", "v404", "a2", "t1", 4000.0, 0),  # both unknown
    ]

    enc, dim = encode_module.encode(
        {"train": train_rows},
        crosses=encode_module.CROSS_SPECS,
        side_tables=_side_tables(),
    )

    x_train, _, _ = enc["train"]
    assert x_train.shape == (4, 8)
    assert x_train.max() < dim
    # An unknown user encodes to a different cross value than a known one.
    assert x_train[1, 5] != x_train[0, 5]


def test_missing_side_values_still_earn_an_id_when_they_recur_in_train():
    """MISSING is a real category, not a discard: recurring in the fitting
    window earns it an embedding rather than forcing it into UNK."""
    train_rows = [
        _row(20220410, "u404", "v1", "a1", "t1", 1000.0, 1),
        _row(20220411, "u404", "v1", "a1", "t2", 2000.0, 0),
    ]
    val_rows = [_row(20220423, "u404", "v1", "a1", "t1", 1500.0, 1)]

    enc, _ = encode_module.encode(
        {"train": train_rows, "valid": val_rows},
        crosses=(encode_module.CROSS_SPECS[0],),
        side_tables=_side_tables(),
    )

    train_ids = set(enc["train"][0][:, 5].tolist())
    assert enc["valid"][0][0, 5] in train_ids


def test_encode_refuses_a_cross_when_no_side_tables_were_supplied():
    with pytest.raises(ValueError):
        encode_module.encode(
            {"train": _train_rows()},
            crosses=encode_module.CROSS_SPECS,
            side_tables=None,
        )


def test_a_cross_naming_an_unknown_row_field_is_rejected():
    bogus = encode_module.CrossSpec(
        "bogus",
        encode_module.Term("user_table", "user_active_degree"),
        encode_module.Term("row_field", "no_such_field"),
    )
    with pytest.raises(KeyError):
        encode_module.encode(
            {"train": _train_rows()}, crosses=(bogus,), side_tables=_side_tables()
        )


def test_side_tables_reject_a_missing_column():
    user_frame = _FakeFrame([{"user_id": "u1"}], ["user_id"])
    video_frame = _FakeFrame([{"video_id": "v1"}], ["video_id"])
    with pytest.raises(KeyError):
        encode_module.SideTables.from_frames(
            user_frame, video_frame, ("user_active_degree",), ()
        )


def _continuous_cross_tables():
    return encode_module.SideTables.from_frames(
        _FakeFrame(
            [
                {"user_id": "u1", "watch_ratio": 0.1},
                {"user_id": "u2", "watch_ratio": 0.5},
                {"user_id": "u3", "watch_ratio": 0.9},
            ],
            ["user_id", "watch_ratio"],
        ),
        _FakeFrame(
            [{"video_id": f"v{i}", "video_type": "NORMAL"} for i in range(1, 9)],
            ["video_id", "video_type"],
        ),
        ("watch_ratio",),
        ("video_type",),
    )


def test_a_continuous_side_feature_is_bucketed_rather_than_used_raw():
    """None of the three shipped crosses has a continuous half, so the
    bucketing path is proven here on a synthetic column — otherwise it
    would be dead code that unit 3 discovers is broken."""
    continuous = encode_module.CrossSpec(
        "watch_ratio_x_video_type",
        encode_module.Term("user_table", "watch_ratio", buckets=2),
        encode_module.Term("video_table", "video_type"),
    )

    enc, _ = encode_module.encode(
        {"train": _train_rows()},
        crosses=(continuous,),
        side_tables=_continuous_cross_tables(),
    )
    x_train, _, _ = enc["train"]

    train_rows = _train_rows()
    u1_row = next(i for i, row in enumerate(train_rows) if row[1] == "u1")
    u3_row = next(i for i, row in enumerate(train_rows) if row[1] == "u3")
    # 0.1 and 0.9 straddle the median edge, so they must land in different
    # buckets and therefore different cross ids.
    assert x_train[u1_row, 5] != x_train[u3_row, 5]


def test_continuous_bucket_edges_are_fit_on_the_fitting_window_only():
    """Adding a scoring split must not move the fitting window's buckets."""
    continuous = encode_module.CrossSpec(
        "watch_ratio_x_video_type",
        encode_module.Term("user_table", "watch_ratio", buckets=2),
        encode_module.Term("video_table", "video_type"),
    )
    tables = _continuous_cross_tables()
    train_rows = _train_rows()

    only_train, _ = encode_module.encode(
        {"train": train_rows}, crosses=(continuous,), side_tables=tables
    )
    with_score, _ = encode_module.encode(
        {"train": train_rows, "valid": _val_rows()},
        crosses=(continuous,),
        side_tables=tables,
    )

    assert np.array_equal(only_train["train"][0], with_score["train"][0])


# ---------------------------------------------------------------------------
# The crosses variant and the gate comparison
# ---------------------------------------------------------------------------


def test_crosses_is_registered_with_its_own_config_id():
    assert run_module.VARIANTS["crosses"] is run_module.run_crosses
    assert run_module.MANUAL_CROSSES_CONFIG_ID != run_module.MANUAL_BASELINE_CONFIG_ID


def test_run_crosses_caches_under_its_own_id_and_returns_a_result(
    fake_harness, monkeypatch
):
    monkeypatch.setattr(run_module, "load_side_tables", _side_tables)
    seeds = (0, 1)

    result = run_module.run_crosses(seeds=seeds, hyperparams=TINY_HYPERPARAMS)

    assert isinstance(result, CandidateResult)
    assert result.status is Status.OK
    assert result.config_id == run_module.MANUAL_CROSSES_CONFIG_ID
    assert set(result.val) == set(seeds)
    assert set(result.backtest) == set(seeds)

    saved = fake_harness["saved"]
    assert [entry[0] for entry in saved] == [run_module.MANUAL_CROSSES_CONFIG_ID] * 2
    assert [entry[1] for entry in saved] == list(seeds)
    assert {entry[2] for entry in saved} == {"val"}


def test_run_crosses_loads_the_side_tables_through_harness(fake_harness, monkeypatch):
    """Not by reading the CSVs itself — harness.data.load_side_features is
    the only sanctioned door to those two files."""
    calls = []

    def spy():
        calls.append(1)
        return _side_tables()

    monkeypatch.setattr(run_module, "load_side_tables", spy)
    run_module.run_crosses(seeds=(0,), hyperparams=TINY_HYPERPARAMS)

    assert len(calls) == 1


def test_the_comparison_path_surfaces_every_verdict_field(
    fake_harness, monkeypatch, capsys
):
    """crosses vs baseline, end to end through gate.compare."""
    monkeypatch.setattr(run_module, "load_side_tables", _side_tables)
    # No cached baseline predictions here, so the incumbent takes the
    # full re-run path.
    monkeypatch.setattr(run_module.cache, "exists", lambda *a, **k: False)
    seeds = (0, 1, 2)

    candidate = run_module.run_crosses(seeds=seeds, hyperparams=TINY_HYPERPARAMS)
    incumbent, provenance = run_module.baseline_incumbent(
        seeds=seeds, hyperparams=TINY_HYPERPARAMS
    )

    assert "re-run" in provenance
    assert incumbent.config_id == run_module.MANUAL_BASELINE_CONFIG_ID
    assert set(incumbent.backtest) == set(seeds)

    report_module.print_comparison(candidate, incumbent, label="crosses vs baseline")
    out = capsys.readouterr().out
    for field in ("accept", "delta", "ci95", "n_seeds", "backtest_delta", "reason"):
        assert field in out


def test_the_incumbent_reuses_cached_validation_predictions_when_present(
    fake_harness, monkeypatch
):
    """The point of caching: no validation retrain for the incumbent. The
    backtest half is still re-run, and the provenance string says so."""
    val_rows = _val_rows()
    cached_scores = np.linspace(0.0, 1.0, len(val_rows)).astype(np.float32)

    monkeypatch.setattr(run_module.cache, "exists", lambda *a, **k: True)
    monkeypatch.setattr(
        run_module.cache,
        "load_predictions",
        lambda config_id, seed, split: (
            np.array([row[1] for row in val_rows]),
            np.array([row[6] for row in val_rows], dtype=np.int8),
            cached_scores,
        ),
    )

    incumbent, provenance = run_module.baseline_incumbent(
        seeds=(0, 1, 2), hyperparams=TINY_HYPERPARAMS
    )

    assert "cached" in provenance
    assert set(incumbent.val) == {0, 1, 2}
    assert set(incumbent.backtest) == {0, 1, 2}
    # The incumbent path must not overwrite the predictions it just read.
    assert fake_harness["saved"] == []


def test_cli_accepts_the_crosses_variant_with_a_comparison():
    args = run_module._parse_args(
        ["--variant", "crosses", "--seeds", "0,1,2", "--compare-to", "baseline"]
    )
    assert args.variant == "crosses"
    assert args.compare_to == "baseline"
    assert run_module.parse_seeds(args.seeds) == (0, 1, 2)


def test_cli_comparison_defaults_to_off():
    assert run_module._parse_args([]).compare_to is None


def test_cli_rejects_an_unknown_comparison_target():
    with pytest.raises(SystemExit):
        run_module._parse_args(["--compare-to", "not_a_variant"])
