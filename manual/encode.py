"""Categorical encoding for the manual ceiling: the vendor's `encode()`,
re-expressed with the field list as a parameter.

WHY RE-IMPLEMENT SOMETHING THAT ALREADY WORKS
---------------------------------------------
The vendor's encode() is read-only and its field list is hard-coded in
two places at once — `FIELDS` names five fields, and a closure literally
spelled `[x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]`
extracts exactly those five from the row tuple. There is no seam. Adding
an engineered cross column (unit 2's job) means changing that closure,
which means editing the vendored kit, which is forbidden.

So this module keeps every rule the vendor's encoder enforces and moves
only the field list behind a parameter. A cross becomes one more
FieldSpec in the list; nothing here changes again.

THE RULES BEING PRESERVED, AND WHY EACH ONE MATTERS
---------------------------------------------------
1. VOCABULARIES ARE FIT ON THE TRAIN ROWS ONLY. Not on the union, not on
   whatever split is being encoded. A vocabulary fit on validation rows
   would let the model allocate a parameter for a video it is about to be
   scored on, which is leakage and inflates the score.
2. THE DURATION BUCKET EDGES ARE ALSO FIT ON TRAIN ONLY, for the same
   reason — quantile edges computed over validation durations encode the
   validation distribution into the features.
3. EVERY FIELD GETS A TRAILING UNK SLOT, and an unseen value lands in it
   rather than minting a new id. Minting would produce ids past the
   embedding table's width and index out of bounds at predict time; it
   would also mean a value seen once at score time got an untrained
   parameter, which is worse than admitting the model has never seen it.
4. FIELDS ARE OFFSET INTO ONE SHARED EMBEDDING TABLE. `FM.logits` does
   `E = self.V[X]` then `E.sum(1)` — one table, one lookup, summed over
   whatever columns exist. This is also exactly why adding a cross column
   needs no change to FM at all.

FIDELITY IS TESTED, NOT ASSERTED. tests/test_manual.py encodes the same
synthetic splits through this module and through the vendor's own
encode(), and requires identical ids, identical offsets and an identical
dim for the five baseline fields. That test is the reason this file can
be trusted without the dataset present.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

import numpy as np

# Row tuple layout, fixed by harness.data (which inherits it from the
# vendor loader). Named here so the extractors below read as field access
# rather than as magic indices.
_DATE, _USER_ID, _VIDEO_ID, _AUTHOR_ID, _TAB, _DURATION_MS, _LABEL = range(7)

_N_DURATION_BUCKETS = 10

Row = tuple

MISSING = "__MISSING__"
"""Stand-in for a side-table lookup that found nothing.

A real value, not None, and deliberately not silently dropped. An id that
appears in the interaction log but not in the side table is a fact about
the data, and "this user's activity level is unknown" is a perfectly good
thing to cross with a video type — if that combination recurs in the
training window it earns its own embedding, and if it does not it lands in
the UNK slot like anything else. Dropping the row or raising would both be
worse: the log is the source of truth for which rows exist.
"""

_CROSS_SEP = "\x1f"
"""ASCII unit separator, joining the two halves of a cross value.

Not "_" or "|": those occur inside real values, and "a_b" x "c" would then
collide with "a" x "b_c" — two different conjunctions sharing one
embedding. U+001F cannot appear in any of these CSV fields.
"""


class FieldSpec(NamedTuple):
    """One categorical field: a name, and how to read it off a row.

    `extract` takes the row and the duration-bucket edges (the only piece
    of fitted state any current field needs) and returns the field's value
    as a STRING. String rather than the raw value because vocabularies are
    dicts keyed on it, and `10` and `"10"` must not become two ids.
    """

    name: str
    extract: Callable[[Row, np.ndarray], str]


def bucket_edges(durations: Sequence[float], n: int = _N_DURATION_BUCKETS) -> np.ndarray:
    """Quantile edges for the duration bucket, verbatim from the vendor.

    `np.linspace(0, 1, n + 1)[1:-1]` drops the 0.0 and 1.0 quantiles, so
    n buckets need n-1 interior edges. Reproduced exactly rather than
    approximated: `np.searchsorted` against a differently-derived edge
    array would silently shift rows between buckets and the encoding
    would stop matching the vendor's.
    """
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


BASELINE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("user_id", lambda row, edges: row[_USER_ID]),
    FieldSpec("video_id", lambda row, edges: row[_VIDEO_ID]),
    FieldSpec("author_id", lambda row, edges: row[_AUTHOR_ID]),
    FieldSpec("tab", lambda row, edges: row[_TAB]),
    FieldSpec(
        "dur_bucket",
        lambda row, edges: str(int(np.searchsorted(edges, row[_DURATION_MS]))),
    ),
)
"""The organizers' five fields, in the vendor's exact order.

Order is load-bearing, not cosmetic: it fixes each field's column index
in X and therefore its offset into the shared table. Reordering would
produce a valid encoding that simply is not the vendor's, and the
fidelity test would catch it.

`np.searchsorted`'s default side='left' is the vendor's behaviour and is
relied on here.
"""


# ---------------------------------------------------------------------------
# Crosses: user-side x item-side conjunctions
# ---------------------------------------------------------------------------


class Term(NamedTuple):
    """One half of a cross: where to read a value, and how to discretize it.

    `kind` is one of:
      "user_table"  - a column of the user side table, keyed by the row's
                      user_id
      "video_table" - a column of the video side table, keyed by video_id
      "row_field"   - one of the already-encoded baseline fields, named by
                      its FieldSpec name (e.g. "dur_bucket", "tab")

    "row_field" exists so a cross can reuse a value the baseline already
    derives instead of re-deriving it. Crossing against `dur_bucket` this
    way is guaranteed to use the SAME quantile edges the baseline column
    used, because it calls the same FieldSpec; a second, independently
    fitted bucketing would put the same row in two different buckets
    depending on which column you read.

    `buckets`, when set, means the raw value is CONTINUOUS and must be
    quantile-bucketed into that many bins, with edges fit on the train
    rows only — the same discipline `dur_bucket` follows, and for the same
    reason: edges computed over the scoring window would encode that
    window's distribution into the feature. A non-numeric value under a
    bucketed term becomes MISSING rather than raising, since one unparseable
    cell must not kill a 1.1M-row encode.
    """

    kind: str
    key: str
    buckets: Optional[int] = None


class CrossSpec(NamedTuple):
    """A single new categorical column: the conjunction of two terms.

    WHY A CONJUNCTION AND NOT TWO COLUMNS. Adding the side features as
    their own columns is the experiment the organizers already ran and
    published as a loss (primary 0.5940 vs 0.5950). Their stated reason is
    the one that matters here: a term that is constant within a user cannot
    change that user's ranking, because the metrics rank within a user's
    own impressions — so a pure user-side column contributes exactly
    nothing to GAUC or nDCG@5, no matter how informative it looks.

    A conjunction is not constant within a user: it varies with the item
    half. `user_active_degree x video_type` gives a full-active user one
    embedding for NORMAL videos and another for AD videos, which is
    precisely the "user-side features can only act through crosses with
    the item side" hint, expressed as a feature. That is what this unit
    tests, and it is why the raw side features are deliberately NOT added
    alongside.
    """

    name: str
    user_term: Term
    item_term: Term


CROSS_SPECS: tuple[CrossSpec, ...] = (
    CrossSpec(
        "user_active_degree_x_video_type",
        Term("user_table", "user_active_degree"),
        Term("video_table", "video_type"),
    ),
    CrossSpec(
        "user_active_degree_x_dur_bucket",
        Term("user_table", "user_active_degree"),
        # "row_field", not a second bucketing of video_duration: this
        # reuses the baseline's own dur_bucket FieldSpec and therefore its
        # exact train-fitted edges.
        Term("row_field", "dur_bucket"),
    ),
    CrossSpec(
        "register_days_range_x_video_type",
        Term("user_table", "register_days_range"),
        Term("video_table", "video_type"),
    ),
)
"""The three crosses this unit measures, and nothing else.

Each pairs a user-side attribute the FM has never seen with an item-side
one, so the conjunction varies within a user and can therefore move a
within-user ranking. Crossing two of the FM's OWN five fields would be
redundant — its bilinear term already models every pairwise interaction
among them.

`user_active_degree` twice, against two different item halves, is
deliberate: it is the strongest user-side signal in the table, and running
it against both a video attribute and a duration bucket separates "does
this user prefer this KIND of video" from "does this user prefer videos of
this LENGTH". `register_days_range x video_type` is the tenure analogue of
the first, and tells us whether the effect is about activity or about
account age.

Columns verified present in the shipped tables: user_features_pure.csv
carries user_active_degree and register_days_range;
video_features_basic_pure.csv carries video_type.
"""

CROSS_USER_COLUMNS: tuple[str, ...] = ("user_active_degree", "register_days_range")
CROSS_VIDEO_COLUMNS: tuple[str, ...] = ("video_type",)
"""Exactly the side-table columns CROSS_SPECS reads, so SideTables can
retain those and drop the other 38."""


class SideTables(NamedTuple):
    """User- and video-side attribute lookups, keyed by id STRING.

    Keyed by string because the interaction rows carry ids as strings
    (csv.DictReader gives text, and harness.data keeps the vendor's row
    shape) while pandas reads the side-table id columns as int64. Left
    unconverted, every single lookup would miss and every cross would be
    MISSING — a silent, total failure that still trains and still reports
    a plausible number. `from_frames` does the conversion in one place.
    """

    user: Mapping[str, Mapping[str, str]]
    video: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_frames(
        cls,
        user_frame,
        video_frame,
        user_columns: Sequence[str],
        video_columns: Sequence[str],
    ) -> "SideTables":
        """Builds the lookups from harness.data.load_side_features()'s frames.

        Only the named columns are retained: the user table ships 30
        columns and the video table 12, almost all unused here, and
        carrying them all would multiply this dict's footprint for nothing.
        """
        return cls(
            user=_frame_lookup(user_frame, "user_id", user_columns),
            video=_frame_lookup(video_frame, "video_id", video_columns),
        )


def _frame_lookup(frame, id_column: str, columns: Sequence[str]):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"side table is missing column(s) {missing}; it has "
            f"{list(frame.columns)}"
        )
    lookup: dict[str, dict[str, str]] = {}
    wanted = list(columns)
    for record in frame[[id_column, *wanted]].itertuples(index=False):
        values = tuple(record)
        lookup[str(values[0])] = {
            column: str(value) for column, value in zip(wanted, values[1:])
        }
    return lookup


def _term_value(
    term: Term,
    row: Row,
    side_tables: Optional[SideTables],
    edges: np.ndarray,
    field_by_name: Mapping[str, FieldSpec],
) -> str:
    """The term's raw (pre-bucketing) value for one row, or MISSING."""
    if term.kind == "row_field":
        spec = field_by_name.get(term.key)
        if spec is None:
            raise KeyError(
                f"cross term names row field {term.key!r}, which is not among "
                f"the encoded fields {sorted(field_by_name)}"
            )
        return spec.extract(row, edges)

    if side_tables is None:
        raise ValueError(
            f"cross term {term.kind}.{term.key} needs side tables, but "
            "encode() was called with side_tables=None"
        )

    if term.kind == "user_table":
        record = side_tables.user.get(str(row[_USER_ID]))
    elif term.kind == "video_table":
        record = side_tables.video.get(str(row[_VIDEO_ID]))
    else:
        raise ValueError(f"unknown cross term kind {term.kind!r}")

    if record is None:
        return MISSING
    return record.get(term.key, MISSING)


def _fit_term_edges(
    term: Term,
    train_rows: Sequence[Row],
    side_tables: Optional[SideTables],
    edges: np.ndarray,
    field_by_name: Mapping[str, FieldSpec],
) -> Optional[np.ndarray]:
    """Quantile edges for a continuous term, fit on the TRAIN rows only."""
    if not term.buckets:
        return None
    values = []
    for row in train_rows:
        raw_value = _term_value(term, row, side_tables, edges, field_by_name)
        if raw_value == MISSING:
            continue
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            continue
    if not values:
        # Every train value was missing or unparseable. Returning None
        # degrades this term to its raw string form rather than crashing;
        # with no numeric values there is nothing to fit edges from, and a
        # quantile over an empty array raises.
        return None
    return bucket_edges(values, n=term.buckets)


def _discretized_term_value(
    term: Term,
    row: Row,
    side_tables: Optional[SideTables],
    edges: np.ndarray,
    field_by_name: Mapping[str, FieldSpec],
    term_edges: Optional[np.ndarray],
) -> str:
    raw_value = _term_value(term, row, side_tables, edges, field_by_name)
    if term_edges is None:
        return raw_value
    if raw_value == MISSING:
        return MISSING
    try:
        return str(int(np.searchsorted(term_edges, float(raw_value))))
    except (TypeError, ValueError):
        return MISSING


class _Column(NamedTuple):
    """One encoded column: a name and a row -> string value function."""

    name: str
    value: Callable[[Row], str]


def _build_columns(
    fields: Sequence[FieldSpec],
    crosses: Sequence[CrossSpec],
    side_tables: Optional[SideTables],
    train_rows: Sequence[Row],
    edges: np.ndarray,
) -> list[_Column]:
    """The baseline fields first, then the crosses, in declaration order.

    Order matters for the same reason BASELINE_FIELDS' order does: it
    fixes each column's index in X and its offset into the shared table.
    Crosses are strictly APPENDED so the five baseline columns keep the
    exact ids they had with no crosses at all — which is what lets the
    fidelity test against the vendor's encoder keep passing unchanged.
    """
    field_by_name = {spec.name: spec for spec in fields}
    columns = [
        _Column(spec.name, lambda row, spec=spec: spec.extract(row, edges))
        for spec in fields
    ]

    for cross in crosses:
        user_edges = _fit_term_edges(
            cross.user_term, train_rows, side_tables, edges, field_by_name
        )
        item_edges = _fit_term_edges(
            cross.item_term, train_rows, side_tables, edges, field_by_name
        )

        def cross_value(
            row: Row,
            cross: CrossSpec = cross,
            user_edges: Optional[np.ndarray] = user_edges,
            item_edges: Optional[np.ndarray] = item_edges,
        ) -> str:
            left = _discretized_term_value(
                cross.user_term, row, side_tables, edges, field_by_name, user_edges
            )
            right = _discretized_term_value(
                cross.item_term, row, side_tables, edges, field_by_name, item_edges
            )
            return f"{left}{_CROSS_SEP}{right}"

        columns.append(_Column(cross.name, cross_value))

    names = [column.name for column in columns]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate column names in the encoding: {names}")
    return columns


def encode(
    splits: dict[str, list[Row]],
    fields: Sequence[FieldSpec] = BASELINE_FIELDS,
    train_key: str = "train",
    crosses: Sequence[CrossSpec] = (),
    side_tables: Optional[SideTables] = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[Any]]], int]:
    """Encodes every split in `splits`, fitting only on `splits[train_key]`.

    Returns `(enc, dim)` in the vendor's exact shape:

        enc[name] = (X, y, users)
            X     : int32, (n_rows, n_columns) - offset ids
            y     : float32, (n_rows,)         - the 0/1 label
            users : list                        - user_id per row, for
                                                  the within-user metrics
        dim       : int - total width of the shared embedding table

    `n_columns` is `len(fields) + len(crosses)`. With `crosses=()` — the
    default — this is byte-for-byte the unit-1 encoding, and therefore
    still byte-for-byte the vendor's.

    `train_key` is a parameter because this encoder is called twice per
    seed with different windows under the same name: once with the real
    train/val split, once with the backtest's fit/score windows. Both
    times the fitting window is the one passed as `train_key`, which is
    what keeps the backtest honest — it must fit on its own earlier
    window, never on the full train split. Every cross vocabulary and
    every bucket edge is fit on that same window, for the same reason.
    """
    if train_key not in splits:
        raise KeyError(
            f"encode() needs a {train_key!r} entry in splits to fit vocabularies "
            f"on; got keys {sorted(splits)}"
        )
    if not fields:
        raise ValueError("encode() needs at least one field")

    train_rows = splits[train_key]
    if not train_rows:
        raise ValueError(
            f"encode() cannot fit on an empty {train_key!r} split: there are no "
            "rows to build vocabularies or duration-bucket edges from"
        )

    edges = bucket_edges([row[_DURATION_MS] for row in train_rows])
    columns = _build_columns(fields, crosses, side_tables, train_rows, edges)

    def raw(row: Row) -> list[str]:
        return [column.value(row) for column in columns]

    # Insertion-ordered, so ids are assigned in first-appearance order
    # over the train rows — the vendor's behaviour, and what makes two
    # encodings of the same data comparable id-for-id.
    vocabs: list[dict[str, int]] = [{} for _ in columns]
    for row in train_rows:
        for i, value in enumerate(raw(row)):
            if value not in vocabs[i]:
                vocabs[i][value] = len(vocabs[i])

    # One UNK slot per column, sitting immediately after that column's
    # known values — so a column with v distinct values occupies v+1 ids.
    # A cross pair seen only at score time lands here, exactly like an
    # unseen video_id does.
    unk = [len(vocab) for vocab in vocabs]
    field_dims = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc: dict[str, tuple[np.ndarray, np.ndarray, list[Any]]] = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(columns)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users: list[Any] = []
        for n, row in enumerate(rows):
            for i, value in enumerate(raw(row)):
                X[n, i] = vocabs[i].get(value, unk[i]) + offsets[i]
            y[n] = row[_LABEL]
            users.append(row[_USER_ID])
        enc[name] = (X, y, users)

    return enc, int(sum(field_dims))
