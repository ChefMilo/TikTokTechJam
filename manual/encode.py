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

from typing import Any, Callable, NamedTuple, Sequence

import numpy as np

# Row tuple layout, fixed by harness.data (which inherits it from the
# vendor loader). Named here so the extractors below read as field access
# rather than as magic indices.
_DATE, _USER_ID, _VIDEO_ID, _AUTHOR_ID, _TAB, _DURATION_MS, _LABEL = range(7)

_N_DURATION_BUCKETS = 10

Row = tuple


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


def encode(
    splits: dict[str, list[Row]],
    fields: Sequence[FieldSpec] = BASELINE_FIELDS,
    train_key: str = "train",
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[Any]]], int]:
    """Encodes every split in `splits`, fitting only on `splits[train_key]`.

    Returns `(enc, dim)` in the vendor's exact shape:

        enc[name] = (X, y, users)
            X     : int32, (n_rows, len(fields)) - offset ids
            y     : float32, (n_rows,)           - the 0/1 label
            users : list                          - user_id per row, for
                                                    the within-user metrics
        dim       : int - total width of the shared embedding table

    `train_key` is a parameter because this encoder is called twice per
    seed with different windows under the same name: once with the real
    train/val split, once with the backtest's fit/score windows. Both
    times the fitting window is the one passed as `train_key`, which is
    what keeps the backtest honest — it must fit on its own earlier
    window, never on the full train split.
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

    def raw(row: Row) -> list[str]:
        return [spec.extract(row, edges) for spec in fields]

    # Insertion-ordered, so ids are assigned in first-appearance order
    # over the train rows — the vendor's behaviour, and what makes two
    # encodings of the same data comparable id-for-id.
    vocabs: list[dict[str, int]] = [{} for _ in fields]
    for row in train_rows:
        for i, value in enumerate(raw(row)):
            if value not in vocabs[i]:
                vocabs[i][value] = len(vocabs[i])

    # One UNK slot per field, sitting immediately after that field's known
    # values — so a field with v distinct values occupies v+1 ids.
    unk = [len(vocab) for vocab in vocabs]
    field_dims = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc: dict[str, tuple[np.ndarray, np.ndarray, list[Any]]] = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users: list[Any] = []
        for n, row in enumerate(rows):
            for i, value in enumerate(raw(row)):
                X[n, i] = vocabs[i].get(value, unk[i]) + offsets[i]
            y[n] = row[_LABEL]
            users.append(row[_USER_ID])
        enc[name] = (X, y, users)

    return enc, int(sum(field_dims))
