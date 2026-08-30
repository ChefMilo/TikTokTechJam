"""Tests for executor.realize's moves 2 (exp_decay weighting) and 3
(recent_window data_view). Uses small synthetic rows throughout so these
run in seconds, not minutes — the real, full-scale numbers are produced
separately by scripts/compare_moves.py.
"""

import numpy as np
import pytest

from contracts import PipelineConfig, SlotConfig
from executor import realize


def _row(date, user, label, video="v", tab="t"):
    return (date, user, video, "author", tab, 10.0, label)


def test_recency_window_keeps_only_the_last_n_days_inclusive():
    rows = [
        _row(20220410, "u1", 1),
        _row(20220412, "u2", 0),
        _row(20220415, "u3", 1),
        _row(20220417, "u4", 0),
    ]

    windowed = realize._apply_recency_window(rows, {"days": 3})

    assert [row[0] for row in windowed] == [20220415, 20220417]


def test_recency_window_defaults_to_seven_days():
    rows = [_row(20220401 + i, f"u{i}", i % 2) for i in range(10)]  # dates 20220401..20220410

    windowed = realize._apply_recency_window(rows, {})

    assert min(row[0] for row in windowed) == 20220404  # last 7 of 20220401-20220410


def test_recency_window_is_correct_across_a_month_boundary():
    rows = [_row(20220328, "u1", 1), _row(20220330, "u2", 0), _row(20220401, "u3", 1)]

    windowed = realize._apply_recency_window(rows, {"days": 3})

    # Naive integer arithmetic (20220401 - 2 = 20220399) would wrongly
    # keep everything; real date arithmetic must exclude 20220328.
    assert [row[0] for row in windowed] == [20220330, 20220401]


def test_recency_window_never_filters_score_rows():
    """Only realize()'s dispatch is responsible for this, since
    _apply_recency_window only ever receives fit_rows — asserted here by
    checking the function's own contract: it filters exactly what it's
    given, nothing more."""
    rows = [_row(20220410, "u1", 1), _row(20220417, "u2", 0)]
    windowed = realize._apply_recency_window(rows, {"days": 1})
    assert windowed == [_row(20220417, "u2", 0)]


def test_exp_decay_weights_match_the_formula():
    rows = [_row(20220410, "u1", 1), _row(20220412, "u2", 0), _row(20220417, "u3", 1)]

    weights = realize._compute_exp_decay_weights(rows, {"half_life_days": 5.0})

    expected = np.array([0.5 ** (7 / 5.0), 0.5 ** (5 / 5.0), 0.5 ** (0 / 5.0)], dtype=np.float32)
    np.testing.assert_allclose(weights, expected, rtol=1e-6)


def test_exp_decay_weights_default_half_life_is_five_days():
    rows = [_row(20220412, "u1", 1), _row(20220417, "u2", 0)]

    weights = realize._compute_exp_decay_weights(rows, {})

    assert weights[1] == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.5 ** 1.0)


def _synthetic_rows(n_users=60, rows_per_user=4, dates=(20220410, 20220411, 20220412, 20220413)):
    rng = np.random.default_rng(0)
    rows = []
    for u in range(n_users):
        for i in range(rows_per_user):
            date = dates[i % len(dates)]
            label = int(rng.integers(0, 2))
            rows.append((date, f"u{u}", f"v{u}_{i}", f"a{u}", "tab1", 10.0, label))
    return rows


def test_exp_decay_with_absurd_half_life_matches_unweighted_baseline():
    """The required sanity check: half_life_days=1e6 makes every weight
    ~1.0, so results must match the unweighted baseline to within
    floating-point noise. If this fails, the weighting is not actually
    being applied (or is applied with a normalization bug) — it would
    NOT reduce to the unweighted case just because the weights happen to
    be near-uniform.
    """
    fit_rows = _synthetic_rows()
    score_rows = fit_rows[:80]
    model_params = {"k": 4, "lr": 0.01, "epochs": 5, "patience": 2}
    model_slot = SlotConfig(impl="fm", params=model_params)
    seed = 0

    _, _, unweighted_scores = realize._realize_fm(model_slot, fit_rows, score_rows, seed)

    weights = realize._compute_exp_decay_weights(fit_rows, {"half_life_days": 1e6})
    _, _, weighted_scores = realize._realize_fm(
        model_slot, fit_rows, score_rows, seed, sample_weights=weights
    )

    np.testing.assert_allclose(weighted_scores, unweighted_scores, rtol=1e-3, atol=1e-4)


def test_exp_decay_with_real_half_life_diverges_from_unweighted():
    """The complement of the sanity check above: a real (small)
    half_life must actually change the result — otherwise the weighting
    code path is a no-op dressed up as a feature."""
    fit_rows = _synthetic_rows()
    score_rows = fit_rows[:80]
    model_params = {"k": 4, "lr": 0.01, "epochs": 5, "patience": 2}
    model_slot = SlotConfig(impl="fm", params=model_params)
    seed = 0

    _, _, unweighted_scores = realize._realize_fm(model_slot, fit_rows, score_rows, seed)

    weights = realize._compute_exp_decay_weights(fit_rows, {"half_life_days": 1.0})
    _, _, weighted_scores = realize._realize_fm(
        model_slot, fit_rows, score_rows, seed, sample_weights=weights
    )

    assert not np.allclose(weighted_scores, unweighted_scores, rtol=1e-3, atol=1e-4)


def test_realize_dispatches_recent_window_and_exp_decay():
    fit_rows = _synthetic_rows()
    score_rows = fit_rows[:40]

    slots_window = dict(realize.DEFAULT_SLOTS)
    slots_window["model"] = SlotConfig(impl="fm", params={"k": 4, "lr": 0.01, "epochs": 3, "patience": 1})
    slots_window["data_view"] = SlotConfig(impl="recent_window", params={"days": 2})
    config = PipelineConfig(slots=slots_window, seed=0)

    user_ids, labels, scores = realize.realize(config, fit_rows, score_rows, seed=0)
    assert len(user_ids) == len(score_rows)

    slots_weight = dict(realize.DEFAULT_SLOTS)
    slots_weight["model"] = SlotConfig(impl="fm", params={"k": 4, "lr": 0.01, "epochs": 3, "patience": 1})
    slots_weight["weighting"] = SlotConfig(impl="exp_decay", params={"half_life_days": 5.0})
    config2 = PipelineConfig(slots=slots_weight, seed=0)

    user_ids2, labels2, scores2 = realize.realize(config2, fit_rows, score_rows, seed=0)
    assert len(user_ids2) == len(score_rows)


def test_realize_raises_not_implemented_for_unknown_slot_impls():
    fit_rows = _synthetic_rows()
    score_rows = fit_rows[:20]

    slots = dict(realize.DEFAULT_SLOTS)
    slots["model"] = SlotConfig(impl="lightgbm", params={})
    config = PipelineConfig(slots=slots, seed=0)
    with pytest.raises(NotImplementedError, match="lightgbm"):
        realize.realize(config, fit_rows, score_rows, seed=0)

    slots2 = dict(realize.DEFAULT_SLOTS)
    slots2["data_view"] = SlotConfig(impl="something_else", params={})
    config2 = PipelineConfig(slots=slots2, seed=0)
    with pytest.raises(NotImplementedError, match="something_else"):
        realize.realize(config2, fit_rows, score_rows, seed=0)

    slots3 = dict(realize.DEFAULT_SLOTS)
    slots3["weighting"] = SlotConfig(impl="something_else", params={})
    config3 = PipelineConfig(slots=slots3, seed=0)
    with pytest.raises(NotImplementedError, match="something_else"):
        realize.realize(config3, fit_rows, score_rows, seed=0)


# ---------------------------------------------------------------------------
# Move 8: pairwise_loss (objective slot, impl "bpr")
# ---------------------------------------------------------------------------


def test_one_positive_three_negatives_produces_exactly_three_pairs():
    users = np.array(["u1", "u1", "u1", "u1"])
    labels = np.array([1, 0, 0, 0])

    index = realize._build_bpr_pair_index(users, labels)

    assert len(index.eligible_codes) == 1
    code = index.eligible_codes[0]
    assert index.pos_counts[code] * index.neg_counts[code] == 3


def test_all_positive_or_all_negative_user_produces_zero_pairs():
    all_positive = realize._build_bpr_pair_index(np.array(["u1", "u1", "u1"]), np.array([1, 1, 1]))
    assert len(all_positive.eligible_codes) == 0

    all_negative = realize._build_bpr_pair_index(np.array(["u2", "u2"]), np.array([0, 0]))
    assert len(all_negative.eligible_codes) == 0


def test_sample_bpr_pairs_raises_when_no_user_is_eligible():
    index = realize._build_bpr_pair_index(np.array(["u1", "u1"]), np.array([1, 1]))
    with pytest.raises(ValueError, match="no user has both"):
        realize._sample_bpr_pairs(index, 10, np.random.default_rng(0))


def test_sampled_pairs_always_pair_a_positive_with_a_negative_from_the_same_user():
    users = np.array(["u1", "u1", "u2", "u2", "u2", "u3"])
    labels = np.array([1, 0, 1, 1, 0, 1])  # u3 has no negative — ineligible
    index = realize._build_bpr_pair_index(users, labels)

    pos_idx, neg_idx = realize._sample_bpr_pairs(index, 200, np.random.default_rng(0))

    assert np.all(labels[pos_idx] == 1)
    assert np.all(labels[neg_idx] == 0)
    assert np.all(users[pos_idx] == users[neg_idx])  # same user on both sides of every pair
    assert not np.any(users[pos_idx] == "u3")  # ineligible user never drawn


def test_pairwise_training_on_trivially_separable_data_ranks_positive_above_negative():
    """The required sanity check: a video that's always the positive
    label and one that's always the negative label, repeated across many
    users, is a trivial pattern for FM's per-field embeddings to pick up
    — if step() were wired wrong (wrong sign, wrong pairing), training
    would not reliably separate them."""
    rows = []
    for u in range(40):
        rows.append((20220410, f"u{u}", "v_pos", "a_pos", "t", 10.0, 1))
        rows.append((20220410, f"u{u}", "v_neg", "a_neg", "t", 10.0, 0))

    model_slot = SlotConfig(impl="fm", params={"k": 4, "lr": 0.05, "epochs": 30, "patience": 6})
    user_ids, labels, scores = realize._realize_fm_pairwise(
        model_slot, rows, rows, seed=0, objective_params={"pairs_per_batch": 64}
    )

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    assert pos_scores.mean() > neg_scores.mean()


def test_realize_dispatches_bpr_objective():
    fit_rows = _synthetic_rows()
    score_rows = fit_rows[:40]

    slots = dict(realize.DEFAULT_SLOTS)
    slots["model"] = SlotConfig(impl="fm", params={"k": 4, "lr": 0.01, "epochs": 3, "patience": 1})
    slots["objective"] = SlotConfig(impl="bpr", params={"pairs_per_batch": 32})
    config = PipelineConfig(slots=slots, seed=0)

    user_ids, labels, scores = realize.realize(config, fit_rows, score_rows, seed=0)
    assert len(user_ids) == len(score_rows)


def test_realize_rejects_bpr_combined_with_non_default_weighting():
    fit_rows = _synthetic_rows()
    score_rows = fit_rows[:40]

    slots = dict(realize.DEFAULT_SLOTS)
    slots["objective"] = SlotConfig(impl="bpr", params={"pairs_per_batch": 32})
    slots["weighting"] = SlotConfig(impl="exp_decay", params={"half_life_days": 5.0})
    config = PipelineConfig(slots=slots, seed=0)

    with pytest.raises(NotImplementedError, match="bpr"):
        realize.realize(config, fit_rows, score_rows, seed=0)
