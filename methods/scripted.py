"""ScriptedGenerator: a deterministic, non-LLM hypothesis generator.

Purpose: integration checkpoint I2 needs to run controller + real harness
+ executor end to end with ZERO LLM variance, so a failure there is
provably a wiring bug and not a bad model draw. This generator satisfies
that by emitting a fixed, hand-authored script of ten moves — no model
call, no randomness, no dependence on run state.

It deliberately does NOT implement controller.ports.GeneratorPort as-is:
that Protocol's `propose(state_card, target_slot)` takes the target slot
as an input chosen by the search policy, because a real (LLM) generator
proposes an idea FOR a slot it's told to attack. This generator instead
dictates its OWN slot order — that is the entire point of a canned
script — so `target_slot` is an output (inside the returned payload),
not an input. Wiring this into the real GeneratorPort seam, if needed,
is a thin adapter around this class; it is not this class's job.

ORDERING RATIONALE
-------------------
The organizers' convergence rule (epsilon=0.002, N=3, see
harness/HANDOFF.md) ends a run after three consecutive weak iterations.
High-variance structural changes therefore go FIRST, while the evidence
budget is fullest, and cheap tuning goes LAST, where self-termination
costing nothing is a feature: if the structural moves already found the
real gains, burning the last few iterations on a learning-rate sweep is
exactly the point at which the run SHOULD stop on its own.

WHAT IS DELIBERATELY NOT HERE: static side-feature injection (adding
video/user side-feature columns beyond the 5-field FM baseline). The
organizers' own vendor/kuairand-starter-kit/ablation_features.py exists
specifically to demonstrate this does NOT help on this dataset (primary
0.5940 for 13 fields vs 0.5950 for the 5-field baseline — noise, if
anything slightly worse; see harness/SCHEMA_NOTES.md's ablation
section). Do not re-add it here; the organizers already ran this
experiment so we don't have to.

Each move is a (target_slot, SlotConfig, HypothesisPayload) triple. This
module does not implement any move — it only proposes configs and
hypotheses; the executor (W3) realizes them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from contracts import Citation, HypothesisPayload, SlotConfig, SlotName


def _move(
    *,
    target_slot: SlotName,
    impl: str,
    params: dict[str, Any],
    rationale: str,
    citation: Citation,
    expected_gain: float,
    expected_cost_s: float,
) -> tuple[SlotConfig, HypothesisPayload]:
    slot_config = SlotConfig(impl=impl, params=params)
    hypothesis = HypothesisPayload(
        target_slot=target_slot,
        rationale=rationale,
        citation=citation,
        expected_gain=expected_gain,
        expected_cost_s=expected_cost_s,
        # This is a pre-planned script, not an adaptive one: every move
        # here was chosen from the initial data analysis (SCHEMA_NOTES,
        # the vendor README's own headroom list), never from observing a
        # specific predecessor's realized result. Leaving this empty is
        # the honest answer, not a placeholder — inventing config_ids
        # for candidates that were never actually run would misrepresent
        # what motivated the proposal.
        predecessor_evidence=(),
    )
    return slot_config, hypothesis


# ---------------------------------------------------------------------------
# The ten moves, in fixed order. See module docstring for why this order.
# ---------------------------------------------------------------------------

_MOVES: tuple[tuple[SlotConfig, HypothesisPayload], ...] = (
    # 1. baseline_reproduce — the control.
    _move(
        target_slot="model",
        impl="fm",
        params={"k": 16, "lr": 0.001, "epochs": 40},
        rationale=(
            "Reproduce the organizers' own published FM baseline exactly "
            "(k=16, lr=0.001), to confirm the harness end-to-end — data "
            "loading, encoding, training, evaluation — reproduces "
            "validation primary 0.6016 before any structural change is "
            "judged against it. This is the control, not a candidate "
            "expected to win."
        ),
        citation=Citation(
            key="rendle2010fm",
            url="https://ieeexplore.ieee.org/document/5694074",
            library_entry="methods/library/fm.yaml#factorization_machine",
        ),
        expected_gain=0.0,
        expected_cost_s=40.0,
    ),
    # 2. recency_weight_exp — strongest, data-grounded hypothesis.
    _move(
        target_slot="weighting",
        impl="exp_decay",
        params={"half_life_days": 5.0},
        rationale=(
            "Train volume is heavily front-loaded — 278,835 rows on "
            "20220411 decaying to ~20-24k/day by 20220418 — while "
            "validation is flat at 14-27k/day. Validation resembles the "
            "tail plateau, not the burst, so early training rows are "
            "drawn from a materially different regime. Downweighting "
            "them by recency should help the model fit the regime "
            "validation is actually drawn from."
        ),
        citation=Citation(
            key="koren2009temporal",
            url="https://dl.acm.org/doi/10.1145/1557019.1557072",
            library_entry="methods/library/recency_weighting.yaml#exponential_decay",
        ),
        expected_gain=0.008,
        expected_cost_s=45.0,
    ),
    # 3. recency_window — the blunt version of move 2.
    _move(
        target_slot="data_view",
        impl="recent_window",
        params={"days": 7},
        rationale=(
            "The blunt version of move 2: a hard cutoff so training only "
            "ever sees the plateau regime validation resembles, instead "
            "of continuously downweighting older rows. Kept separate "
            "from move 2 because a hard cutoff and a smooth decay can "
            "behave differently in practice — dropping rows outright "
            "also reduces user_id/video_id ID coverage, which the "
            "baseline's own diagnosis says carries most of the learnable "
            "signal, so this could underperform move 2 despite the same "
            "motivation."
        ),
        citation=Citation(
            key="kuairand_volume_shape_analysis",
            url="harness/HANDOFF.md",
            library_entry="methods/library/recency_window.yaml#hard_cutoff",
        ),
        expected_gain=0.005,
        expected_cost_s=35.0,
    ),
    # 4. multitask_longview_click — denser auxiliary signal.
    _move(
        target_slot="objective",
        impl="multitask_bce",
        params={"primary_label": "long_view", "auxiliary_label": "click", "auxiliary_weight": 0.3},
        rationale=(
            "KuaiRand ships 12 feedback signals (is_click, is_like, "
            "is_follow, is_comment, is_forward, play_time_ms, ...) and "
            "only long_view is scored. click is denser than long_view "
            "(more positives per user), so an auxiliary click head "
            "shares gradient signal into the shared representation "
            "before the sparser long_view head has to do all the work "
            "alone."
        ),
        citation=Citation(
            key="ma2018esmm",
            url="https://dl.acm.org/doi/10.1145/3209978.3210104",
            library_entry="methods/library/multitask.yaml#esmm_click_longview",
        ),
        expected_gain=0.008,
        expected_cost_s=60.0,
    ),
    # 5. fm_rank_k — cheap capacity check, low expectation by design.
    _move(
        target_slot="model",
        impl="fm",
        params={"k": 32, "lr": 0.001, "epochs": 40},
        rationale=(
            "Capacity ablation. The organizers' own sweep (k=8/16/32) "
            "already showed near-flat scores (0.5895/0.5902/0.5887) — "
            "the bottleneck is not capacity. This is a cheap "
            "confirmation, not a high-expectation move: kept to "
            "re-verify the finding still holds once earlier accepted "
            "changes have shifted the operating point, not because "
            "capacity is expected to matter now."
        ),
        citation=Citation(
            key="kuairand_capacity_ablation",
            url="harness/SCHEMA_NOTES.md",
            library_entry="methods/library/fm.yaml#capacity_k32",
        ),
        expected_gain=0.001,
        expected_cost_s=45.0,
    ),
    # 6. duration_debias — organizers' own flagged confound.
    _move(
        target_slot="calibration",
        impl="duration_debias_cwm",
        params={"method": "counterfactual_watch_time", "duration_field": "dur_bucket"},
        rationale=(
            "The scored label long_view is a watch-time threshold, so it "
            "is mechanically entangled with video duration — which the "
            "baseline already encodes only crudely as dur_bucket. The "
            "organizers' own reference [4], Counterfactual Watch Time "
            "(KDD 2024), flags this duration bias explicitly and treats "
            "watch time as needing a duration-conditional (censored) "
            "correction rather than being modeled directly."
        ),
        citation=Citation(
            key="cwm_kdd2024",
            url="https://github.com/hyz20/CWM",
            library_entry="methods/library/cwm.yaml#duration_debias",
        ),
        expected_gain=0.010,
        expected_cost_s=50.0,
    ),
    # 7. model_lightgbm — model-class swap.
    _move(
        target_slot="model",
        impl="lightgbm",
        params={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
        rationale=(
            "Swap FM for gradient boosting over the same 5 fields, "
            "testing whether non-linear tree splits capture interactions "
            "FM's bilinear form misses. The baseline's own diagnosis is "
            "that the user_id x video_id crossing already captures most "
            "learnable signal, so this is a genuine test of that ceiling "
            "rather than a guaranteed win."
        ),
        citation=Citation(
            key="ke2017lightgbm",
            url="https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
            library_entry="methods/library/lightgbm.yaml#gbdt_baseline",
        ),
        expected_gain=0.006,
        expected_cost_s=90.0,
    ),
    # 8. pairwise_loss — organizers' own top-ranked unexplored direction.
    _move(
        target_slot="objective",
        impl="bpr",
        params={"pairs_per_batch": 8192},
        rationale=(
            "GAUC and nDCG@5 are both ranking metrics evaluated within a "
            "user's own impressions, but the baseline trains pointwise "
            "BCE, which optimizes calibrated probability rather than "
            "relative order. A pairwise objective directly optimizes "
            "what's measured — this is also the direction the "
            "organizers themselves rank as most likely to help, ahead "
            "of behavioural-sequence and multi-task directions."
        ),
        citation=Citation(
            key="rendle2009bpr",
            url="https://arxiv.org/abs/1205.2618",
            library_entry="methods/library/pairwise.yaml#bpr",
        ),
        expected_gain=0.012,
        expected_cost_s=55.0,
    ),
    # 9. popularity_prior — cheap blend, real but modest mechanism.
    _move(
        target_slot="calibration",
        impl="popularity_blend",
        params={"prior": "item_popularity", "blend_weight": 0.2, "prior_smoothing": 20.0},
        rationale=(
            "Item popularity alone already reaches primary 0.5807 on "
            "validation. Popularity is a marginal (video-only) "
            "statistic while FM's user_id x video_id crossing is joint —"
            " the two are not the same signal, so blending FM's score "
            "with a popularity prior is cheap (no retraining) and has a "
            "plausible mechanism, even though it's a lower-variance move "
            "than the structural changes above."
        ),
        citation=Citation(
            key="kuairand_item_popularity_baseline",
            url="vendor/kuairand-starter-kit/baseline.py",
            library_entry="methods/library/popularity_blend.yaml#item_popularity_prior",
        ),
        expected_gain=0.004,
        expected_cost_s=5.0,
    ),
    # 10. tune_lr_epochs — deliberately last; cheap, reliably small gains.
    _move(
        target_slot="model",
        impl="fm",
        params={"lr": 0.0005, "epochs": 60, "patience": 6},
        rationale=(
            "Final hyperparameter sweep once the pipeline's structural "
            "shape is settled. The organizers' own choices already look "
            "well-tuned (published std 0.0008 across seeds is tight), so "
            "this is the lowest-expected-value move in the script. "
            "Deliberately last: if the structural moves above already "
            "found the real gains, self-terminating after three flat "
            "iterations here costs nothing worth having."
        ),
        citation=Citation(
            key="kuairand_fm_baseline_config",
            url="vendor/kuairand-starter-kit/baseline_scores.json",
            library_entry="methods/library/fm.yaml#hyperparameter_sweep",
        ),
        expected_gain=0.002,
        expected_cost_s=60.0,
    ),
)


class ScriptedGenerator:
    """Emits the ten moves above, in fixed order, one per `propose()` call.

    Deterministic and stateless beyond a position counter: the same
    instance always returns the same move for the same call count, and
    two fresh instances behave identically. `state` is accepted (to look
    like a generator call site expects) but never read — this generator
    does not adapt to the run, by design; see the module docstring.
    """

    def __init__(self) -> None:
        self._index = 0

    def propose(self, state: Any) -> tuple[SlotConfig, dict]:
        """Returns (slot_config, hypothesis_payload_dict) for the next
        scripted move. `hypothesis_payload_dict["target_slot"]` names
        which slot `slot_config` is for.

        Raises StopIteration once all ten moves have been emitted.
        """
        if self._index >= len(_MOVES):
            raise StopIteration(
                f"ScriptedGenerator exhausted after {len(_MOVES)} moves"
            )
        slot_config, hypothesis = _MOVES[self._index]
        self._index += 1
        return slot_config, dataclasses.asdict(hypothesis)

    def reset(self) -> None:
        """Restarts the script from move 1."""
        self._index = 0
