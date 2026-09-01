"""Guard: controller.BASELINE_SLOTS and executor.realize.DEFAULT_SLOTS must
keep describing the SAME baseline pipeline.

WHY THIS EXISTS. autonomy/adapters.py is a "diff and delegate" adapter. It
diffs an incoming PipelineConfig against the CONTROLLER's baseline to recover
the one changed slot, then re-splices that fragment onto the EXECUTOR's
baseline before delegating to executor.run.run_candidate. Both halves are
silent assumptions about definitions the adapter does not own and cannot see
change:

  - the diff assumes a candidate is "BASELINE_SLOTS with exactly one slot
    replaced", so if BASELINE_SLOTS drifts, `_diff_against_baseline` starts
    seeing 2+ differing slots and raises ExecutorAdapterError on candidates
    that are perfectly fine;
  - the re-splice assumes DEFAULT_SLOTS still fills the other five slots with
    the same baseline, so if DEFAULT_SLOTS drifts, delegation quietly trains
    something that is no longer the published baseline — and, worse, produces
    a different config_id, which silently misses the harness cache that
    scripts/run_agent.py populated.

Neither failure announces itself. That is what this file is for.

WHAT "THE SAME" MEANS HERE. The two are NOT equal dicts and are not supposed
to be: five of six slots use different impl vocabularies for one baseline,
owned by two workstreams. autonomy/adapters.py's module docstring writes that
correspondence down. This file pins it — so a rename, a param change, an added
slot, or a genuine divergence on either side fails a test instead of surfacing
as a confusing adapter error or a silent cache miss weeks later.

BOTH DEFINITIONS ARE READ ONLY HERE. This file asserts about them; the fix for
a failure is a conversation between the two workstreams, not an edit that makes
the test pass.
"""

from contracts import SLOT_ORDER
from controller.controller import BASELINE_SLOTS
from executor.realize import DEFAULT_SLOTS

import autonomy.adapters as adapters


# The documented correspondence, one row per slot:
#   slot -> (controller impl, executor impl)
# Sourced from autonomy/adapters.py's module docstring, which is the
# reviewable statement of the assumption. Changing a value here without
# changing the definition it mirrors is the failure this file exists to make
# loud, so do not "fix" a failure by editing this table.
BASELINE_CORRESPONDENCE = {
    "data_view": ("full_log", "full"),
    "features": ("five_field_categorical", "baseline_5"),
    "weighting": ("uniform", "none"),
    "model": ("fm", "fm"),
    "objective": ("logloss", "bce"),
    "calibration": ("none", "none"),
}


def test_both_baselines_fill_exactly_the_contract_slots():
    """Neither side may add, drop, or rename a slot unilaterally.

    A slot present in one and missing from the other breaks the adapter
    immediately: `_diff_against_baseline` indexes `base[slot]` for every slot
    in SLOT_ORDER, and build_config's overlay indexes DEFAULT_SLOTS.
    """
    assert set(BASELINE_SLOTS) == set(SLOT_ORDER)
    assert set(DEFAULT_SLOTS) == set(SLOT_ORDER)


def test_the_two_vocabularies_still_match_the_documented_correspondence():
    """Each slot's impl pair is the one autonomy/adapters.py documents.

    This is the drift alarm. If either definition renames an impl, the pair
    stops matching and this fails naming the slot.
    """
    for slot in SLOT_ORDER:
        expected_controller, expected_executor = BASELINE_CORRESPONDENCE[slot]
        assert BASELINE_SLOTS[slot].impl == expected_controller, (
            f"controller.BASELINE_SLOTS[{slot!r}].impl drifted: expected "
            f"{expected_controller!r}, found {BASELINE_SLOTS[slot].impl!r}. "
            "If this is intentional, autonomy/adapters.py's diff assumption and "
            "its docstring table both need updating."
        )
        assert DEFAULT_SLOTS[slot].impl == expected_executor, (
            f"executor.realize.DEFAULT_SLOTS[{slot!r}].impl drifted: expected "
            f"{expected_executor!r}, found {DEFAULT_SLOTS[slot].impl!r}. "
            "If this is intentional, the adapter's re-splice now produces a "
            "different config_id and will miss the harness cache."
        )


def test_exactly_five_of_six_slots_differ_as_the_adapter_docstring_says():
    """Pins "five of six differ" from the adapter docstring.

    Broken out precisely, because the count mixes two kinds of difference:
    four slots differ by IMPL NAME, and `model` differs only by PARAMS (both
    say "fm"; the controller's copy also carries `epochs`). Only `calibration`
    is identical.

    Stated as a count so that a change making the vocabularies converge — a
    good thing, and one that would let the adapter drop its translation — also
    trips this and gets noticed, rather than silently leaving dead translation
    code in place.
    """
    differ_by_impl = {
        slot for slot in SLOT_ORDER if BASELINE_SLOTS[slot].impl != DEFAULT_SLOTS[slot].impl
    }
    assert differ_by_impl == {"data_view", "features", "weighting", "objective"}

    # model: same impl, different params.
    assert BASELINE_SLOTS["model"].impl == DEFAULT_SLOTS["model"].impl
    assert BASELINE_SLOTS["model"].params != DEFAULT_SLOTS["model"].params

    identical = {
        slot for slot in SLOT_ORDER if BASELINE_SLOTS[slot] == DEFAULT_SLOTS[slot]
    }
    assert identical == {"calibration"}, (
        f"exactly one slot should be spelled identically; found {sorted(identical)}"
    )
    assert len(SLOT_ORDER) - len(identical) == 5


def test_the_model_slot_agrees_on_every_shared_hyperparameter():
    """The one slot that MUST agree numerically.

    Both sides name the organizers' FM at k=16, lr=0.001. The controller's
    copy also carries `epochs`, which the executor's does not — that
    asymmetry is allowed. What is not allowed is the two disagreeing on a
    parameter they both carry: that would mean the Controller believes it
    evaluated a baseline the executor never trained.
    """
    assert BASELINE_SLOTS["model"].impl == DEFAULT_SLOTS["model"].impl == "fm"

    controller_params = BASELINE_SLOTS["model"].params
    executor_params = DEFAULT_SLOTS["model"].params
    shared = set(controller_params) & set(executor_params)

    assert shared, "the two model params share no keys at all; one of them was rewritten"
    assert {"k", "lr"} <= shared

    for key in sorted(shared):
        assert controller_params[key] == executor_params[key], (
            f"model param {key!r} disagrees: controller={controller_params[key]!r}, "
            f"executor={executor_params[key]!r}. These describe one published "
            "baseline (k=16, lr=0.001) and must not diverge."
        )

    # The published figures, pinned directly so a coordinated edit to both
    # sides still has to face them.
    assert controller_params["k"] == 16
    assert controller_params["lr"] == 0.001


def test_calibration_is_the_one_slot_both_spell_the_same_way():
    """Both say "none". The adapter's translation is a no-op here, and a
    change on either side would make it stop being one."""
    assert BASELINE_SLOTS["calibration"].impl == DEFAULT_SLOTS["calibration"].impl == "none"


def test_executor_baseline_weighting_is_none_so_the_bpr_move_stays_legal():
    """A composition the re-splice makes legal, pinned.

    executor/realize.py refuses objective="bpr" combined with any weighting
    other than "none". BASELINE_SLOTS says weighting="uniform", so a candidate
    diffed out of the Controller's vocabulary would be illegal if delegated
    as-is; it is legal only because the fragment is re-spliced onto
    DEFAULT_SLOTS, where weighting IS "none". If DEFAULT_SLOTS ever carries a
    real weighting, move 8 starts raising NotImplementedError mid-run.
    """
    assert DEFAULT_SLOTS["weighting"].impl == "none"
    assert BASELINE_SLOTS["weighting"].impl != "none"


def test_the_adapter_docstring_still_states_this_correspondence():
    """The table above mirrors prose in autonomy/adapters.py. If that prose is
    edited away or changed, the mirror is stale and this file is guarding a
    claim the codebase no longer makes."""
    doc = adapters.__doc__
    assert doc, "autonomy/adapters.py lost its module docstring"

    for slot, (controller_impl, executor_impl) in BASELINE_CORRESPONDENCE.items():
        row = [line for line in doc.splitlines() if line.strip().startswith(slot)]
        assert row, f"the adapter docstring no longer has a table row for {slot!r}"
        line = row[0]
        if slot == "model":
            # Written as fm{k,lr,epochs} / fm{k,lr} rather than as bare impls.
            assert "fm{" in line
        else:
            assert f'"{controller_impl}"' in line, f"{slot}: docstring lost {controller_impl!r}"
            assert f'"{executor_impl}"' in line, f"{slot}: docstring lost {executor_impl!r}"


def test_this_guard_reads_both_definitions_without_mutating_them():
    """Cheap belt-and-braces: the asserts above index into the real module
    dicts, so an accidental in-place edit here would leak into every other
    test in the session. Confirms the two are still the published values after
    this module has finished touching them."""
    assert BASELINE_SLOTS["model"].params.get("epochs") == 40
    assert "epochs" not in DEFAULT_SLOTS["model"].params
    assert len(BASELINE_SLOTS) == len(DEFAULT_SLOTS) == len(SLOT_ORDER)
