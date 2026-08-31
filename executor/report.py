"""Renders a journal into the four deliverable artifacts under
artifacts/report/: iterations.md (deliverable #3), results.md
(deliverable #4's table), trajectory.csv, and forecast_calibration.md.

SCOPE DISCIPLINE: reads whatever a journal actually contains and renders
it faithfully; it does not require a full controller loop to exist first
— the tests exercise it against a synthetic journal for exactly that
reason.

PAIRING NOTE: HYPOTHESIS is paired to its resulting DECISION/EVAL_RESULT
by NODE, not by the `iteration` field, even though forecast calibration
is conceptually "does the forecast match what got decided". `iteration`
counts COMMITTED revisions (contracts.JournalEvent's own docstring) and
only advances once a DECISION accepts — a HYPOTHESIS logged before that
decision cannot correctly claim the post-acceptance iteration number
without corrupting what `iteration` means for every other reader of the
log (a rejected hypothesis must NOT have advanced it). `node` is stable
across accept/reject and is shared by every event describing the same
evaluation attempt, so it's the correct join key here.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from contracts import EventKind, JournalEvent
from executor.journal import Journal
from executor.realize import DEFAULT_SLOTS

BASELINE_GAUC = 0.6674
BASELINE_NDCG5 = 0.5357
BASELINE_PRIMARY = 0.6016
ITERATION_CAP = 50


def _group_by_node(events: list[JournalEvent]) -> dict[int, dict[EventKind, list[JournalEvent]]]:
    by_node: dict[int, dict[EventKind, list[JournalEvent]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        by_node[event.node][event.kind].append(event)
    return by_node


def _eval_result_metrics(event: JournalEvent) -> dict[str, Any]:
    """The ONE place EVAL_RESULT's payload is read for metrics — every
    other function in this module goes through this rather than
    re-detecting the shape itself.

    Two shapes exist in the wild:

    - The durable helper's (executor/journal.py's log_eval_result): a
      `per_seed` dict of {seed: {"values": {"GAUC": ..., "nDCG@5": ...},
      "primary": ...}}. GAUC/nDCG@5/primary are each the mean across
      seeds.
    - The Controller's own (controller/controller.py's `_emit` at its
      EVAL_RESULT sites): FLAT — {config_id, status, primary,
      wall_seconds, gpu_seconds, tokens}. No per-seed breakdown and no
      GAUC/nDCG@5 at all; `primary` is already the Controller's own
      seed-blended mean (see `_mean_primary` in controller/controller.py).

    Returns {"gauc": float|None, "ndcg5": float|None, "primary":
    float|None, "per_seed": dict|None} — `per_seed` is None for the flat
    shape, which is how callers tell "no per-seed table to render" from
    "a per-seed table with nothing in it".
    """
    payload = event.payload
    per_seed = payload.get("per_seed") or None
    if per_seed:
        primaries = [seed_data["primary"] for seed_data in per_seed.values()]
        gauc_values = [
            seed_data["values"]["GAUC"] for seed_data in per_seed.values() if "GAUC" in seed_data["values"]
        ]
        ndcg_values = [
            seed_data["values"]["nDCG@5"] for seed_data in per_seed.values() if "nDCG@5" in seed_data["values"]
        ]
        return {
            "gauc": sum(gauc_values) / len(gauc_values) if gauc_values else None,
            "ndcg5": sum(ndcg_values) / len(ndcg_values) if ndcg_values else None,
            "primary": sum(primaries) / len(primaries) if primaries else None,
            "per_seed": per_seed,
        }
    # Controller's flat shape: one blended primary, no GAUC/nDCG@5.
    return {"gauc": None, "ndcg5": None, "primary": payload.get("primary"), "per_seed": None}


def _mean_across_seeds(eval_result: JournalEvent, key: str) -> Optional[float]:
    metrics = _eval_result_metrics(eval_result)
    if key == "primary":
        return metrics["primary"]
    if key == "GAUC":
        return metrics["gauc"]
    if key == "nDCG@5":
        return metrics["ndcg5"]
    return None


def _node_was_accepted(kinds: dict[EventKind, list[JournalEvent]]) -> bool:
    decisions = kinds.get(EventKind.DECISION, [])
    if not decisions:
        # No gate ruling at all is the baseline-adoption case (nothing to
        # compare against yet) — treated as accepted, matching the real
        # controller's own baseline special-case.
        return True
    return bool(decisions[-1].payload.get("verdict"))


def _diff_fragment(
    previous: Optional[tuple[str, dict[str, Any]]], impl: str, params: dict[str, Any]
) -> str:
    if previous is None:
        return f"impl: {impl!r} (first change to this slot; baseline default shown for reference)"
    prev_impl, prev_params = previous
    if prev_impl == impl and prev_params == params:
        return "no change from the last candidate that touched this slot"
    parts = []
    if prev_impl != impl:
        parts.append(f"impl: {prev_impl!r} -> {impl!r}")
    if prev_params != params:
        parts.append(f"params: {prev_params} -> {params}")
    return "; ".join(parts)


def _format_hypothesis(event: JournalEvent) -> list[str]:
    payload = event.payload
    citation = payload.get("citation") or {}
    lines = [
        f"**Hypothesis** (target_slot=`{payload.get('target_slot')}`)",
        "",
        f"- Rationale: {payload.get('rationale')}",
        f"- Citation: [{citation.get('key')}]({citation.get('url')}) ({citation.get('library_entry')})",
        f"- Expected gain: {payload.get('expected_gain'):+.4f}",
        f"- Expected cost: {payload.get('expected_cost_s')}s",
    ]
    predecessors = payload.get("predecessor_evidence") or []
    if predecessors:
        lines.append(f"- Predecessor evidence: {', '.join(predecessors)}")
    return lines


def _format_metrics_table(per_seed: dict) -> list[str]:
    lines = ["| seed | GAUC | nDCG@5 | primary |", "|---|---|---|---|"]
    for seed in sorted(per_seed, key=int):
        values = per_seed[seed]["values"]
        primary = per_seed[seed]["primary"]
        gauc = values.get("GAUC")
        ndcg = values.get("nDCG@5")
        lines.append(
            f"| {seed} | {gauc:.4f} | {ndcg:.4f} | {primary:.4f} |"
            if gauc is not None and ndcg is not None
            else f"| {seed} | - | - | {primary:.4f} |"
        )
    return lines


def _format_eval_result(event: JournalEvent) -> list[str]:
    payload = event.payload
    lines = [f"**Result** (config_id=`{payload.get('config_id')}`)", ""]
    metrics = _eval_result_metrics(event)
    per_seed = metrics["per_seed"]
    if per_seed is not None:
        lines.append("Validation:")
        lines.append("")
        lines.extend(_format_metrics_table(per_seed))
        # backtest_per_seed is optional enrichment (see log_eval_result's
        # docstring) — rendered alongside validation, not instead of it,
        # because a DECISION's accept reason can read "backtest confirms"
        # and a judge has no way to check that claim without seeing the
        # numbers it refers to. Only the durable helper's shape carries
        # this; the Controller's flat EVAL_RESULT has no backtest split.
        backtest_per_seed = payload.get("backtest_per_seed") or {}
        if backtest_per_seed:
            lines.append("")
            lines.append("Backtest:")
            lines.append("")
            lines.extend(_format_metrics_table(backtest_per_seed))
    else:
        # Controller's flat shape: no per-seed/GAUC/nDCG@5 breakdown to
        # tabulate, only one blended primary and a status.
        primary = metrics["primary"]
        primary_str = f"{primary:.4f}" if primary is not None else "n/a"
        lines.append(f"status: {payload.get('status')}")
        lines.append(f"primary: {primary_str}")
    lines.append("")
    lines.append(f"wall_seconds: {payload.get('wall_seconds')}")
    # gpu_seconds/tokens are top-level on EVAL_RESULT in both shapes (see
    # executor/journal.py's log_eval_result and controller/controller.py's
    # EVAL_RESULT emit — same field names, deliberately). Only shown when
    # present, so a journal from before either shape carried them doesn't
    # grow a spurious "gpu_seconds: None" line.
    gpu_seconds = payload.get("gpu_seconds")
    if gpu_seconds is not None:
        lines.append(f"gpu_seconds: {gpu_seconds}")
    tokens = payload.get("tokens")
    if tokens is not None:
        lines.append(f"tokens: {tokens}")
    return lines


def _format_decision(event: JournalEvent) -> list[str]:
    payload = event.payload
    verdict_str = "ACCEPTED" if payload.get("verdict") else "REJECTED"
    ci95 = payload.get("ci95") or [None, None]
    backtest_delta = payload.get("backtest_delta")
    lines = [
        f"**Verdict: {verdict_str}**",
        "",
        f"- delta: {payload.get('delta_primary'):+.4f} (n_seeds={payload.get('n_seeds')})",
        f"- ci95: [{ci95[0]:.4f}, {ci95[1]:.4f}]" if ci95[0] is not None else "- ci95: n/a",
        f"- backtest_delta: {backtest_delta:+.5f}" if backtest_delta is not None else "- backtest_delta: n/a",
        f"- reason: {payload.get('reason')}",
    ]
    return lines


def _format_convergence_check(event: JournalEvent) -> str:
    payload = event.payload
    if "delta" in payload:
        # The durable helper's shape (executor/journal.py's
        # log_convergence_check, used by e.g. scripts/run_agent.py): one
        # delta checked against one epsilon.
        delta = payload.get("delta")
        epsilon = payload.get("epsilon")
        cleared = "cleared" if payload.get("clears_epsilon") else "did NOT clear"
        return f"**Convergence check**: delta {delta:+.5f} vs epsilon {epsilon:.3f} — {cleared}"

    # The Controller's own shape (controller/controller.py's
    # _check_convergence): two rules assessed over a window of committed
    # revisions, no single "delta" — see controller/convergence.py's
    # ConvergenceStatus for the full field set this mirrors.
    converged = payload.get("converged")
    by_rule = payload.get("by_rule")
    epsilon = payload.get("epsilon")
    iterations_considered = payload.get("iterations_considered")
    n_required = payload.get("n_required")
    recent_deltas = payload.get("recent_deltas") or []
    deltas_str = ", ".join(f"{d:+.5f}" for d in recent_deltas) if recent_deltas else "none yet"
    status = f"CONVERGED (by {by_rule})" if converged else "not converged"
    return (
        f"**Convergence check**: {status} — "
        f"iterations_considered={iterations_considered}/{n_required}, "
        f"epsilon={epsilon}, recent_deltas=[{deltas_str}]"
    )


def _is_baseline_adoption(reason: Optional[str]) -> bool:
    """True for a DECISION reason naming the trivial "nothing to compare
    against yet" case — accepted by construction, not because it beat
    anything. Two different wordings exist for the same case: the durable
    helper's scripts write "adopted as initial incumbent; no prior
    candidate to compare against" (see scripts/run_agent.py), while
    controller/controller.py's own first-candidate branch writes "first
    candidate adopted as incumbent; nothing to compare against". Both
    share "compare against"; neither shares "initial incumbent" with the
    other, so that phrase alone (an earlier version of this check) missed
    the Controller's wording.
    """
    return "compare against" in (reason or "").lower()


def _recovered_nodes(events: list[JournalEvent]) -> set[int]:
    """Nodes whose ERROR is known to have been survived by the run.

    Two ways to know that, matching the two journal shapes this module
    reads: an explicit RECOVERY event at the same node (the durable
    helper's convention — see executor/journal.py's log_recovery), or the
    journal reaching a terminal FINALIZE/RUN_END at all. The Controller
    never emits RECOVERY (controller/controller.py's docstring: "A failed
    candidate must never stop the run... Log what broke, count the node,
    move on" — the loop continuing IS the recovery, with no separate event
    for it), so without this second check every Controller-run error
    would misreport as unrecovered even when the run finished cleanly.
    Journal.replay stops at the first line it can't decode (a real crash
    leaves a torn last line), so a FINALIZE/RUN_END actually being present
    is sound proof the run survived past every error before it — a real
    unhandled crash would never reach one.
    """
    explicit = {e.node for e in events if e.kind is EventKind.RECOVERY}
    if any(e.kind in (EventKind.FINALIZE, EventKind.RUN_END) for e in events):
        return explicit | {e.node for e in events if e.kind is EventKind.ERROR}
    return explicit


def _error_class_and_excerpt(payload: dict[str, Any]) -> tuple[str, str]:
    """Normalizes an ERROR payload's two display fields across the shapes
    in the wild. The durable helper (executor/journal.py's log_error)
    always writes {"error_class", "excerpt", ...}. The Controller
    (controller/controller.py) writes several different shapes depending
    on which failure path fired: {"config_id", "error_class",
    "error_excerpt"} for a failed executor result; {"reason", "stage",
    "detail"} for generator_exhausted, which has no error_class at all —
    running out of scripted hypotheses is a normal end of run, not a
    classified failure; {"config_id", "reason", "requested_slot",
    "proposed_slot", ...} for generator_slot_mismatch, likewise no
    error_class. `reason` is the fallback classification label when no
    error_class exists; `error_excerpt`/`detail`/`reason` are tried in
    that order for the excerpt when `excerpt` itself is absent.
    """
    error_class = payload.get("error_class") or payload.get("reason") or "unknown"
    excerpt = payload.get("excerpt") or payload.get("error_excerpt") or payload.get("detail")
    if excerpt is None:
        excerpt = payload.get("reason") or "n/a"
    return error_class, excerpt


def _format_errors_table(events: list[JournalEvent]) -> list[str]:
    """A run-level summary of every ERROR event, independent of the
    per-node Error/Recovery lines _write_iterations_md already renders
    in each node's own section — this is the "can a judge tell at a
    glance how many candidates failed and whether the run survived
    them" view, not a replacement for the per-node narrative.
    """
    errors = [e for e in events if e.kind is EventKind.ERROR]
    if not errors:
        return []
    recovered_nodes = _recovered_nodes(events)
    lines = ["## Errors and recovery", ""]
    lines.append("| node | error_class | policy | recovered |")
    lines.append("|---|---|---|---|")
    for error in errors:
        error_class, _ = _error_class_and_excerpt(error.payload)
        policy = error.payload.get("policy", "n/a")
        recovered = "yes" if error.node in recovered_nodes else "no"
        lines.append(f"| {error.node} | {error_class} | {policy} | {recovered} |")
    n_recovered = sum(1 for e in errors if e.node in recovered_nodes)
    lines.append("")
    if n_recovered == len(errors):
        lines.append(f"{len(errors)} candidate(s) failed; the run continued past every one.")
    else:
        lines.append(
            f"{len(errors)} candidate(s) failed; {n_recovered} of them had a logged RECOVERY "
            f"— the remaining {len(errors) - n_recovered} did not."
        )
    lines.append("")
    return lines


def _write_iterations_md(events: list[JournalEvent], by_node: dict, out_dir: Path) -> None:
    lines = [
        "# Iterations",
        "",
        "_`node` counts every evaluation attempted; `iteration` counts "
        "committed revisions and only advances when a DECISION accepts. "
        "Several consecutive nodes sharing one iteration number means "
        "several consecutive rejections, not a stall — see "
        "executor/journal.py's log_decision docstring._",
        "",
    ]
    lines.extend(_format_errors_table(events))
    last_fragment_by_slot: dict[str, tuple[str, dict[str, Any]]] = {}

    for node in sorted(by_node):
        kinds = by_node[node]
        if not any(
            k in kinds for k in (EventKind.HYPOTHESIS, EventKind.EVAL_RESULT, EventKind.ERROR)
        ):
            continue  # a node with only e.g. STAGE_CHANGE has nothing to report here

        decisions = kinds.get(EventKind.DECISION, [])
        iteration = decisions[-1].iteration if decisions else (kinds.get(EventKind.EVAL_RESULT) or kinds.get(EventKind.HYPOTHESIS))[0].iteration
        lines.append(f"## Node {node} (iteration {iteration})")
        lines.append("")

        for hyp in kinds.get(EventKind.HYPOTHESIS, []):
            lines.extend(_format_hypothesis(hyp))
            lines.append("")

        for result in kinds.get(EventKind.EVAL_RESULT, []):
            target_slot = result.payload.get("target_slot")
            impl = result.payload.get("fragment_impl")
            params = result.payload.get("fragment_params")
            if target_slot and impl is not None:
                previous = last_fragment_by_slot.get(target_slot)
                if previous is None and target_slot in DEFAULT_SLOTS:
                    default = DEFAULT_SLOTS[target_slot]
                    previous = (default.impl, default.params)
                lines.append(f"**Config diff vs parent** (slot: `{target_slot}`)")
                lines.append("")
                lines.append(f"- {_diff_fragment(previous, impl, params or {})}")
                lines.append("")
                last_fragment_by_slot[target_slot] = (impl, params or {})
            lines.extend(_format_eval_result(result))
            lines.append("")

        for decision in decisions:
            lines.extend(_format_decision(decision))
            lines.append("")

        for check in kinds.get(EventKind.CONVERGENCE_CHECK, []):
            lines.append(_format_convergence_check(check))
            lines.append("")

        for error in kinds.get(EventKind.ERROR, []):
            error_class, excerpt = _error_class_and_excerpt(error.payload)
            lines.append(f"**Error**: `{error_class}` — {excerpt}")
            lines.append("")

        for recovery in kinds.get(EventKind.RECOVERY, []):
            # A clean sentence, not the raw payload dict — this line is
            # what tells a reader "the taxonomy handled this on purpose"
            # rather than "something crashed and here's the debug dump".
            # Pairs with the Error line above it: error_class names WHAT
            # went wrong, this names WHAT WAS DONE about it.
            policy = recovery.payload.get("policy")
            message = recovery.payload.get("message")
            lines.append(f"**Recovery** (policy=`{policy}`): {message}")
            lines.append("")

        for blocked in kinds.get(EventKind.SLOT_BLOCKED, []):
            lines.append(
                f"**Circuit breaker**: slot `{blocked.payload.get('target_slot')}` blocked "
                f"after {blocked.payload.get('consecutive_failures')} consecutive failures."
            )
            lines.append("")

        lines.append("---")
        lines.append("")

    (out_dir / "iterations.md").write_text("\n".join(lines), encoding="utf-8")


def _select_best_node(by_node: dict) -> Optional[tuple[int, JournalEvent]]:
    best: Optional[tuple[int, float, JournalEvent]] = None
    for node, kinds in by_node.items():
        eval_results = kinds.get(EventKind.EVAL_RESULT, [])
        if not eval_results:
            continue
        if not _node_was_accepted(kinds):
            continue
        mean_primary = _mean_across_seeds(eval_results[-1], "primary")
        if mean_primary is None:
            continue
        if best is None or mean_primary > best[1]:
            best = (node, mean_primary, eval_results[-1])
    if best is None:
        return None
    return best[0], best[2]


def _convergence_consequence(n_accepted: int, n_cleared: int, epsilon: float) -> str:
    if n_accepted == 0:
        return "No candidate was accepted, so the epsilon question does not arise."
    if n_cleared == 0:
        return (
            f"{n_accepted} candidate(s) were accepted as statistically real improvements, "
            f"but none cleared epsilon={epsilon}. Under the organizers' N=3 no-improvement "
            "rule, this run would still be judged as stalled despite the real gain(s)."
        )
    if n_cleared == n_accepted:
        return f"All {n_accepted} accepted candidate(s) cleared epsilon={epsilon}; the N=3 counter would reset each time."
    return (
        f"{n_cleared} of {n_accepted} accepted candidate(s) cleared epsilon={epsilon}; "
        f"the other {n_accepted - n_cleared} reset nothing under the organizers' N=3 rule."
    )


def _write_results_md(
    events: list[JournalEvent],
    by_node: dict,
    out_dir: Path,
    training_wall_clock_seconds: Optional[float] = None,
) -> None:
    best = _select_best_node(by_node)
    if best is not None:
        _, best_result = best
        best_gauc = _mean_across_seeds(best_result, "GAUC")
        best_ndcg = _mean_across_seeds(best_result, "nDCG@5")
        best_primary = _mean_across_seeds(best_result, "primary")
    else:
        best_gauc = best_ndcg = best_primary = None

    final_iteration = events[-1].iteration if events else 0
    eval_results = [e for e in events if e.kind is EventKind.EVAL_RESULT]
    total_wall_seconds = sum(e.payload.get("wall_seconds", 0.0) for e in eval_results)
    # Computed from the journal, not a fixed string — every EVAL_RESULT
    # carries gpu_seconds/tokens now (see executor/journal.py's
    # log_eval_result), defaulting to 0 for a candidate that recorded
    # neither (true of every candidate today: no GPU, no LLM in this
    # pipeline). Summing rather than hardcoding means the day a real
    # LLM/GPU cost shows up here, this table reports it without anyone
    # having to remember to come back and un-hardcode it.
    total_gpu_seconds = sum(e.payload.get("gpu_seconds", 0.0) for e in eval_results)
    total_tokens = sum(e.payload.get("tokens", 0) for e in eval_results)
    intervention_count = sum(1 for e in events if e.kind is EventKind.INTERVENTION)

    def _fmt(value: Optional[float]) -> str:
        return f"{value:.4f}" if value is not None else "n/a"

    def _delta(value: Optional[float], baseline: float) -> str:
        return f"{value - baseline:+.4f}" if value is not None else "n/a"

    lines = [
        "# Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Validation-best GAUC | {_fmt(best_gauc)} |",
        f"| Validation-best nDCG@5 | {_fmt(best_ndcg)} |",
        f"| Validation-best primary | {_fmt(best_primary)} |",
        f"| Delta vs official baseline GAUC ({BASELINE_GAUC}) | {_delta(best_gauc, BASELINE_GAUC)} |",
        f"| Delta vs official baseline nDCG@5 ({BASELINE_NDCG5}) | {_delta(best_ndcg, BASELINE_NDCG5)} |",
        f"| Delta vs official baseline primary ({BASELINE_PRIMARY}) | {_delta(best_primary, BASELINE_PRIMARY)} |",
        f"| Iterations used | {final_iteration} / {ITERATION_CAP} |",
        f"| Total agent wall-clock (s) | {total_wall_seconds:.1f} |",
        f"| Total GPU-seconds | {total_gpu_seconds:.1f} |",
        f"| Total tokens | {total_tokens} |",
        f"| Manual interventions | {intervention_count} |",
    ]
    if training_wall_clock_seconds is not None:
        lines.append(f"| Total training wall-clock (s), measured | {training_wall_clock_seconds:.1f} |")
    lines.append("")
    lines.append(
        "_\"Total agent wall-clock\" above sums each EVAL_RESULT's own "
        "wall_seconds — it covers only the time this render's run spent "
        "re-evaluating, which is ~0s whenever candidates are rebuilt from "
        "cache rather than trained. Actual training wall-clock is "
        + (
            "reported in the row above, passed in separately by the caller."
            if training_wall_clock_seconds is not None
            else "not reported here — no measured total was passed to render()."
        )
        + "_"
    )

    convergence_events = [e for e in events if e.kind is EventKind.CONVERGENCE_CHECK]
    if convergence_events:
        # A DECISION naming the trivial "nothing to compare against yet"
        # case (see _is_baseline_adoption) accepted trivially — it
        # established the baseline, it didn't beat one. Counting it
        # alongside real gate acceptances overstates how many candidates
        # this run actually improved on something.
        decisions_by_node = {e.node: e for e in events if e.kind is EventKind.DECISION}
        baseline_nodes = {
            node
            for node, decision in decisions_by_node.items()
            if _is_baseline_adoption(decision.payload.get("reason"))
        }
        # "accept"/"clears_epsilon" are read off the matching DECISION at
        # the same node rather than off the CONVERGENCE_CHECK payload
        # itself: the durable helper's CONVERGENCE_CHECK carries both
        # directly (and they always agree with the co-located DECISION,
        # since both come from the same Verdict — see
        # executor/journal.py's log_convergence_check), but the
        # Controller's own CONVERGENCE_CHECK (controller/controller.py's
        # _check_convergence) has neither key at all — it reports run-
        # level convergence status, not a per-candidate verdict. Deriving
        # both from the DECISION works identically for either shape and
        # means this aggregate doesn't need its own shape detection.
        n_checks = len(convergence_events)
        n_baseline = sum(1 for e in convergence_events if e.node in baseline_nodes)
        n_accepted = 0
        n_cleared = 0
        epsilon = convergence_events[0].payload.get("epsilon")
        for check in convergence_events:
            if check.node in baseline_nodes:
                continue
            decision = decisions_by_node.get(check.node)
            if decision is None or not decision.payload.get("verdict"):
                continue
            n_accepted += 1
            delta = decision.payload.get("delta_primary")
            check_epsilon = check.payload.get("epsilon")
            if delta is not None and check_epsilon is not None and delta > check_epsilon:
                n_cleared += 1
        lines.append("")
        lines.append("## Convergence")
        lines.append("")
        lines.append(f"- Candidates decided: {n_checks}")
        if n_baseline:
            lines.append(f"- Baseline adopted (not an improvement): {n_baseline}")
        lines.append(f"- Accepted as improvements: {n_accepted}")
        lines.append(f"- Cleared epsilon={epsilon}: {n_cleared}")
        lines.append(f"- Consequence: {_convergence_consequence(n_accepted, n_cleared, epsilon)}")

    errors = [e for e in events if e.kind is EventKind.ERROR]
    if errors:
        recovered_nodes = _recovered_nodes(events)
        n_recovered = sum(1 for e in errors if e.node in recovered_nodes)
        # str(...) here isn't cosmetic: an error shape with no error_class
        # at all (e.g. the Controller's generator_exhausted) falls back to
        # `reason` via _error_class_and_excerpt, but without normalizing
        # first, a bare `e.payload.get("error_class")` would put a real
        # None into this set, and ", ".join() raises TypeError on that.
        classes = sorted({_error_class_and_excerpt(e.payload)[0] for e in errors})
        lines.append("")
        lines.append(
            f"{len(errors)} candidate(s) failed (error classes: {', '.join(classes)}); "
            f"the run continued past all {n_recovered} of them and reached FINALIZE."
            if n_recovered == len(errors)
            else f"{len(errors)} candidate(s) failed (error classes: {', '.join(classes)}); "
            f"{n_recovered} recovered, {len(errors) - n_recovered} did not."
        )

    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_trajectory_csv(by_node: dict, out_dir: Path) -> None:
    with open(out_dir / "trajectory.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteration", "node", "config_id", "val_primary", "accepted", "delta"])
        for node in sorted(by_node):
            kinds = by_node[node]
            eval_results = kinds.get(EventKind.EVAL_RESULT, [])
            if not eval_results:
                continue
            result = eval_results[-1]
            config_id = result.payload.get("config_id")
            val_primary = _mean_across_seeds(result, "primary")
            decisions = kinds.get(EventKind.DECISION, [])
            if decisions:
                accepted = bool(decisions[-1].payload.get("verdict"))
                delta = decisions[-1].payload.get("delta_primary")
                iteration = decisions[-1].iteration
            else:
                accepted = True  # baseline adoption, see _node_was_accepted
                delta = None
                iteration = result.iteration
            writer.writerow(
                [iteration, node, config_id, f"{val_primary:.6f}" if val_primary is not None else "", accepted, delta]
            )


def _write_forecast_calibration_md(by_node: dict, out_dir: Path) -> None:
    rows = []
    for node in sorted(by_node):
        kinds = by_node[node]
        hypotheses = kinds.get(EventKind.HYPOTHESIS, [])
        decisions = kinds.get(EventKind.DECISION, [])
        if not hypotheses or not decisions:
            continue
        hypothesis = hypotheses[0]
        decision = decisions[-1]
        expected_gain = hypothesis.payload.get("expected_gain")
        realized_delta = decision.payload.get("delta_primary")
        if expected_gain is None or realized_delta is None:
            continue
        rows.append(
            {
                "node": node,
                "target_slot": hypothesis.payload.get("target_slot"),
                "expected_gain": expected_gain,
                "realized_delta": realized_delta,
                "abs_error": abs(expected_gain - realized_delta),
            }
        )

    lines = [
        "# Forecast calibration",
        "",
        "Pairs each HYPOTHESIS's `expected_gain` (logged before evaluation)",
        "against the realized delta from the matching DECISION event, by node.",
        "",
        "| node | target_slot | expected_gain | realized_delta | abs_error |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['node']} | {row['target_slot']} | {row['expected_gain']:+.4f} "
            f"| {row['realized_delta']:+.4f} | {row['abs_error']:.4f} |"
        )
    if rows:
        mae = sum(row["abs_error"] for row in rows) / len(rows)
        lines.append("")
        lines.append(f"Mean absolute error: {mae:.4f}")
    else:
        lines.append("")
        lines.append("(no node has both a HYPOTHESIS and a DECISION event yet)")

    (out_dir / "forecast_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(
    journal_path: str,
    output_dir: str = "artifacts/report",
    training_wall_clock_seconds: Optional[float] = None,
) -> None:
    """Reads `journal_path` and writes iterations.md, results.md,
    trajectory.csv, and forecast_calibration.md into `output_dir`.

    `training_wall_clock_seconds` is optional and defaults to None: a
    caller that rebuilds every CandidateResult from cache (e.g.
    scripts/run_agent.py) has real EVAL_RESULT.wall_seconds of ~0 for
    every node, which is honest about THIS render but would otherwise
    make results.md read as if the whole trajectory cost nothing. Pass
    the real measured total here to have results.md report it
    separately instead of silently omitting it.
    """
    events = Journal.replay(journal_path)
    by_node = _group_by_node(events)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_iterations_md(events, by_node, out_dir)
    _write_results_md(events, by_node, out_dir, training_wall_clock_seconds=training_wall_clock_seconds)
    _write_trajectory_csv(by_node, out_dir)
    _write_forecast_calibration_md(by_node, out_dir)
