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


def _mean_across_seeds(eval_result: JournalEvent, key: str) -> Optional[float]:
    per_seed = eval_result.payload.get("per_seed") or {}
    if not per_seed:
        return None
    if key == "primary":
        values = [seed_data["primary"] for seed_data in per_seed.values()]
    else:
        values = [seed_data["values"][key] for seed_data in per_seed.values() if key in seed_data["values"]]
    if not values:
        return None
    return sum(values) / len(values)


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


def _format_eval_result(event: JournalEvent) -> list[str]:
    payload = event.payload
    lines = [f"**Result** (config_id=`{payload.get('config_id')}`)", ""]
    per_seed = payload.get("per_seed") or {}
    if per_seed:
        lines.append("| seed | GAUC | nDCG@5 | primary |")
        lines.append("|---|---|---|---|")
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
    lines.append("")
    lines.append(f"wall_seconds: {payload.get('wall_seconds')}")
    return lines


def _format_decision(event: JournalEvent) -> list[str]:
    payload = event.payload
    verdict_str = "ACCEPTED" if payload.get("verdict") else "REJECTED"
    ci95 = payload.get("ci95") or [None, None]
    lines = [
        f"**Verdict: {verdict_str}**",
        "",
        f"- delta: {payload.get('delta_primary'):+.4f} (n_seeds={payload.get('n_seeds')})",
        f"- ci95: [{ci95[0]:.4f}, {ci95[1]:.4f}]" if ci95[0] is not None else "- ci95: n/a",
        f"- backtest_delta: {payload.get('backtest_delta')}",
        f"- reason: {payload.get('reason')}",
    ]
    return lines


def _write_iterations_md(events: list[JournalEvent], by_node: dict, out_dir: Path) -> None:
    lines = ["# Iterations", ""]
    last_fragment_by_slot: dict[str, tuple[str, dict[str, Any]]] = {}

    for node in sorted(by_node):
        kinds = by_node[node]
        if not any(k in kinds for k in (EventKind.HYPOTHESIS, EventKind.EVAL_RESULT, EventKind.ERROR)):
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

        for error in kinds.get(EventKind.ERROR, []):
            lines.append(f"**Error**: `{error.payload.get('error_class')}` — {error.payload.get('excerpt')}")
            lines.append("")

        for recovery in kinds.get(EventKind.RECOVERY, []):
            lines.append(f"**Recovery**: {recovery.payload}")
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


def _write_results_md(events: list[JournalEvent], by_node: dict, out_dir: Path) -> None:
    best = _select_best_node(by_node)
    if best is not None:
        _, best_result = best
        best_gauc = _mean_across_seeds(best_result, "GAUC")
        best_ndcg = _mean_across_seeds(best_result, "nDCG@5")
        best_primary = _mean_across_seeds(best_result, "primary")
    else:
        best_gauc = best_ndcg = best_primary = None

    final_iteration = events[-1].iteration if events else 0
    total_wall_seconds = sum(
        e.payload.get("wall_seconds", 0.0) for e in events if e.kind is EventKind.EVAL_RESULT
    )
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
        "| Total tokens | 0 (no LLM in the loop yet) |",
        f"| Manual interventions | {intervention_count} |",
    ]
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


def render(journal_path: str, output_dir: str = "artifacts/report") -> None:
    """Reads `journal_path` and writes iterations.md, results.md,
    trajectory.csv, and forecast_calibration.md into `output_dir`.
    """
    events = Journal.replay(journal_path)
    by_node = _group_by_node(events)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_iterations_md(events, by_node, out_dir)
    _write_results_md(events, by_node, out_dir)
    _write_trajectory_csv(by_node, out_dir)
    _write_forecast_calibration_md(by_node, out_dir)
