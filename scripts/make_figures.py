"""Generates figures from existing run artifacts.

READ-ONLY: parses artifacts/journal_run.jsonl directly (no import of
executor/harness/controller needed, no training, no re-running
scripts/run_agent.py). Safe to run alongside another process that is
using the cache or writing its own artifacts.

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display server, just write the PNG

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_PATH = REPO_ROOT / "artifacts" / "journal_run.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "move_outcomes.png"

# The organizers' own convergence rule (epsilon=0.002) — see
# harness/gate.py's CONVERGENCE_EPSILON / controller/convergence.py's
# EPSILON. Not imported from either: this script is deliberately
# dependency-free (stdlib + matplotlib only), reading the journal as
# plain JSONL rather than through contracts.py's types.
CONVERGENCE_EPSILON = 0.002

ACCEPTED_COLOR = "#2ca02c"
REJECTED_COLOR = "#d62728"
BASELINE_COLOR = "#7f7f7f"


def _load_events(path: Path) -> list[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _is_baseline_adoption(reason: str) -> bool:
    """True for the trivial "nothing to compare against yet" decision —
    accepted by construction, not because it beat anything. Matches
    executor/report.py's own _is_baseline_adoption: both the durable
    helper's and the Controller's wording for this case share "compare
    against" (see that function's docstring for why "initial incumbent"
    alone isn't a safe-enough substring).
    """
    return "compare against" in (reason or "").lower()


def _move_label(node: int, hypothesis_by_node: dict, config_id_by_node: dict) -> str:
    """Pulls a readable label for `node` from its HYPOTHESIS event
    (target_slot + citation key — the two most identifying fields a
    HypothesisPayload carries; there is no single "move name" field),
    falling back to the node's config_id when no HYPOTHESIS was logged
    for it.
    """
    hyp = hypothesis_by_node.get(node)
    if hyp is not None:
        target_slot = hyp.get("target_slot") or "?"
        citation_key = (hyp.get("citation") or {}).get("key") or "?"
        return f"{target_slot} ({citation_key})"
    return config_id_by_node.get(node, f"node {node}")


def main() -> None:
    events = _load_events(JOURNAL_PATH)

    hypothesis_by_node: dict[int, dict] = {}
    config_id_by_node: dict[int, str] = {}
    for event in events:
        if event["kind"] == "hypothesis":
            hypothesis_by_node.setdefault(event["node"], event["payload"])
        elif event["kind"] == "eval_result":
            config_id = event["payload"].get("config_id")
            if config_id:
                config_id_by_node.setdefault(event["node"], config_id)

    decisions = [event for event in events if event["kind"] == "decision"]
    if not decisions:
        raise SystemExit(f"no DECISION events found in {JOURNAL_PATH}")

    rows = []
    for decision in decisions:
        payload = decision["payload"]
        delta = payload.get("delta_primary")
        if delta is None:
            continue
        reason = payload.get("reason") or ""
        if _is_baseline_adoption(reason):
            category = "baseline"
        elif payload.get("verdict"):
            category = "accepted"
        else:
            category = "rejected"
        rows.append(
            {
                "node": decision["node"],
                "delta": delta,
                "category": category,
                "label": _move_label(decision["node"], hypothesis_by_node, config_id_by_node),
            }
        )
    rows.sort(key=lambda r: r["node"])

    color_by_category = {
        "accepted": ACCEPTED_COLOR,
        "rejected": REJECTED_COLOR,
        "baseline": BASELINE_COLOR,
    }
    labels = [row["label"] for row in rows]
    deltas = [row["delta"] for row in rows]
    colors = [color_by_category[row["category"]] for row in rows]

    # 1200x800 px at dpi=100 -> figsize in inches x dpi = pixels.
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    y_pos = range(len(rows))
    ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # node 1 at the top, reading top-to-bottom like the run itself

    # Padding keeps both reference lines and every bar clearly inside the
    # frame — the accepted candidate's whole point is being visibly
    # between the two, so neither line can sit flush against an edge.
    min_delta = min(deltas + [0.0])
    max_delta = max(deltas + [CONVERGENCE_EPSILON])
    span = max_delta - min_delta
    ax.set_xlim(min_delta - 0.10 * span, max_delta + 0.14 * span)

    # A number on every bar removes any ambiguity about exactly where the
    # accepted candidate falls relative to 0 and epsilon — the reader
    # doesn't have to eyeball position against the axis ticks for the one
    # finding this figure exists to make legible.
    label_offset = 0.006 * span
    for y, delta in zip(y_pos, deltas):
        ha = "left" if delta >= 0 else "right"
        ax.text(
            delta + (label_offset if delta >= 0 else -label_offset),
            y,
            f"{delta:+.4f}",
            ha=ha,
            va="center",
            fontsize=9,
            zorder=4,
        )

    ax.axvline(0.0, color="black", linewidth=1.2, zorder=2)
    ax.axvline(CONVERGENCE_EPSILON, color="dimgray", linewidth=1.2, linestyle="--", zorder=2)
    # Rotated and placed just right of the threshold line, inside the
    # axes near the top — reads alongside the line it labels instead of
    # competing with the title for the same horizontal band above the
    # plot.
    ax.text(
        CONVERGENCE_EPSILON,
        0.98,
        " organizers' convergence threshold (epsilon = 0.002)",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        rotation=90,
        fontsize=9,
        color="dimgray",
    )

    ax.set_xlabel("delta_primary (validation, vs. incumbent at time of decision)")
    ax.set_title("Move outcomes vs. the organizers' convergence threshold", pad=14)
    ax.legend(
        handles=[
            Patch(facecolor=ACCEPTED_COLOR, edgecolor="black", label="Accepted"),
            Patch(facecolor=REJECTED_COLOR, edgecolor="black", label="Rejected"),
            Patch(facecolor=BASELINE_COLOR, edgecolor="black", label="Baseline (adopted, not compared)"),
        ],
        loc="upper left",
        fontsize=9,
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=100)
    plt.close(fig)

    size_bytes = OUTPUT_PATH.stat().st_size
    print(f"wrote {OUTPUT_PATH} ({size_bytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
