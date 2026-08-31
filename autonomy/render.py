"""Renders the autonomy section: the run's integrity evidence, read back
out of the journal it was written into.

WHY THIS IS ITS OWN RENDERER RATHER THAN A PATCH TO executor/report.py.
Two reasons, one practical and one about what the artifact is for.

The practical one: executor/report.py is W3's file and is being edited.
Threading the autonomy section through it would couple this evidence to
someone else's in-flight work, and the section would be unproducible for
as long as that work is unfinished. This module replays the journal
itself, so autonomy.md renders regardless.

The one that matters more: the metric report answers "what did the agent
achieve", and this answers "did the agent do it by itself". Those are
different claims, defended by different evidence, and a reviewer
attacking one is not attacking the other. Keeping them in separate
documents means neither can quietly borrow the other's credibility.

READS ONLY WHAT PR2 WROTE. autonomy/integrity.py puts a launch
fingerprint in the RUN_START payload's run_metadata, an end-of-run
summary in RUN_END's, and emits EventKind.INTERVENTION for genuine manual
touches. This module renders exactly those three things and invents
nothing.

DEGRADES, NEVER CRASHES. A journal written before that mechanism existed
— scripts/run_agent.py's, or any run predating PR2 — has none of those
keys. Rendering it must say so plainly rather than raising a KeyError on
a report generator, so every lookup here is a `.get` chain. "Integrity
evidence absent" is itself a finding worth printing: it is the honest
description of a run whose autonomy cannot be checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from contracts import EventKind, JournalEvent
from executor.journal import Journal

__all__ = ["ABSENT_NOTE", "POLICY_PATH", "render_autonomy", "select_run"]

POLICY_PATH = "autonomy/INTERVENTION_POLICY.md"

ABSENT_NOTE = "integrity evidence absent for this run"
"""Rendered when a run carries no launch fingerprint.

Deliberately a statement rather than a blank section. A reader who sees
nothing assumes the tool failed; a reader who sees this knows the run
itself carried no evidence, which is a different and more useful fact.
"""

_TREE_STATE = {True: "DIRTY", False: "clean", None: "unknown"}


def _integrity(event: Optional[JournalEvent]) -> dict[str, Any]:
    """The integrity block off one event's payload, or {}.

    Four levels of `.get` because every one of them is genuinely optional:
    the event may not exist, its payload may predate run_metadata, the
    metadata may carry something other than integrity, and the block
    itself may be partial.
    """
    if event is None:
        return {}
    payload = event.payload or {}
    metadata = payload.get("run_metadata") or {}
    return metadata.get("integrity") or {}


def select_run(
    events: Sequence[JournalEvent], run_id: Optional[str] = None
) -> tuple[Optional[str], list[JournalEvent]]:
    """Pick one run's events out of a journal that may hold several.

    Returns (run_id, that run's events). With no `run_id` the LAST run in
    the file wins — a journal accumulates runs by appending, so the most
    recent is the one a launcher just wrote and the one a reader almost
    always means.

    Runs are never merged. A journal holding an interrupted run and the
    manual restart that followed it holds two different stories about two
    different processes, and averaging them would describe neither.
    """
    if not events:
        return None, []
    if run_id is None:
        run_id = events[-1].run_id
    return run_id, [event for event in events if event.run_id == run_id]


def _first(events: Sequence[JournalEvent], kind: EventKind) -> Optional[JournalEvent]:
    for event in events:
        if event.kind is kind:
            return event
    return None


def _kv_table(rows: Sequence[tuple[str, str]]) -> list[str]:
    lines = ["| | |", "|---|---|"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return lines


def _short(value: Any, n: int = 12) -> str:
    text = str(value or "")
    return text[:n] if text else "—"


def _render_launch(launch: Mapping[str, Any]) -> list[str]:
    dirty = launch.get("dirty")
    tree = _TREE_STATE.get(dirty, "unknown")
    if dirty:
        tree += f" ({len(launch.get('dirty_files') or [])} uncommitted file(s))"

    dirs = ", ".join(launch.get("source_dirs") or []) or "—"
    rows = [
        ("Commit", f"`{launch.get('commit', '—')}`"),
        ("Working tree at launch", f"**{tree}**"),
        ("Source fingerprint", f"`{_short(launch.get('code_hash'), 16)}…`"),
        ("Fingerprint covers", dirs),
        ("Started", launch.get("started_ts", "—")),
        ("Python / platform", f"{launch.get('python_version', '—')} on {launch.get('platform', '—')}"),
    ]
    lines = ["## Launch provenance", ""]
    lines.extend(_kv_table(rows))

    dirty_files = launch.get("dirty_files") or []
    if dirty_files:
        lines.append("")
        lines.append("Uncommitted at launch:")
        lines.append("")
        lines.extend(f"- `{entry}`" for entry in dirty_files)
    lines.append("")
    return lines


def _render_during(
    summary: Mapping[str, Any], relaunch: Mapping[str, Any], finished: bool
) -> list[str]:
    stable = summary.get("code_fingerprint_stable")
    rows = [
        ("Fingerprint checks", str(summary.get("checks_performed", "—"))),
        (
            "Candidates checked",
            f"{summary.get('candidates_checked', '—')} "
            f"(floor for verified: {summary.get('min_candidates_for_verified', '—')})",
        ),
        (
            "Fingerprint stable",
            {True: "**yes**", False: "**NO — source changed mid-run**"}.get(stable, "unknown"),
        ),
        ("Final fingerprint", f"`{_short(summary.get('final_code_hash'), 16)}…`"),
        (
            "Run finished normally",
            "**yes** (reached RUN_END)"
            if finished
            else "**NO — no RUN_END; the process did not finish through its normal path**",
        ),
    ]
    if relaunch:
        counted = " — counted as an intervention" if relaunch.get("counts_as_intervention") else ""
        rows.append(("Why this process started", f"`{relaunch.get('kind', '—')}`{counted}"))

    lines = ["## Integrity during the run", ""]
    lines.extend(_kv_table(rows))
    if relaunch.get("reason"):
        lines.append("")
        lines.append(f"> {relaunch['reason']}")
    lines.append("")
    return lines


def _render_interventions(
    interventions: Sequence[JournalEvent], summary: Mapping[str, Any]
) -> list[str]:
    lines = ["## Interventions", ""]

    if not interventions:
        lines.append("**0 — none recorded.**")
        lines.append("")
        lines.append(
            "No code edit and no manual restart was detected while this run "
            f"was in flight. See [{POLICY_PATH}]({POLICY_PATH}) for exactly "
            "what that claim does and does not cover."
        )
        lines.append("")
        return lines

    lines.append(f"**{len(interventions)} recorded.**")
    lines.append("")
    lines.append("| iteration | who | type | reason |")
    lines.append("|---|---|---|---|")
    for event in interventions:
        payload = event.payload or {}
        lines.append(
            f"| {payload.get('iteration_affected', event.iteration)} "
            f"| `{payload.get('who', '—')}` "
            f"| `{payload.get('type', '—')}` "
            f"| {payload.get('reason', '—')} |"
        )
    lines.append("")

    # The monitor's own tally and the journal's event count are supposed
    # to be the same number by construction (scripts/run_controller.py
    # cross-checks them before exiting). Saying so here means a reader who
    # only ever sees this file can still tell that they agreed.
    counted = summary.get("manual_interventions")
    if counted is not None and counted != len(interventions):
        lines.append(
            f"> **Accounting mismatch:** the run's own summary counted "
            f"{counted}, but {len(interventions)} INTERVENTION event(s) are "
            "in the journal. Do not publish either number until this is "
            "explained."
        )
        lines.append("")
    return lines


def _render_verdict(summary: Mapping[str, Any], one_line: str) -> list[str]:
    verified = summary.get("verified")
    badge = "**VERIFIED AUTONOMOUS — YES**" if verified else "**VERIFIED AUTONOMOUS — NO**"

    lines = [badge, "", f"> {one_line}", ""]

    reasons = summary.get("unverified_because") or []
    if not verified:
        lines.append("Not verified because:")
        lines.append("")
        if reasons:
            lines.extend(f"- {reason}" for reason in reasons)
        else:
            # An older summary (PR2's shape) has no unverified_because.
            lines.append(
                "- (this run's summary predates the itemised reasons; check "
                "the tree state, fingerprint stability and intervention "
                "count above)"
            )
        lines.append("")
    return lines


def _one_line_from(summary: Mapping[str, Any]) -> str:
    """Rebuild IntegrityMonitor.one_line from a replayed summary.

    Rebuilt rather than stored because the summary is the durable record
    and the sentence is a view of it; keeping a second copy on disk would
    create something that can disagree with the numbers beside it.
    """
    # `tree_clean_at_launch` is already the positive sense (True means
    # clean), unlike the raw git `dirty` flag it was derived from. It
    # collapses "dirty" and "unknown" into False, so this cannot
    # distinguish them — the launch table above can, and does.
    tree = "clean" if summary.get("tree_clean_at_launch") else "not clean"
    stable = "stable" if summary.get("code_fingerprint_stable") else "CHANGED"
    return (
        f"commit={_short(summary.get('commit'))}, tree at launch={tree}, "
        f"code fingerprint {stable} across "
        f"{summary.get('checks_performed', '?')} check(s) covering "
        f"{summary.get('candidates_checked', '?')} candidate(s), "
        f"manual interventions={summary.get('manual_interventions', '?')}"
    )


def render_autonomy(
    journal_path: str,
    out_dir: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
) -> str:
    """Render one run's autonomy evidence as markdown.

    Writes `<out_dir>/autonomy.md` when `out_dir` is given, and always
    returns the markdown. Returning it as well as writing it is what lets
    the tests assert on content without a filesystem round trip, and lets
    a caller print it.

    `run_id` selects which run in the journal to render; the default is
    the most recent. See `select_run` for why runs are never merged.
    """
    events = Journal.replay(journal_path)
    selected_id, run_events = select_run(events, run_id)

    other_runs = sorted({e.run_id for e in events} - {selected_id}) if selected_id else []

    start = _first(run_events, EventKind.RUN_START)
    end = _first(run_events, EventKind.RUN_END)
    launch_block = _integrity(start)
    end_block = _integrity(end)

    launch = launch_block.get("launch") or {}
    relaunch = launch_block.get("relaunch") or {}
    # The end-of-run summary is authoritative; RUN_START's is a snapshot
    # taken before any work happened and would understate every count.
    summary = end_block.get("summary") or {}

    interventions = [e for e in run_events if e.kind is EventKind.INTERVENTION]

    lines: list[str] = ["# Autonomy", ""]

    if selected_id is None:
        lines.append(f"_{ABSENT_NOTE}: the journal `{journal_path}` is empty._")
        lines.append("")
        return _finish(lines, out_dir)

    lines.append(f"_Run `{selected_id}` — journal `{journal_path}`._")
    if other_runs:
        lines.append("")
        lines.append(
            f"_This journal also holds {len(other_runs)} other run(s) "
            f"({', '.join(f'`{r}`' for r in other_runs)}), not shown here._"
        )
    lines.append("")

    if not launch and not summary:
        lines.append(f"**{ABSENT_NOTE}.**")
        lines.append("")
        lines.append(
            "This run recorded no launch fingerprint and no end-of-run "
            "integrity summary, so there is nothing to verify against. That "
            "is expected for a journal written before "
            "`autonomy/integrity.py` existed, or by a runner that does not "
            "use it (`scripts/run_agent.py`, for instance). A run with no "
            "evidence is not the same as a run with clean evidence, and this "
            "document will not present it as one."
        )
        lines.append("")
        if interventions:
            lines.extend(_render_interventions(interventions, {}))
        lines.extend(_render_footer())
        return _finish(lines, out_dir)

    lines.extend(_render_verdict(summary, _one_line_from(summary)))
    lines.extend(_render_launch(launch))
    lines.extend(_render_during(summary, relaunch, finished=end is not None))
    lines.extend(_render_interventions(interventions, summary))
    lines.extend(_render_footer())
    return _finish(lines, out_dir)


def _render_footer() -> list[str]:
    return [
        "## What this claim rests on",
        "",
        "The definition of an intervention — what counts, what pointedly "
        "does not, and the known limits of the mechanism — is written down "
        f"in [{POLICY_PATH}]({POLICY_PATH}). Read it before trusting the "
        "number above.",
        "",
        "The evidence here is replayed from the run's own append-only "
        "journal: the launch fingerprint from its `RUN_START`, the summary "
        "from its `RUN_END`, and one row per `INTERVENTION` event. Nothing "
        "in this document is computed after the fact.",
        "",
    ]


def _finish(lines: list[str], out_dir: Optional[str]) -> str:
    markdown = "\n".join(lines).rstrip() + "\n"
    if out_dir is not None:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "autonomy.md").write_text(markdown, encoding="utf-8")
    return markdown
