"""Run-integrity evidence: what makes an intervention count of zero mean
something.

THE PROBLEM THIS SOLVES. `executor/report.py` renders a "Manual
interventions" row by counting EventKind.INTERVENTION events. Nothing in
the agent loop has ever emitted one, so that row reads 0 on every journal
the repo can produce — and it would read 0 just the same after fifty
manual restarts and a dozen mid-run edits. A number that cannot be
anything but zero is not evidence of autonomy; it is the absence of
evidence wearing evidence's clothes.

A zero becomes credible only when the run POSITIVELY records what it
checked. That is this module: a launch fingerprint pinning the commit and
the working tree, a content hash over the source that actually runs, and
a monitor that re-checks that hash as the run proceeds and counts the
checks. "Checked 7 times across 41 minutes, stable, tree clean at launch,
commit 62a61f8" is a claim a reviewer can attack. "0" is not.

See autonomy/INTERVENTION_POLICY.md for what does and does not count as
an intervention, and why the distinction is what keeps the count
trustworthy.


NO NEW EventKind, DELIBERATELY. contracts.EventKind is a closed enum
enforced at decode time (`JournalEvent.from_jsonl` raises
JournalDecodeError on an unknown kind), so adding a member is a
contracts.py change and a cross-team event. It is also unnecessary:
`JournalEvent.payload` is an unconstrained `dict[str, Any]` whose shapes
contracts.py describes as "documented, not enforced". Launch metadata
therefore rides in the RUN_START payload, end-of-run evidence rides in
RUN_END, and real interventions use the INTERVENTION kind that already
exists — which has the side benefit that executor/report.py's existing
counter sees exactly the events we intend it to see, no more.


KNOWN LIMITATION, STATED UP FRONT: `code_fingerprint` covers .py source
only. It does not cover the DATASET. A changed CSV under data/ would
invalidate a run's results as surely as a changed line of code, and
nothing here would notice — harness.cache is keyed on config_id, which
hashes slot configuration and knows nothing about data content. Data
fingerprinting is deliberately out of scope for this change; it is a
different question (what was the run computed FROM) from the one this
module answers (was the run's CODE stable while it ran).
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "DEFAULT_SOURCE_DIRS",
    "UNKNOWN",
    "CheckedExecutor",
    "IntegrityMetadata",
    "IntegrityMonitor",
    "RelaunchClassification",
    "classify_relaunch",
    "code_fingerprint",
    "git_state",
    "launch_fingerprint",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE_DIRS: tuple[str, ...] = (
    "executor",
    "harness",
    "controller",
    "methods",
    "autonomy",
)
"""The source a run's behaviour actually depends on.

`tests/` is excluded on purpose: editing a test mid-run changes nothing
about what the run computes, and counting it would produce interventions
that are not interventions — exactly the false positives that would make
the count untrustworthy in the other direction. `scripts/` is excluded
for a subtler reason: the launcher is already running, so a later edit to
it cannot affect the process in flight.
"""

UNKNOWN = "unknown"
"""Recorded when git cannot answer. Never raises, never guesses.

An honest "unknown" is worth more than a fabricated hash: it tells a
reviewer the provenance claim is weaker for this run, rather than
offering a value that looks authoritative and is not.
"""

_GIT_TIMEOUT_S = 10.0


def _run_git(args: Sequence[str], *, repo_root: Path) -> Optional[str]:
    """`git *args` in `repo_root`, or None if git cannot answer.

    Swallows every failure mode on purpose — git missing from PATH, the
    directory not being a repository, a timeout, a non-zero exit. This
    function is called at the very start of an unattended run whose whole
    point is to finish without a human; dying because provenance could
    not be established would be a self-inflicted wound. The caller
    records UNKNOWN and carries on.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def git_state(repo_root: Optional[Path] = None) -> dict[str, Any]:
    """{commit, dirty, dirty_files} for `repo_root`.

    `dirty` is None — not False — when git could not answer. The three
    states are genuinely different: a clean tree, a dirty tree, and no
    idea. Collapsing "no idea" into "clean" would manufacture exactly the
    reassurance this module exists to make earnable.

    `dirty_files` holds the porcelain status lines, so a dirty launch
    says WHICH files were uncommitted rather than only that some were.
    """
    root = REPO_ROOT if repo_root is None else Path(repo_root)

    commit_out = _run_git(["rev-parse", "HEAD"], repo_root=root)
    status_out = _run_git(["status", "--porcelain"], repo_root=root)

    if status_out is None:
        dirty: Optional[bool] = None
        dirty_files: list[str] = []
    else:
        dirty_files = [line for line in status_out.splitlines() if line.strip()]
        dirty = bool(dirty_files)

    return {
        "commit": commit_out.strip() if commit_out else UNKNOWN,
        "dirty": dirty,
        "dirty_files": dirty_files,
    }


def _source_files(dirs: Iterable[str], *, repo_root: Path) -> list[Path]:
    """Every .py file under `dirs`, sorted, __pycache__ excluded.

    Sorted by POSIX-style relative path rather than by OS path, so the
    same tree fingerprints identically on Windows and Linux — a hash that
    changed with the separator would report an intervention every time
    the run moved machines.
    """
    files: list[Path] = []
    for name in dirs:
        directory = repo_root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(repo_root).as_posix())


def code_fingerprint(
    dirs: Iterable[str] = DEFAULT_SOURCE_DIRS,
    *,
    repo_root: Optional[Path] = None,
) -> str:
    """sha256 over the source under `dirs`. 64 hex chars.

    Hashes each file's RELATIVE PATH as well as its bytes, so renaming a
    file, or adding one, changes the fingerprint just as editing one
    does. Reads bytes rather than decoded text deliberately: two files
    are the same file or they are not, and a decode step would make the
    answer depend on the platform's default encoding — the same trap that
    makes tests/test_invariants.py fail under cp932 on this machine.

    Catches an uncommitted edit, which is the case that matters most: a
    mid-run change that never reaches git is invisible to `git status` at
    launch and invisible to the commit hash forever.

    ~5ms over the repo's five source directories, which is why the
    monitor can afford to re-check at every node boundary rather than
    only at the ends.
    """
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    hasher = hashlib.sha256()
    for path in _source_files(dirs, repo_root=root):
        hasher.update(path.relative_to(root).as_posix().encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def launch_fingerprint(
    dirs: Iterable[str] = DEFAULT_SOURCE_DIRS,
    *,
    repo_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Everything knowable about the state this run started from.

    Keys: commit, dirty, dirty_files, code_hash, source_dirs,
    python_version, platform, started_ts.

    `commit` plus `dirty` is the provenance claim — "this run started
    from a named commit with nothing uncommitted" — and `code_hash` is
    what makes it checkable later, since a commit hash says nothing about
    edits made after launch. The interpreter and platform rows are there
    because a result that reproduces on one and not the other is a real
    finding, and reconstructing which was used from an artifact is
    otherwise guesswork.
    """
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    state = git_state(root)
    return {
        "commit": state["commit"],
        "dirty": state["dirty"],
        "dirty_files": state["dirty_files"],
        "code_hash": code_fingerprint(dirs, repo_root=root),
        "source_dirs": list(dirs),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "started_ts": datetime.now(timezone.utc).isoformat(),
    }


class RelaunchClassification:
    """Why this process is starting, and whether that counts against the
    intervention tally.

    Four outcomes, and the distinctions are the whole point:

      fresh              No prior journal, or the prior run ended through
                         RUN_END. Nothing to explain.
      autonomous_resume  A prior run was interrupted, --resume was given,
                         and the code is byte-identical to what that run
                         launched with. A machine restarting itself over
                         unchanged code is the autonomy story working, not
                         a human intervening in it. NOT counted.
      manual_restart     A prior run was interrupted and either --resume
                         was absent (a human decided to start over) or the
                         code has changed since (whatever is running now is
                         not what was running then). COUNTED.
      unknown_prior      A prior journal exists but carries no launch
                         fingerprint to compare against — a run from before
                         this mechanism existed. Counted as a manual
                         restart, because an unverifiable resume is not a
                         verified one.

    `counts_as_intervention` is the single field the tally reads, so the
    policy lives in one place rather than being re-derived at each call
    site.
    """

    def __init__(
        self,
        kind: str,
        *,
        counts_as_intervention: bool,
        reason: str,
        prior_run_id: Optional[str] = None,
        prior_code_hash: Optional[str] = None,
        prior_last_kind: Optional[str] = None,
    ) -> None:
        self.kind = kind
        self.counts_as_intervention = counts_as_intervention
        self.reason = reason
        self.prior_run_id = prior_run_id
        self.prior_code_hash = prior_code_hash
        self.prior_last_kind = prior_last_kind

    def as_payload(self) -> dict[str, Any]:
        """The shape that goes into RUN_START metadata."""
        return {
            "kind": self.kind,
            "counts_as_intervention": self.counts_as_intervention,
            "reason": self.reason,
            "prior_run_id": self.prior_run_id,
            "prior_code_hash": self.prior_code_hash,
            "prior_last_kind": self.prior_last_kind,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RelaunchClassification({self.kind!r}, counts={self.counts_as_intervention})"


def classify_relaunch(
    prior_events: Sequence[Any],
    *,
    code_hash: str,
    resume_requested: bool,
) -> RelaunchClassification:
    """Decide why this process is starting, from the journal already on disk.

    `prior_events` is whatever `Journal.replay(path)` returned — empty for
    a brand-new journal. Takes events rather than a path so the decision
    is a pure function of what was recorded, and can be tested without a
    filesystem.

    NOT A RESUME OF THE CONTROLLER'S STATE. Nothing here reconstructs an
    incumbent, a history or a stage; `Controller.run()` always starts at
    Stage.INIT and this change does not alter that. "Autonomous resume"
    means only "this process restarted itself over unchanged code, and no
    human touched anything" — which is precisely the claim the
    intervention count is about. Calling it a resume of the RUN would
    overclaim.
    """
    if not prior_events:
        return RelaunchClassification(
            "fresh",
            counts_as_intervention=False,
            reason="no prior journal at this path; nothing to resume or explain",
        )

    last = prior_events[-1]
    last_kind = getattr(last.kind, "value", str(last.kind))

    if last_kind == "run_end":
        return RelaunchClassification(
            "fresh",
            counts_as_intervention=False,
            reason=(
                "the prior run in this journal reached RUN_END, so it finished "
                "through its normal path; this is a new run, not a restart"
            ),
            prior_run_id=getattr(last, "run_id", None),
            prior_last_kind=last_kind,
        )

    # An interrupted prior run. Find what IT launched with.
    # The path is payload["run_metadata"]["integrity"]["launch"]["code_hash"]
    # — Controller._with_run_metadata nests caller context under
    # "run_metadata" so it cannot shadow a field the Controller owns, and
    # scripts/run_controller.py files this module's records under
    # "integrity" within that. Every step is a .get, because a journal
    # written before this mechanism existed has none of them.
    prior_run_id = getattr(last, "run_id", None)
    prior_code_hash: Optional[str] = None
    for event in reversed(prior_events):
        if getattr(event.kind, "value", str(event.kind)) != "run_start":
            continue
        if getattr(event, "run_id", None) != prior_run_id:
            continue
        metadata = (event.payload or {}).get("run_metadata") or {}
        launch = (metadata.get("integrity") or {}).get("launch") or {}
        prior_code_hash = launch.get("code_hash")
        break

    if prior_code_hash is None:
        return RelaunchClassification(
            "unknown_prior",
            counts_as_intervention=True,
            reason=(
                f"the prior run {prior_run_id!r} was interrupted (last event "
                f"{last_kind!r}) and recorded no launch fingerprint, so this "
                "relaunch cannot be verified as autonomous; counted as manual"
            ),
            prior_run_id=prior_run_id,
            prior_last_kind=last_kind,
        )

    if not resume_requested:
        return RelaunchClassification(
            "manual_restart",
            counts_as_intervention=True,
            reason=(
                f"the prior run {prior_run_id!r} was interrupted (last event "
                f"{last_kind!r}) and this process was launched without "
                "--resume, so a human chose to start it again"
            ),
            prior_run_id=prior_run_id,
            prior_code_hash=prior_code_hash,
            prior_last_kind=last_kind,
        )

    if prior_code_hash != code_hash:
        return RelaunchClassification(
            "manual_restart",
            counts_as_intervention=True,
            reason=(
                f"--resume was requested after the prior run {prior_run_id!r} "
                f"was interrupted, but the source changed since it launched "
                f"({prior_code_hash[:12]} -> {code_hash[:12]}); whatever is "
                "running now is not what was running then"
            ),
            prior_run_id=prior_run_id,
            prior_code_hash=prior_code_hash,
            prior_last_kind=last_kind,
        )

    return RelaunchClassification(
        "autonomous_resume",
        counts_as_intervention=False,
        reason=(
            f"the prior run {prior_run_id!r} was interrupted (last event "
            f"{last_kind!r}) and relaunched with --resume over byte-identical "
            f"source ({code_hash[:12]}); no human touched the code"
        ),
        prior_run_id=prior_run_id,
        prior_code_hash=prior_code_hash,
        prior_last_kind=last_kind,
    )


class IntegrityMonitor:
    """Holds the launch fingerprint and re-checks it as the run proceeds.

    THE COUNTER OF CHECKS IS THE EVIDENCE. A run that says "stable" having
    looked once at the start and once at the end is making a much weaker
    claim than one that looked at every node boundary across four hours,
    and a reader cannot tell those apart unless the number is recorded. So
    `checks_performed` is part of the summary, not an implementation
    detail.

    Interventions are reported through `on_intervention`, a callable the
    launcher supplies (in practice `Journal.log_intervention`), rather
    than written here. This class decides WHAT counts; the journal decides
    how it is recorded, and keeping those apart is what lets the whole
    classification be unit-tested with no filesystem at all.
    """

    def __init__(
        self,
        *,
        launch: Mapping[str, Any],
        on_intervention: Any,
        dirs: Iterable[str] = DEFAULT_SOURCE_DIRS,
        repo_root: Optional[Path] = None,
        who: str = "scripts/run_controller.py",
    ) -> None:
        self.launch = dict(launch)
        self._on_intervention = on_intervention
        self._dirs = tuple(dirs)
        self._repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
        self._who = who

        self.checks_performed = 0
        self.interventions: list[dict[str, Any]] = []
        self.drifted = False
        self.current_code_hash: str = str(self.launch.get("code_hash", UNKNOWN))

    # -- recording ------------------------------------------------------

    def record_intervention(self, kind: str, reason: str) -> None:
        """Log one real intervention and count it.

        Every call here becomes an EventKind.INTERVENTION event, which is
        what executor/report.py counts. That is the invariant the policy
        document exists to protect: this method is only ever called for a
        genuine manual touch, so the report's number and ours cannot
        disagree.
        """
        self.interventions.append({"type": kind, "reason": reason})
        self._on_intervention(self._who, kind, reason)

    def record_relaunch(self, classification: RelaunchClassification) -> None:
        """Act on a relaunch classification.

        Only the counted kinds become INTERVENTION events. An autonomous
        resume is recorded in RUN_START metadata by the caller and
        deliberately emits nothing here — logging it would inflate the
        very count it is supposed to leave alone.
        """
        if classification.counts_as_intervention:
            self.record_intervention(classification.kind, classification.reason)

    # -- checking -------------------------------------------------------

    def check(self, label: str = "") -> bool:
        """Re-hash the source and compare against launch. True if stable.

        A drift is reported ONCE. A mid-run edit that is not reverted
        would otherwise fire at every subsequent check and report one
        human action as a dozen interventions — inflating the count is as
        dishonest as suppressing it. After the first report the new hash
        becomes the reference, so a SECOND, distinct edit is still caught.
        """
        self.checks_performed += 1
        observed = code_fingerprint(self._dirs, repo_root=self._repo_root)
        if observed == self.current_code_hash:
            return True

        previous = self.current_code_hash
        self.current_code_hash = observed
        self.drifted = True
        self.record_intervention(
            "code_changed_midrun",
            (
                f"source fingerprint changed during the run"
                f"{f' at {label}' if label else ''}: "
                f"{previous[:12]} -> {observed[:12]} over {list(self._dirs)}; "
                "the code that produced later nodes is not the code that "
                "produced earlier ones"
            ),
        )
        return False

    # -- evidence -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """The positive record: what was checked, and what was found.

        Written into the RUN_END payload so it is inside the run's own
        terminal event rather than appended after it — nothing may follow
        RUN_END, or a replay can no longer tell a finished run from a
        killed one (contracts.EventKind.RUN_END's own docstring).

        `verified` is the one-line claim, and it is deliberately
        conservative: it is only true when the tree was known-clean at
        launch, the fingerprint never moved, and no intervention was
        recorded. A run launched from a dirty tree still produces a
        complete record — it simply does not get to claim this.
        """
        clean_launch = self.launch.get("dirty") is False
        return {
            "commit": self.launch.get("commit", UNKNOWN),
            "tree_clean_at_launch": clean_launch,
            "dirty_files_at_launch": list(self.launch.get("dirty_files") or []),
            "launch_code_hash": self.launch.get("code_hash", UNKNOWN),
            "final_code_hash": self.current_code_hash,
            "code_fingerprint_stable": not self.drifted,
            "checks_performed": self.checks_performed,
            "source_dirs": list(self._dirs),
            "manual_interventions": len(self.interventions),
            "intervention_types": [entry["type"] for entry in self.interventions],
            "verified": bool(
                clean_launch and not self.drifted and not self.interventions
            ),
        }

    def check_at_node_boundary(self, config_id: str) -> None:
        """Convenience label for CheckedExecutor's call site."""
        self.check(label=f"node boundary before {config_id}")

    def one_line(self) -> str:
        """The summary as a sentence, for the console and for a reader who
        wants the claim without the dict."""
        s = self.summary()
        # `dirty` is the git sense: True means there WERE uncommitted
        # changes. Reading it as "clean" is the exact inversion that would
        # let a dirty launch describe itself as a clean one.
        tree = {True: "DIRTY", False: "clean", None: "unknown"}[
            self.launch.get("dirty")
        ]
        return (
            f"commit={str(s['commit'])[:12]}, tree at launch={tree}, "
            f"code fingerprint {'stable' if s['code_fingerprint_stable'] else 'CHANGED'} "
            f"across {s['checks_performed']} check(s), "
            f"manual interventions={s['manual_interventions']}"
        )


class CheckedExecutor:
    """An ExecutorPort that re-verifies the source before every candidate.

    WHY WRAP THE EXECUTOR RATHER THAN HOOK THE CONTROLLER. Checking only
    at the two ends of a run leaves the whole middle unobserved, and the
    middle is where the time goes — a four-hour run checked twice is
    making a much weaker claim than one checked at every node. But adding
    a hook to the Controller for this would be a second, less defensible
    edit to the audited state machine.

    Every `ExecutorPort.run` call IS a node boundary: the Controller calls
    it exactly once per candidate evaluation. Wrapping the port we already
    own gets per-node checking for free and touches nothing of Terry's or
    the Controller's. At ~5ms a check against candidate evaluations
    measured in hundreds of seconds, the cost does not register.

    Transparent by construction: it forwards `run` and delegates every
    other attribute, so the wrapped adapter's own surface (`calls`, used
    by the run summary) stays reachable through it.
    """

    def __init__(self, inner: Any, monitor: "IntegrityMonitor") -> None:
        self._inner = inner
        self._monitor = monitor

    def run(self, config: Any, seeds: Sequence[int]) -> Any:
        # Before, not after: a candidate that takes six minutes should be
        # attributed to the code that was on disk when it STARTED.
        self._monitor.check_at_node_boundary(config.config_id)
        return self._inner.run(config, seeds)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class IntegrityMetadata(Mapping):
    """The `run_metadata` the launcher hands the Controller.

    A READ OF THIS MAPPING PERFORMS AN INTEGRITY CHECK. That is the
    design, not a side effect that slipped in, so it is the first thing
    said about it.

    The Controller reads its `run_metadata` exactly twice — once building
    RUN_START and once building RUN_END — and those are precisely the two
    moments a run's endpoints should be verified. Making the read do the
    check is what lets RUN_END carry a summary that is true AT RUN_END
    rather than true at the last node boundary before it. The alternative
    is a post-run check appended after RUN_END, and nothing may follow
    RUN_END without destroying a replay's ability to tell a finished run
    from a killed one (see EventKind.RUN_END).

    So a clean run performs: one check at RUN_START, one per node
    boundary via CheckedExecutor, one at RUN_END. `checks_performed` in
    the summary is the honest total.
    """

    def __init__(
        self,
        monitor: "IntegrityMonitor",
        *,
        relaunch: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._monitor = monitor
        self._relaunch = dict(relaunch) if relaunch is not None else None
        self._reads = 0

    def __getitem__(self, key: str) -> Any:
        if key != "integrity":
            raise KeyError(key)
        self._reads += 1
        self._monitor.check(label=f"run_metadata read #{self._reads}")
        payload: dict[str, Any] = {
            "policy": "autonomy/INTERVENTION_POLICY.md",
            "launch": dict(self._monitor.launch),
            "summary": self._monitor.summary(),
        }
        if self._relaunch is not None:
            payload["relaunch"] = self._relaunch
        return payload

    def __iter__(self):
        return iter(("integrity",))

    def __len__(self) -> int:
        return 1
