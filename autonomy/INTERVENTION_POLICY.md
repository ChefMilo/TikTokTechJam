# What counts as a manual intervention

This is the definition the autonomy artifact's headline number rests on.
It is written down so it can be argued with, because a count is only as
good as the rule that produced it.

## The claim

An unattended run reports **"N manual interventions"**. For that number to
mean anything, two things have to be true at once:

1. Every real manual touch produces an `EventKind.INTERVENTION` event.
2. Every `EventKind.INTERVENTION` event is a real manual touch.

The second half is the one that is easy to lose. `executor/report.py`
counts *every* INTERVENTION event regardless of its `type`, so anything
we log "for completeness" lands in the headline number. If autonomous
crash-recovery were logged as an INTERVENTION, a run that recovered
itself three times — the autonomy story working exactly as intended —
would report three interventions and look worse than a run that never had
to. That is why the rule below is exclusive rather than generous.

## An intervention IS

A genuine manual touch of the running system:

| What | Detected by | `type` |
|---|---|---|
| A human edits code while the run is in flight | `code_fingerprint` drift between checks | `code_changed_midrun` |
| A human restarts an interrupted run by hand | prior journal not ending in `RUN_END`, and either no `--resume` or a changed fingerprint | `manual_restart` |
| A prior interrupted run cannot be verified | prior run recorded no launch fingerprint | `unknown_prior` |

`unknown_prior` counts. An unverifiable resume is not a verified one, and
the tie breaks against us on purpose — the number is supposed to be hard
to flatter.

## An intervention is NOT

None of these change what the run computes, so none of them are logged:

- **Watching.** An operator reading the console, tailing the journal with
  `Get-Content -Wait`, or opening the artifacts directory.
- **Inspecting.** `git status`, `git log`, `git diff` in another shell;
  reading any source file; attaching a debugger in read-only mode.
- **Editing things the run does not depend on.** Tests, README, notes.
  `code_fingerprint` covers `executor/`, `harness/`, `controller/`,
  `methods/` and `autonomy/` — not `tests/`, and not `scripts/` (the
  launcher is already running; a later edit to it cannot reach the
  process in flight).
- **Autonomous crash-recovery.** A relaunch with `--resume` over
  byte-identical source. No human touched the code, so nothing is counted
  — the classification is still recorded in `RUN_START` metadata, where a
  reviewer can see it happened.

The rule underneath all four: *did a human change what the running system
does?* Looking is not touching.

## Why a zero is credible here

It would be trivial to report zero by never emitting the event — which is
what the repo did before this change, and the number was meaningless.
What makes it mean something is the **positive** record, written into the
journal's own `RUN_START` and `RUN_END` payloads:

- `commit` and `dirty` — the run started from a named commit, with a
  working tree that was clean (or was not, and says which files).
- `code_hash` — a sha256 over the source that actually runs, so an
  uncommitted edit is caught even though `git status` at launch and the
  commit hash never would.
- `checks_performed` — how many times that hash was re-verified during
  the run. Once at each end is a weak claim; once per node boundary
  across four hours is a strong one, and the number is what tells them
  apart.
- `verified` — the badge. Four conditions, all required (see below).

## The `verified` rule

`VERIFIED AUTONOMOUS` is true only when **all four** hold:

1. **The working tree was known clean at launch.** Not dirty, and not
   unknown — a run that cannot say what code it ran cannot claim to have
   run it unattended. `--require-clean` turns this into a refusal to
   start rather than a badge withheld afterwards, and the real artifact
   run passes it.
2. **The source fingerprint never moved.** Later nodes ran the same code
   as earlier ones.
3. **Nothing was counted as a manual intervention.**
4. **At least `min_candidates` candidates were integrity-checked at a
   node boundary** (default 3, `--min-verified-candidates`).

Condition 4 was added after the first version of this document, and it is
worth saying why. Without it, the cheapest route to a green badge is the
*shortest possible run*: evaluate nothing, check the source twice at the
two endpoints, and report "verified, fingerprint stable". Every word of
that is true and none of it is worth anything — nothing happened between
the two checks, so their agreeing says nothing about whether an agent
worked unattended. The floor makes the badge mean "this run did real work
and stayed clean throughout" rather than "this run was too short to go
wrong".

The count is of **node-boundary** checks specifically, not of checks in
general. A run whose executor was never wrapped in `CheckedExecutor`
still accumulates the two endpoint checks and would otherwise look
verified while the entire middle went unobserved.

When the badge is false, `unverified_because` names every condition that
failed, and the rendered `autonomy.md` prints them. An unexplained
"false" is barely more useful than no field at all.

A reviewer can attack any of these. That is the point: "checked 7 times,
stable, clean at launch, commit 62a61f8" is a falsifiable claim, and "0"
on its own is not.

## Known limitations

- **Data is not fingerprinted.** A changed CSV under `data/` invalidates a
  run's results as surely as changed code, and nothing here notices.
  Out of scope for this change; it answers a different question (what the
  run was computed *from*, rather than whether its code held still).
- **Checks are sampled, not continuous.** An edit made and reverted
  between two checks is invisible. Node-boundary checking narrows the
  window to one candidate evaluation; it does not close it.
- **Resume is not state resume.** `autonomous_resume` means the process
  restarted itself over unchanged code. It does *not* mean the Controller
  picked up where it left off — `Controller.run()` always starts at
  `Stage.INIT`, and this change does not alter that. Calling it a resume
  of the *run* would overclaim.
- **`git` may be unavailable.** Provenance degrades to `"unknown"` rather
  than failing the run, and `dirty` is `None` — distinct from `False`, so
  "no idea" is never rendered as "clean".
