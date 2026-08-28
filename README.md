# TikTokTechJam

An autonomous ML research agent that proposes, runs, and evaluates
recommender-system methods against the [KuaiRand](https://kuairand.com/)
benchmark. The agent searches over a library of candidate methods,
backtests them through a noise-aware harness, and journals its decisions
so the search process stays auditable.

## Project layout

The codebase is split into four independently-owned packages plus a
shared contract module, so four people can build in parallel without
stepping on each other's files:

| Package      | Owns                                                              |
|--------------|--------------------------------------------------------------------|
| `harness/`   | Data loading, splits, metrics, the noise gate, and caching        |
| `controller/`| The agent's state machine, search policy, and journal writing     |
| `executor/`  | Sandboxed execution, error taxonomy, automated repair, telemetry  |
| `methods/`   | The method library, prompt templates, and hypothesis generation   |

Cross-package interfaces live in `contracts.py` at the repo root — import
from there rather than reaching into another package's internals.

Other top-level directories:

- `vendor/` — the organizer's starter kit, vendored unmodified (added later).
- `scripts/` — one-off/CLI entry points (e.g. `characterize.py`).
- `tests/` — test suite.
- `data/`, `artifacts/` — local, gitignored working directories for
  datasets and run outputs.

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verify the packages import cleanly:

```bash
python -c "import harness, controller, executor, methods"
```
