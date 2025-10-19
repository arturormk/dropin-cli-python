# dropin-cli-python

[![CI](https://github.com/arturormk/dropin-cli-python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/arturormk/dropin-cli-python/actions/workflows/ci.yml)

Minimal, dependency‑light helper to build single‑file CLIs in Python:
- Collects command functions automatically with a decorator.
- Prints results as JSON, YAML, or ASCII/pretty tables.
- Optional progress bars and a simple thread‑pool runner.
- Batteries for a pragmatic DVDT (Discover → Validate → Do → Tell) workflow.

Works great for quick internal utilities with a polished command-line UX.

> **Python:** 3.9+ (uses modern type hints)
> **Dependencies:** none required. Optional: `rich`, `tabulate`, `PyYAML`, `tqdm`.
> **Examples:** `Pillow` is needed only for `icons.py`.

---

## TL;DR

- Copy `src/cli.py` into your repo (or add `src/` to `PYTHONPATH`).
- Write your commands in your own Python file and use the decorator + dispatcher:

```python
from cli import command, dispatch

@command(add_args=lambda p: p.add_argument("name"))
def cmd_hello(args):
    return {"hello": args.name}

if __name__ == "__main__":
    raise SystemExit(dispatch())
```

- Run it: `python mytool.py hello world` (prints JSON by default, or `--table`, `--pretty`, `--yaml`).
- No required dependencies: pure standard library. If available, it will use `rich` (pretty tables), `tabulate` (ASCII tables), `PyYAML` (YAML output), and `tqdm` (progress bars) for nicer UX.

## Highlights

- Pure standard library by default (no required dependencies)
- Rich output options: JSON, YAML, pretty/ASCII tables (with graceful fallbacks)
- Simple progress bar abstraction (rich → tqdm → basic)
- Small concurrent executor with ordered results and timings
- Tiny DVDT helpers to keep CLIs testable and organized

## Install

You can run the script directly or install as a package.

- Local script: clone and use `src/cli.py` directly (see Quick start below)
- Package build: `python -m build` then `pip install dist/dropin_cli_python-*.whl`
- Editable dev install: `pip install -e .`

Optional extras (pretty output/progress/YAML):

- `pip install 'dropin-cli-python[pretty]'` or `pip install -e .[pretty]`

## Try it

Option A — No install (use the drop-in file)

```bash
# Optional: create a venv
python -m venv .venv
source .venv/bin/activate

# Run a tiny demo using the drop-in module (no dependencies required)
PYTHONPATH=src python - <<'PY'
from cli import command, dispatch
@command(add_args=lambda p: p.add_argument("name"))
def cmd_hello(args): return {"hello": args.name}
raise SystemExit(dispatch(["hello", "world"]))
PY

# Table output (ASCII fallback without extras)
PYTHONPATH=src python - <<'PY'
from cli import command, dispatch
@command(add_args=lambda p: p.add_argument("words", nargs="*"))
def cmd_echo(args): return [{"idx": i, "word": w} for i, w in enumerate(args.words)]
raise SystemExit(dispatch(["echo", "one", "two", "--table", "--no-color"]))
PY
```

Option B — Installed package

```bash
# Build and install locally
python -m pip install --upgrade pip build
python -m build
pip install dist/dropin_cli_python-*.whl

# One-off demo using the installed package
python - <<'PY'
from dropin_cli import command, dispatch
@command(add_args=lambda p: p.add_argument("name"))
def cmd_hello(args): return {"hello": args.name}
raise SystemExit(dispatch(["hello", "world"]))
PY
```

Optional pretty output and progress

```bash
# Install extras once to enable rich/tabulate/yaml/tqdm
pip install 'dropin-cli-python[pretty]'

# Now the table demo renders with rich styling (no --no-color needed)
PYTHONPATH=src python - <<'PY'
from cli import command, dispatch
@command(add_args=lambda p: p.add_argument("words", nargs="*"))
def cmd_echo(args): return [{"idx": i, "word": w} for i, w in enumerate(args.words)]
raise SystemExit(dispatch(["echo", "alpha", "beta", "--table"]))
PY
```

## Quick start (two ways)

Option A — Drop-in single file (no install):

1) Copy `src/cli.py` next to your script (or add `src/` to `PYTHONPATH`).
2) Define commands with `@command` and return data; `dispatch()` handles parsing and output.

````python
# mytool.py
from cli import command, dispatch

@command(add_args=lambda p: p.add_argument("words", nargs="*"))
def cmd_echo(args):
    """Echo words"""
    return [{"idx": i, "word": w} for i, w in enumerate(args.words)]

if __name__ == "__main__":
    raise SystemExit(dispatch())
`````

Option B — Installed package:

````python
# mytool.py
from dropin_cli import command, dispatch

@command(add_args=lambda p: p.add_argument("words", nargs="*"))
def cmd_echo(args):
    """Echo words"""
    return [{"idx": i, "word": w} for i, w in enumerate(args.words)]

if __name__ == "__main__":
    raise SystemExit(dispatch())
````

Optional extras:
- Install pretty table/progress/YAML support via extras: `pip install dropin-cli-python[pretty]`
- Extras include: `rich`, `tabulate`, `PyYAML`, `tqdm`

## Examples

See `examples/` for small demo programs:

- `echo-words.py` — minimal command and table output
- `icons.py` — image/emoji demo (requires Pillow)
- `cli.py` — a fuller example harness

## The DVDT (Discover, Validate, Do, Tell) Pattern

DVDT is a lightweight pattern for structuring small CLIs so they stay clear and testable:

- Discover: collect inputs and environment, derive a policy.
- Validate: assert preconditions; fail fast without side effects.
- Do: perform the work (I/O, network, DB), optionally respecting --dry-run.
- Tell: return structured results that the renderer will format.

Use it when a script grows beyond a couple of lines; ignore it for trivial cases.

DVDT emphasizes safe, observable execution: Discover and Validate separate decision-making from effects so you can fail early without mutating state or touching external systems; Do is where effects happen and where you can opt into concurrency, capture per-task failures, and expose progress indicators; Tell closes the loop by summarizing what was attempted and achieved (and what failed), returning structured data that can be rendered as JSON/YAML/tables for operators and logs.

Minimal example with dvdt_run:

```python
from cli import command, dvdt_run

@command(add_args=lambda p: p.add_argument("--dry-run", action="store_true"))
def cmd_sync(args):
    """Sync items from A to B"""
    def build_policy(args): return {"dry": args.dry_run}
    def build_plan(policy): return [{"id": i} for i in range(3)]
    def validate(plan): assert plan, "nothing to do"
    def execute(plan):
        done = []
        for item in plan:
            # perform side effects here
            done.append({"id": item["id"], "status": "ok"})
        return done
    return dvdt_run(args, build_policy, build_plan, validate, execute, dry_run_attr="dry_run")
```

## Screenshot

![Example](docs/screenshot-1.png)

---

## Design decisions (ADRs)

Architectural choices are captured in `docs/adr/`. See the index here:

- `docs/adr/README.md`

## Attribution & Curation

This project is AI-assisted and human curated. See ADR-0010 for the AI curation policy. Default assistant model is "Claude Sonnet 4" enabled for all clients (see `.curator/config.yaml`).

## Development quickstart

- Python 3.9+
- Create and activate a venv, then:

```
pip install -r requirements-dev.txt
pre-commit install
PYTHONPATH=src pytest -q
ruff check . && ruff format --check .
```

CI runs lint, type checks, tests, build, and ADR index verification.

## Contributing

See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. Run CI checks locally:

```
ruff check .
mypy src tests scripts
pytest -q
```

### Pre-commit hooks

This repo ships a `.pre-commit-config.yaml` to enforce fast hygiene and curation policy checks locally.

Enable:

```
pip install -r requirements-dev.txt
pre-commit install
pre-commit install --hook-type pre-push
```

Run all hooks manually:

```
pre-commit run --all-files
```

## License

MIT License — see `LICENSE` for details.
