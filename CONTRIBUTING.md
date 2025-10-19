# Contributing

Thanks for your interest in improving this project. This repository also models a “Software Curator” posture: small, well‑tested, well‑documented, and automation‑friendly.

- Use Python 3.9+
- Run tests locally before committing: `PYTHONPATH=src pytest -q`
- Lint: `ruff check .` (and optionally `ruff format`)
- Type check: `mypy src tests scripts` (strict mode can be considered later)
- Keep changes focused and add tests for behavior changes
- If you change user‑visible behavior or architecture/process, add/update an ADR in `docs/adr/` and update the index

Pre‑commit (recommended):
```
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

## Development quickstart

1) Create a virtualenv and install dev deps
	- `pip install -r requirements-dev.txt`
2) Run tests and linters
	- `PYTHONPATH=src pytest -q`
	- `ruff check .`
	- `mypy src tests scripts`
3) Make changes in `src/` (drop‑in file is `src/cli.py`; installable package is `src/dropin_cli/`)
4) Add or update tests in `tests/` and keep them fast and deterministic

Optional packaging/build check:
```
python -m build
```
This ensures the repository can be packaged (sdist/wheel) when used as an installable library.

## Releasing

- Version is declared in `pyproject.toml` (`[project].version`). A mirrored `__version__` exists in code; keep them consistent when bumping.
- Tag as `vX.Y.Z` and push tags. Consider creating a GitHub Release with notes.
- Ensure CI is green before tagging.

## Attribution

See `AUTHORS` and ADR‑0010 for the AI curation policy. For large AI‑assisted batches, include the footer:

```
Curated-By: <Your Name>
```
