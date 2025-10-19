# Contributing

Thanks for your interest! This project follows a Software Curator methodology: AI can assist, but humans own direction and releases.

- Add or update tests for behavior changes.
- If you change architecture or process, add an ADR in `docs/adr/` and update the index.
- Keep PRs small and focused; reference ADR IDs if relevant.

## Dev setup

- Python 3.9+
- Install dev deps:

```
pip install -r requirements-dev.txt
pre-commit install
```

Run checks locally:

```
pre-commit run --all-files
PYTHONPATH=src pytest -q
```

## Attribution

See `AUTHORS` and ADR 0010 for AI curation policy. Include the footer `Curated-By: <Your Name>` for large AI-assisted batches.
