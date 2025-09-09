# Changelog
All notable changes to this project will be documented in this file.

## [0.1.0] – 2025-09-09
### Added
- Initial public release of `cli.py` (drop-in, single-file helper).
- `@command` decorator with docstring-derived help/description.
- Output renderers: JSON, pretty JSON, YAML, table (rich → tabulate → fallback).
- Progress bar wrapper (rich → tqdm → basic TTY).
- Subcommand parser/dispatcher.
- Examples: `echo-words.py`, `icons.py`.
- Documentation: README, screenshot, MIT LICENSE.
