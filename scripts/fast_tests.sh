#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src pytest -q -k 'cli or concurrent'
