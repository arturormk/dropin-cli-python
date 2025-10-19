#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    adr_dir = root / "docs" / "adr"
    index = adr_dir / "README.md"
    if not adr_dir.exists():
        print("ADR directory missing: docs/adr/", file=sys.stderr)
        return 1
    if not index.exists():
        print("ADR index missing: docs/adr/README.md", file=sys.stderr)
        return 1
    # Ensure each ADR *.md (excluding README.md and TEMPLATE.md) is referenced in index
    entries = [p.name for p in adr_dir.glob("*.md") if p.name not in {"README.md", "TEMPLATE.md"}]
    content = index.read_text(encoding="utf-8") if index.exists() else ""
    missing = [e for e in entries if e not in content]
    if missing:
        print("ADR index missing entries:\n - " + "\n - ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
