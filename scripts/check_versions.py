#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Ensure we can import from the workspace
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    try:
        import cli  # type: ignore
        import dropin_cli  # type: ignore
    except Exception as e:
        print(f"Import error during version check: {e}", file=sys.stderr)
        return 1

    v_dropin = getattr(cli, "__version__", None)
    v_pkg = getattr(dropin_cli, "__version__", None)

    if not v_dropin or not v_pkg:
        print(
            f"Missing __version__: cli={v_dropin!r} dropin_cli={v_pkg!r}",
            file=sys.stderr,
        )
        return 1

    if v_dropin != v_pkg:
        print(
            f"Version mismatch: src/cli.py={v_dropin} vs dropin_cli={v_pkg}",
            file=sys.stderr,
        )
        return 1

    print(f"Versions OK: {v_pkg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
