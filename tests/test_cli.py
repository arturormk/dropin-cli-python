from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure src on path before importing module under test
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cli as dropin  # type: ignore


def test_command_registration_and_help():
    # Define a simple command dynamically to avoid leaking into global state across tests
    @dropin.command(add_args=lambda p: p.add_argument("x", type=int))
    def cmd_inc(args: argparse.Namespace) -> dict[str, Any]:
        """Increment x"""
        return {"x": args.x + 1}

    p = dropin.build_subcommand_parser(prog="tool")
    # Ensure command is registered with help from docstring summary
    sub = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)][0]
    assert "inc" in sub.choices
    assert sub.choices["inc"].description.startswith("Increment x")


def test_dispatch_and_json_output(capsys):
    @dropin.command(add_args=lambda p: p.add_argument("name"))
    def cmd_hello(args: argparse.Namespace):
        """Hello name"""
        return {"hello": args.name}

    rc = dropin.dispatch(["hello", "Alice"])  # default json
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {"hello": "Alice"}


def test_pretty_and_repr_outputs(capsys):
    @dropin.command(add_args=lambda p: None)
    def cmd_obj(args: argparse.Namespace):
        return {"a": 1, "b": [1, 2]}

    rc = dropin.dispatch(["obj", "--pretty"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "\n  \"a\": 1" in out  # pretty JSON contains indentation

    rc = dropin.dispatch(["obj", "--repr"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("{") and "'a': 1" in out


def test_table_output_fallback_plain(capsys, monkeypatch):
    @dropin.command(add_args=lambda p: None)
    def cmd_rows(args: argparse.Namespace):
        return [{"c1": 1, "c2": "x"}, {"c1": 2, "c2": "y"}]

    # Force-disable rich and tabulate by monkeypatching module flags
    monkeypatch.setattr(dropin, "_HAVE_RICH", False)
    monkeypatch.setattr(dropin, "_HAVE_TABULATE", False)

    rc = dropin.dispatch(["rows", "--table", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    # Expect simple header and rows present
    assert "c1" in out and "c2" in out
    assert "1" in out and "x" in out and "2" in out and "y" in out


def test_execute_concurrent_trivial():
    tasks = [1, 2, 3]

    def worker(x: int) -> int:
        return x + 10

    summary = dropin.execute_concurrent(tasks, worker, dropin.ExecOptions(jobs=2, progress=False))
    assert summary.succeeded == 3
    assert summary.failed == 0
    assert sorted([it.result for it in summary.items if it.ok]) == [11, 12, 13]
