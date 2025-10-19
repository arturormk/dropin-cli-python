from __future__ import annotations

"""dropin-cli: a drop-in helper for single-file CLIs.

Features:
- @command registry with docstring-powered help/description
- JSON/YAML/table renderers (rich → tabulate → fallback)
- Progress bar wrapper (rich → tqdm → basic)
- Thread-pool executor with ordered results & timing
- Optional DVDT helpers
"""

import argparse
import datetime as _dt
import enum as _enum
import importlib.util as _importlib_util
import inspect
import json
import os
import pathlib as _pathlib
import shutil
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, cast

__all__ = [
    "command",
    "build_subcommand_parser",
    "dispatch",
    "add_output_format_flags",
    "add_exec_flags",
    "add_debug_flags",
    "render",
    "Progress",
    "ExecOptions",
    "ExecItem",
    "ExecSummary",
    "execute_concurrent",
    "dvdt_run",
    "dvdt_run_concurrent",
    "__version__",
]


# ---- Version ----
__version__ = "0.1.0"


"""
Optional dependencies are imported lazily in the code paths where they are used
to avoid module-level redefinition/type-checking issues and to keep import cost
low for plain JSON/repr usage.
"""

_HAVE_RICH = _importlib_util.find_spec("rich") is not None
_HAVE_TABULATE = _importlib_util.find_spec("tabulate") is not None
_HAVE_YAML = _importlib_util.find_spec("yaml") is not None


# =========================================================
# Section: Subcommand registry/decorator
# =========================================================

_CommandFn = Callable[..., Any]


@dataclass(frozen=True)
class _CmdSpec:
    help: str
    description: Optional[str]
    fn: _CommandFn
    add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None


_COMMANDS: dict[str, _CmdSpec] = {}


def command(
    _fn: Any = None,
    *,
    name: Optional[str] = None,
    help: Optional[str] = None,
    add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> Any:
    """
    Decorator to register a subcommand.

    Usage:
        @command                      # name derives from function (cmd_foo -> "foo")
        def cmd_foo(args): ...

        @command()                    # equivalent
        def cmd_bar(args): ...

        @command("stats")             # positional name
        def cmd_anyname(args): ...

        @command(name="echo", add_args=lambda p: p.add_argument("words", nargs="*"))
        def cmd_echo(args):
            \"\"\"Echo words\"\"\"
            return ...
    """
    cmd_name_from_positional = None
    if isinstance(_fn, str) and name is None:
        cmd_name_from_positional = _fn
        _fn = None

    def _register(fn: _CommandFn) -> _CommandFn:
        # Derive command name if not provided
        derived = fn.__name__
        if derived.startswith("cmd_"):
            derived = derived[4:]
        cmd_name_opt: Optional[str] = name or cmd_name_from_positional or derived
        if not cmd_name_opt:
            raise ValueError("Command name cannot be empty")
        cmd_name = cmd_name_opt
        if cmd_name in _COMMANDS:
            raise ValueError(f"Command '{cmd_name}' is already registered")

        # Help/description from docstring if not provided
        doc = inspect.getdoc(fn)  # dedents & strips, or None
        if doc:
            lines = doc.splitlines()
            summary = lines[0].strip() if lines else ""
            description = textwrap.dedent(doc)
        else:
            summary = help or cmd_name
            description = help

        short_help = help or (summary if summary else cmd_name)

        _COMMANDS[cmd_name] = _CmdSpec(
            help=short_help,
            description=description,
            fn=fn,
            add_args=add_args,
        )
        return fn

    if _fn is None:
        return _register
    if callable(_fn):
        return _register(_fn)
    raise TypeError("command decorator expects a function or an optional positional name string")


# =========================================================
# Section: Output formatting flags & renderer
# =========================================================


def add_output_format_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Output")
    g.add_argument(
        "--format", choices=["json", "pretty", "yaml", "table", "repr"], help="Select output format"
    )
    g.add_argument("--table", action="store_true", help="Render as an ASCII/pretty table")
    g.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    g.add_argument("--yaml", action="store_true", help="YAML output (requires PyYAML)")
    g.add_argument("--repr", action="store_true", help="Python repr() output")
    g.add_argument("--columns", metavar="COLS", help="Comma-separated column list for --table")
    g.add_argument("--limit", type=int, help="Limit rows for --table")
    g.add_argument("--no-color", action="store_true", help="Disable ANSI colors (rich)")


def _resolve_format(args: argparse.Namespace) -> str:
    fmt = getattr(args, "format", None)
    if fmt:
        return str(fmt)
    for name in ("table", "pretty", "yaml", "repr"):
        if getattr(args, name, False):
            return name
    return "json"


def _to_serializable(x: Any) -> Any:
    if is_dataclass(x):
        # Safe: we just checked it's a dataclass instance
        return asdict(cast(Any, x))
    if isinstance(x, (list, tuple)):
        return [_to_serializable(i) for i in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    # Common non-JSON-native types
    if isinstance(x, (_dt.datetime, _dt.date, _dt.time)):
        return x.isoformat()
    if isinstance(x, _pathlib.Path):
        return str(x)
    if isinstance(x, _enum.Enum):
        return _to_serializable(x.value)
    return x


def _rows_from_data(data: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Normalize input into (rows, columns).
    - dict -> [dict]
    - dataclass -> [dict]
    - list[dict] -> as-is (columns = stable union of keys in order of first appearance)
    - list[dataclass] -> list[dict]
    - list[scalar] -> [{"value": item}]
    """
    data = _to_serializable(data)
    if isinstance(data, dict):
        rows: list[dict[str, Any]] = [data]
    elif isinstance(data, list):
        if not data:
            return [], []
        if isinstance(data[0], dict):
            # typing: data is list[dict[str, Any]] here
            from typing import cast

            rows = cast(list[dict[str, Any]], data)
        else:
            rows = [{"value": v} for v in data]
    else:
        rows = [{"value": data}]
    # columns: union in order of first appearance
    cols: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    return rows, cols


def _split_columns(arg: Optional[str], available: list[str]) -> list[str]:
    if not arg:
        return available
    requested = [c.strip() for c in arg.split(",") if c.strip()]
    return [c for c in requested if c in available] or available


def render(obj: Any, args: argparse.Namespace) -> None:
    if obj is None:
        return
    fmt = _resolve_format(args)
    if fmt == "yaml":
        if not _HAVE_YAML:
            print("YAML output requested but PyYAML is not installed.", file=sys.stderr)
            sys.exit(2)
        import yaml as _yaml
        print(_yaml.safe_dump(_to_serializable(obj), sort_keys=False, allow_unicode=True))
        return

    if fmt == "table":
        rows, cols = _rows_from_data(obj)
        if args.limit is not None:
            rows = rows[: max(0, args.limit)]
        cols = _split_columns(getattr(args, "columns", None), cols)
        if not rows:
            print("(no rows)")
            return
        # Try rich first
        if _HAVE_RICH and not getattr(args, "no_color", False):
            from rich.console import Console
            from rich.table import Table as RichTable

            console = Console(stderr=False, force_jupyter=False)
            t = RichTable(show_header=True, header_style="bold")
            for c in cols:
                t.add_column(c)
            for r in rows:
                t.add_row(*[str(r.get(c, "")) for c in cols])
            console.print(t)
            return
        # Fallback to tabulate
        if _HAVE_TABULATE:
            from tabulate import tabulate as _tabulate

            print(
                _tabulate(
                    [[r.get(c, "") for c in cols] for r in rows], headers=cols, tablefmt="github"
                )
            )
            return
        # Last resort: plain text
        widths = [max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
        header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
        sep = "-+-".join("-" * w for w in widths)
        print(header)
        print(sep)
        for r in rows:
            print(" | ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
        return

    if fmt == "pretty":
        print(json.dumps(_to_serializable(obj), indent=2, ensure_ascii=False))
        return

    if fmt == "repr":
        print(repr(obj))
        return

    # default json (compact)
    print(json.dumps(_to_serializable(obj), separators=(",", ":"), ensure_ascii=False))


# =========================================================
# Section: Subcommand parser & dispatcher
# =========================================================


def build_subcommand_parser(
    prog: str = "tool",
    description: Optional[str] = None,
    epilog: Optional[str] = None,
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND", required=True)
    for name, spec in _COMMANDS.items():
        sp = sub.add_parser(name, help=spec.help, description=spec.description or spec.help)
        if spec.add_args:
            spec.add_args(sp)
        # output flags for each command
        add_output_format_flags(sp)
        # debugging flags
        add_debug_flags(sp)
        sp.set_defaults(_fn=spec.fn)
    return p


def dispatch(argv: Optional[Sequence[str]] = None) -> int:
    prog_name = os.path.basename(sys.argv[0]) or "tool"
    p = build_subcommand_parser(prog=prog_name)
    args = p.parse_args(list(argv) if argv is not None else sys.argv[1:])
    try:
        # set via parser.set_defaults(_fn=...) when building subcommands
        from typing import Callable, cast

        fn = cast(Callable[[argparse.Namespace], Any], getattr(args, "_fn"))
        result = fn(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130  # standard SIGINT exit
    except SystemExit:
        raise
    except Exception as e:
        if getattr(args, "trace", False):
            import traceback

            traceback.print_exc()
        else:
            print(f"Error: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1
    render(result, args)
    return 0


# =========================================================
# Section: Progress bar helper (rich → tqdm → basic)
# =========================================================


class Progress:
    """
    A small abstraction over rich/tqdm/TTY that writes to stderr and is safe to no-op.
    Usage:
        pb = Progress(total=N, enabled=True)
        for i, item in enumerate(items, 1):
            ...do work...
            pb.update(i)
        pb.close()
    """

    def __init__(self, total: int, enabled: bool = True, description: str = ""):
        self.total = max(0, int(total))
        self.enabled = bool(enabled) and self.total > 0
        self.description = description or ""
        self._impl = None
        self._done = 0

        if not self.enabled:
            return

        if _HAVE_RICH and sys.stderr.isatty():
            from rich.console import Console
            from rich.progress import BarColumn, TaskProgressColumn, TimeRemainingColumn
            from rich.progress import Progress as RichProgress

            self._console = Console(stderr=True)
            self._progress = RichProgress(
                "[progress.description]{task.description}",
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                "•",
                TimeRemainingColumn(),
                transient=False,
                console=self._console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(self.description or "working", total=self.total)
            self._impl = "rich"
            return

        try:
            from tqdm import tqdm as _tqdm

            self._tqdm = _tqdm(
                total=self.total,
                desc=self.description,
                unit="it",
                file=sys.stderr,
                dynamic_ncols=True,
            )
            self._impl = "tqdm"
            return
        except Exception:
            pass

        # fallback: simple carriage-return bar
        self._impl = "basic"

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        done = max(0, min(done, self.total))
        self._done = done
        if self._impl == "rich":
            self._progress.update(self._task_id, completed=done)
        elif self._impl == "tqdm":
            delta = done - int(self._tqdm.n)
            if delta > 0:
                self._tqdm.update(delta)
        else:
            # basic
            if not sys.stderr.isatty():
                return
            width = max(10, min(60, shutil.get_terminal_size(fallback=(80, 24)).columns - 30))
            filled = int(width * done / self.total) if self.total else width
            bar = "#" * filled + "-" * (width - filled)
            sys.stderr.write(f"\r[{bar}] {done}/{self.total}")
            sys.stderr.flush()

    def close(self) -> None:
        if not self.enabled:
            return
        if self._impl == "rich":
            self._progress.update(self._task_id, completed=self.total)
            self._progress.stop()
        elif self._impl == "tqdm":
            self._tqdm.close()
        else:
            if sys.stderr.isatty():
                sys.stderr.write("\n")
                sys.stderr.flush()


# =========================================================
# Section: Exec flags helpers (jobs, progress)
# =========================================================


def add_exec_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Execution")
    g.add_argument("--jobs", type=int, metavar="N", help="Max parallel workers (default: auto)")
    g.add_argument("--progress", action="store_true", help="Show a live progress bar on stderr")


def add_debug_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Debugging")
    g.add_argument("--trace", action="store_true", help="On error, print full traceback")


# =========================================================
# Section: Concurrent executor (thread pool + progress)
# =========================================================

T = TypeVar("T")  # task
R = TypeVar("R")  # worker result


@dataclass(frozen=True)
class ExecOptions:
    jobs: Optional[int] = None
    progress: bool = False
    progress_desc: str = "Working"
    thread_name_prefix: str = "worker"
    cancel_on_interrupt: bool = True


@dataclass(frozen=True)
class ExecItem(Generic[T, R]):
    task: T
    ok: bool
    result: Optional[R] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ExecSummary(Generic[T, R]):
    items: tuple[ExecItem[T, R], ...]
    elapsed_ms: int
    succeeded: int
    failed: int


def execute_concurrent(
    tasks: Sequence[T],
    worker: Callable[[T], R],
    opts: Optional[ExecOptions] = None,
) -> ExecSummary[T, R]:
    """Run tasks in a thread pool, show progress if requested, and time the run."""

    opts = opts or ExecOptions()

    if not tasks:
        return ExecSummary(items=(), elapsed_ms=0, succeeded=0, failed=0)

    from contextlib import suppress

    start = time.perf_counter()
    max_workers = opts.jobs or min(32, (os.cpu_count() or 1) * 2 + 4)

    pb = Progress(total=len(tasks), enabled=opts.progress, description=opts.progress_desc)

    slots: list[Optional[ExecItem[T, R]]] = [None] * len(tasks)
    done = 0
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=opts.thread_name_prefix
    ) as ex:
        fut2i = {ex.submit(worker, t): i for i, t in enumerate(tasks)}
        try:
            for fut in as_completed(fut2i):
                i = fut2i[fut]
                try:
                    res = fut.result()
                    slots[i] = ExecItem(task=tasks[i], ok=True, result=res)
                except Exception as e:
                    msg = f"{e.__class__.__name__}: {e}"
                    slots[i] = ExecItem(task=tasks[i], ok=False, error=msg)
                done += 1
                pb.update(done)
        except KeyboardInterrupt:
            if opts.cancel_on_interrupt:
                for fut in fut2i:
                    with suppress(Exception):
                        fut.cancel()
        finally:
            pb.close()

    elapsed_ms = round(1000 * (time.perf_counter() - start))
    items: list[ExecItem[T, R]] = [
        s if s is not None else ExecItem(task=tasks[i], ok=False, error="interrupted")
        for i, s in enumerate(slots)
    ]
    succ = sum(1 for s in items if s.ok)
    fail = len(items) - succ
    return ExecSummary(items=tuple(items), elapsed_ms=elapsed_ms, succeeded=succ, failed=fail)


# =========================================================
# Section: DVDT helpers (data-returning)
# =========================================================

# DVDT: Discover → Validate → Do → Tell
# - Discover: collect inputs → Policy, expand to Plan (tasks)
# - Validate: assert preconditions; fail fast (no side effects)
# - Do: perform side-effecting work (I/O, network, DB)
# - Tell: summarize/return structured results (no heavy logic)


def dvdt_run(
    args: argparse.Namespace,
    build_policy: Callable[[argparse.Namespace], Any],
    build_plan: Callable[[Any], Any],
    validate: Callable[[Any], None],
    execute: Callable[[Any], Any],
    prepare: Optional[Callable[[Any], None]] = None,
    dry_run_attr: str = "dry_run",
    report_dry_run: Optional[Callable[[Any], Any]] = None,  # returns data
    to_output: Optional[Callable[[Any, Any], Any]] = None,  # (plan, results) -> data
) -> Any:
    """Generic DVDT launcher for single-command tools (returns data, does not print)."""
    policy = build_policy(args)
    plan = build_plan(policy)
    validate(plan)
    if getattr(args, dry_run_attr, False):
        return report_dry_run(plan) if report_dry_run else {"dry_run": True}
    if prepare:
        prepare(plan)
    results = execute(plan)
    return to_output(plan, results) if to_output else results


def dvdt_run_concurrent(
    args: argparse.Namespace,
    build_policy: Callable[[argparse.Namespace], Any],
    build_plan: Callable[[Any], Any],
    validate: Callable[[Any], None],
    iter_tasks: Callable[[Any], Sequence[T]],  # from plan -> tasks
    worker_factory: Callable[
        [Any, Any], Callable[[T], R]
    ],  # (policy, plan) -> worker(task)->result
    to_output: Callable[[Any, ExecSummary[T, R]], Any],  # (plan, summary) -> data
    jobs_attr: str = "jobs",
    progress_attr: str = "progress",
    progress_desc: str = "Working",
    thread_name_prefix: str = "worker",
) -> Any:
    """
    DVDT with built-in thread pool + progress + timing.
    Returns data (dict/list/dataclass) for your renderer.
    """
    policy = build_policy(args)
    plan = build_plan(policy)
    validate(plan)
    tasks = iter_tasks(plan)
    worker = worker_factory(policy, plan)
    opts = ExecOptions(
        jobs=getattr(args, jobs_attr, None),
        progress=bool(getattr(args, progress_attr, False)),
        progress_desc=progress_desc,
        thread_name_prefix=thread_name_prefix,
    )
    summary = execute_concurrent(tasks, worker, opts)
    return to_output(plan, summary)
