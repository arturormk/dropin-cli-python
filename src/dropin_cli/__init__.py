from __future__ import annotations

"""
dropin_cli package: thin re-export of the single-source drop-in module `cli.py`.

This package exists for users who prefer `pip install` over copying the file.
The only source of logic is `src/cli.py`.
"""

from cli import (
    ExecItem,
    ExecOptions,
    ExecSummary,
    Progress,
    __version__,
    add_debug_flags,
    add_exec_flags,
    add_output_format_flags,
    build_subcommand_parser,
    command,
    dispatch,
    dvdt_run,
    dvdt_run_concurrent,
    execute_concurrent,
    render,
)

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
