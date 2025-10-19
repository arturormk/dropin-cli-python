#!/usr/bin/env python3

"""
A very simple example of a CLI with multiple commands.

Usage:

  $ python examples/echo-words.py echo hello world
  $ python examples/echo-words.py reverse-echo hello world

  $ python examples/echo-words.py echo hello world --pretty
  $ python examples/echo-words.py echo hello world --yaml
  $ python examples/echo-words.py echo hello world --table

  $ python examples/echo-words.py echo hello world --table --columns=idx
  $ python examples/echo-words.py echo one two three --table --limit 2
"""

from cli import command, dispatch

@command(add_args=lambda p: p.add_argument("words", nargs="*"))
def cmd_echo(args):
    """Echo words"""
    # return structured data; renderer will format
    return [{"idx": i, "word": w} for i, w in enumerate(args.words)]

@command(name="reverse-echo", add_args=lambda p: p.add_argument("words", nargs="*"))
def cmd_reverse_echo(args):
    """Reverse echo words"""
    # return structured data; renderer will format
    return [{"idx": i, "word": w} for i, w in enumerate(reversed(args.words))]

if __name__ == "__main__":
    raise SystemExit(dispatch())
