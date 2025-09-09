#!/usr/bin/env python3

"""
Batch-tint monochrome PNG icons with a brand color, preserving transparency.

This is a more elaborate example of a CLI with a single command that performs
a non-trivial task using the DVDT (Discover, Validate, Do, Tell) framework.
"""

import argparse
import functools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image
from cli import command, dispatch, dvdt_run_concurrent

# -------- RGB helpers --------

RE_HEX6 = re.compile(r"^#[0-9A-Fa-f]{6}$")
RGB = tuple[int, int, int]
LUTS = tuple[bytes, bytes, bytes]

def hex_color(s: str) -> str:
    """Validate '#RRGGBB' and normalize to uppercase"""
    if not RE_HEX6.fullmatch(s or ""):
        raise argparse.ArgumentTypeError(
            f"Invalid color '{s}'; expected '#RRGGBB' (e.g. #FDC700)"
        )
    return s.upper()

def parse_hex_rgb(s: str) -> RGB:
    s = s.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)

# -------- Image manipulation --------

def make_luts(rgb: RGB) -> LUTS:
    rC, gC, bC = rgb
    lut_r = bytes(int(rC * x / 255) for x in range(256))
    lut_g = bytes(int(gC * x / 255) for x in range(256))
    lut_b = bytes(int(bC * x / 255) for x in range(256))
    return lut_r, lut_g, lut_b

def apply_tint_rgba_with_luts(img: Image.Image, luts: LUTS) -> Image.Image:
    """Map luminance of img to color_rgb, preserving alpha"""
    img = img.convert("RGBA")
    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    L = rgb.convert("L", matrix=(0.2126, 0.7152, 0.0722, 0))
    lut_r, lut_g, lut_b = luts
    R = L.point(lut_r)
    G = L.point(lut_g)
    B = L.point(lut_b)
    return Image.merge("RGBA", (R, G, B, a))

def make_preview(tinted: Image.Image, bg_rgb: RGB) -> Image.Image:
    """Composite tinted icon on a solid background color."""
    bg = Image.new("RGBA", tinted.size, bg_rgb + (255,))
    return Image.alpha_composite(bg, tinted)

# -------- tint --------

@dataclass(frozen=True)
class TintIconsPolicy:
    in_root: Path
    out_root: Path
    color_rgb: RGB
    size: Optional[int] = None
    preview_bg: Optional[RGB] = None
    jobs: Optional[int] = None
    newer_only: bool = False
    progress: bool = False

@dataclass(frozen=True)
class TintIconsTask:
    src: Path
    dst: Path
    will_resize: bool

@dataclass(frozen=True)
class TintIconsPlan:
    policy: TintIconsPolicy
    tasks: tuple[TintIconsTask, ...]

@dataclass(frozen=True)
class TintIconsTaskResult:
    task: TintIconsTask
    success: bool
    error: Optional[str] = None

def tint_add_args(p):
    io = p.add_argument_group("I/O")
    io.add_argument("--in", dest="in_path", metavar="PATH", required=True, help="Input file or directory")
    io.add_argument("--out", dest="out_path", metavar="DIR", required=True, help="Output directory")
    color = p.add_argument_group("Color")
    color.add_argument("--color", metavar="#RRGGBB", type=hex_color, required=True, help="Tint color")
    color.add_argument("--preview-bg", metavar="#RRGGBB", type=hex_color, help="Also write preview on this background")
    proc = p.add_argument_group("Processing")
    proc.add_argument("--size", metavar="N", type=int, help="Fit icon into an NxN transparent canvas")
    proc.add_argument("--newer-only", action="store_true", help="Skip when dst exists and is newer")
    perf = p.add_argument_group("Performance")
    perf.add_argument("--jobs", metavar="N", type=int, default=None, help="Max parallel workers (default: auto)")
    misc = p.add_argument_group("Misc")
    misc.add_argument("--dry-run", action="store_true", help="Show plan and exit")
    misc.add_argument("--progress", action="store_true", help="Show a live progress bar")

def tint_build_policy(args):
    return TintIconsPolicy(
        in_root=Path(args.in_path),
        out_root=Path(args.out_path),
        color_rgb=parse_hex_rgb(args.color),
        size=args.size,
        preview_bg=parse_hex_rgb(args.preview_bg) if args.preview_bg else None,
        jobs=args.jobs,
        newer_only=args.newer_only,
        progress=args.progress,
    )

def tint_build_plan(pol):
    if pol.in_root.is_file():
        if pol.in_root.suffix.lower() != ".png":
            raise ValueError(f"input file must be .png: {pol.in_root}")
        sources = [pol.in_root]
        base = pol.in_root.parent
    else:
        base = pol.in_root
        sources = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
    tasks = []
    for src in sources:
        rel = src.relative_to(base)
        dst = (pol.out_root / rel).with_suffix(".png")
        tasks.append(TintIconsTask(src=src, dst=dst, will_resize=bool(pol.size)))
    return TintIconsPlan(policy=pol, tasks=tuple(tasks))

def tint_validate(plan):
    def tint_detect_dst_collisions(plan):
        seen = set()
        dup = []
        for task in plan.tasks:
            if task.dst in seen:
                dup.append(task.dst)
            seen.add(task.dst)
        if dup:
            raise ValueError("output collisions detected: " + ", ".join(map(str, dup)))

    pol = plan.policy
    if pol.size is not None and pol.size <= 0:
        raise ValueError(f"--size {pol.size}, must be a positive integer")
    if not pol.in_root.exists():
        raise FileNotFoundError(f"--in '{pol.in_root}', input not found")
    if pol.in_root.resolve() == pol.out_root.resolve():
        raise ValueError(f"--out '{pol.out_root}' must differ from --in '{pol.in_root}'")
    inr, outr = pol.in_root.resolve(), pol.out_root.resolve()
    if outr == inr or outr in inr.parents or inr in outr.parents:
        raise ValueError(f"--out '{pol.out_root}' must not be inside --in '{pol.in_root}', or vice versa")
    tint_detect_dst_collisions(plan)

def tint_iter_tasks(plan):
    def tint_filter_tasks(tasks):
        def is_outdated(t):
            if not t.dst.exists():
                return True
            return t.dst.stat().st_mtime < t.src.stat().st_mtime
        kept = [t for t in tasks if is_outdated(t)]
        before, after = len(tasks), len(kept)
        if before != after:
            print(f"[info] Skipped {before - after} up-to-date file(s).", file=sys.stderr)
        return tuple(kept)

    tasks = plan.tasks
    if plan.policy.newer_only:
        tasks = tint_filter_tasks(tasks)
    # No side effects here (no mkdir). Directories are created in the worker.
    return tasks

def tint_process_task(t, pol, luts):
    try:
        with Image.open(t.src) as im:
            img = im.convert("RGBA")
            if t.will_resize and pol.size:
                img.thumbnail((pol.size, pol.size), Image.LANCZOS)
                canvas = Image.new("RGBA", (pol.size, pol.size), (0, 0, 0, 0))
                ox = (pol.size - img.width) // 2
                oy = (pol.size - img.height) // 2
                canvas.paste(img, (ox, oy))
                img = canvas
            tinted = apply_tint_rgba_with_luts(img, luts)

            # Ensure destination directory exists (Do-phase side effect)
            t.dst.parent.mkdir(parents=True, exist_ok=True)

            tinted.save(t.dst, optimize=True, compress_level=9)
            if pol.preview_bg:
                preview_img = make_preview(tinted, pol.preview_bg)
                preview_path = t.dst.with_name(f"preview-{t.dst.name}")
                preview_img.save(preview_path, optimize=True, compress_level=9)
            return TintIconsTaskResult(task=t, success=True)
    except Exception as e:
        return TintIconsTaskResult(task=t, success=False, error=str(e))

def tint_make_worker(pol, plan):
    return functools.partial(tint_process_task, pol=pol, luts=make_luts(pol.color_rgb))

def tint_to_output(plan, summary):
    failures = [
        {"src": str(it.task.src), "dst": str(it.task.dst), "error": it.error}
        for it in summary.items
        if not it.ok
    ]
    return {
        "total": len(summary.items),
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "elapsed_ms": summary.elapsed_ms,
        "output_dir": str(plan.policy.out_root),
        "failures": failures,
    }

@command(add_args=tint_add_args)
def cmd_tint(args):
    """Batch-tint monochrome PNG icons with a brand color, preserving transparency."""
    # Honor --dry-run by summarizing tasks with no side effects
    pol = tint_build_policy(args)
    plan = tint_build_plan(pol)
    tint_validate(plan)
    tasks = tint_iter_tasks(plan)
    if args.dry_run:
        sample = [{"src": str(t.src), "dst": str(t.dst), "resize": t.will_resize} for t in tasks[:10]]
        return {"dry_run": True, "total": len(tasks), "sample": sample, "more": max(0, len(tasks) - len(sample))}
    # Execute concurrently with progress/timing
    return dvdt_run_concurrent(
        args,
        build_policy=lambda _: pol,
        build_plan=lambda __: plan,
        validate=lambda _: None,          # already validated
        iter_tasks=lambda __: tasks,      # already filtered, no side effects
        worker_factory=tint_make_worker,
        to_output=tint_to_output,
        progress_desc="Tinting",
        thread_name_prefix="tint",
    )

# -------- main --------

if __name__ == "__main__":
    raise SystemExit(dispatch())
