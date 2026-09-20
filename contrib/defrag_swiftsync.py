#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Defragment the body and blob files of an offline SwiftSync database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from swiftsync_fragmentation import extent_count

FILES = ("body", "blobs")
GIB = 1 << 30
MIB = 1 << 20


def mapped_by_process(path: Path) -> list[int]:
    path = path.resolve()
    users: list[int] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            mappings = (process / "maps").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(line.split()[-1] == os.fspath(path) for line in mappings.splitlines()):
            users.append(int(process.name))
    return users


def metrics(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_blocks * 512, extent_count(path)


def describe(name: str, values: tuple[int, int, int]) -> str:
    size, allocated, extents = values
    average = size / extents / MIB if extents else 0.0
    density = extents / (size / GIB) if size else 0.0
    return (
        f"{name}: size={size / GIB:.3f} GiB allocated={allocated / GIB:.3f} GiB "
        f"extents={extents} extents/GiB={density:.1f} avg_extent={average:.3f} MiB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="offline .swiftsync-utxos-* directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report fragmentation; do not modify files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = args.directory.resolve()
    paths = [directory / name for name in FILES]

    if shutil.which("filefrag") is None or (not args.check and shutil.which("e4defrag") is None):
        print("error: filefrag and e4defrag from e2fsprogs are required", file=sys.stderr)
        return 1

    try:
        for path in paths:
            if not path.is_file():
                raise RuntimeError(f"missing regular file: {path}")
            users = mapped_by_process(path)
            if users:
                raise RuntimeError(f"{path} is mapped by process(es) {users}")

        before = {path.name: metrics(path) for path in paths}
        print(f"database={directory}")
        for name in FILES:
            print("before " + describe(name, before[name]))

        if args.check:
            return 0

        for path in paths:
            print(f"defragmenting {path.name}...", flush=True)
            subprocess.run(["e4defrag", "-v", os.fspath(path)], check=True)
            if path.stat().st_size != before[path.name][0]:
                raise RuntimeError(f"size changed while defragmenting {path}")

        after = {path.name: metrics(path) for path in paths}
        for name in FILES:
            print("after  " + describe(name, after[name]))
            before_extents = before[name][2]
            after_extents = after[name][2]
            reduction = 100.0 * (before_extents - after_extents) / before_extents
            print(f"result {name}: extent_reduction={reduction:.2f}%")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
