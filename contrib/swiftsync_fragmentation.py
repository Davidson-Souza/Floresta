#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Summarize filesystem extent fragmentation for a live SwiftSync database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

GIB = 1 << 30
MIB = 1 << 20
DATABASE_FILES = ("body", "blobs")


def florestad_pids() -> list[int]:
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0", 1)[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if Path(os.fsdecode(command)).name == "florestad":
            pids.append(int(entry.name))
    return sorted(pids)


def database_directory(pid: int) -> Path:
    maps_path = Path(f"/proc/{pid}/maps")
    try:
        mappings = maps_path.read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
        raise RuntimeError(f"cannot read {maps_path}: {error}") from error

    directories = {
        Path(line.split()[-1]).parent
        for line in mappings.splitlines()
        if ".swiftsync-utxos-" in line
        and line.split()[-1].endswith(("/body", "/blobs"))
    }
    if len(directories) != 1:
        raise RuntimeError(
            f"expected one mapped SwiftSync database for pid {pid}, found {len(directories)}"
        )
    return directories.pop()


def extent_count(path: Path) -> int:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    result = subprocess.run(
        ["filefrag", "-v", os.fspath(path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    match = re.search(r":\s+(\d+)\s+extents? found\s*$", result.stdout, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not parse filefrag output for {path}")
    return int(match.group(1))


def print_summary(directory: Path) -> None:
    print(f"database={directory}")
    print(
        "file\tsize_gib\tallocated_gib\textents\textents_per_gib\taverage_extent_mib"
    )
    for name in DATABASE_FILES:
        path = directory / name
        stat = path.stat()
        extents = extent_count(path)
        size_gib = stat.st_size / GIB
        allocated_gib = stat.st_blocks * 512 / GIB
        extents_per_gib = extents / size_gib if size_gib else 0.0
        average_extent_mib = stat.st_size / extents / MIB if extents else 0.0
        print(
            f"{name}\t{size_gib:.3f}\t{allocated_gib:.3f}\t{extents}"
            f"\t{extents_per_gib:.1f}\t{average_extent_mib:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pid", type=int, help="florestad process ID")
    source.add_argument("--directory", type=Path, help="SwiftSync database directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.directory is not None:
            directory = args.directory
        else:
            pid = args.pid
            if pid is None:
                pids = florestad_pids()
                if len(pids) != 1:
                    raise RuntimeError(
                        f"expected one running florestad, found {len(pids)}; pass --pid"
                    )
                pid = pids[0]
            directory = database_directory(pid)
        print_summary(directory)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
