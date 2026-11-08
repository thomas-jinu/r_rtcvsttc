#!/usr/bin/env python3
"""Report run folders whose runFor*evolve.cout* file lacks 'Turning off'.

Usage:
    python check_turning_off.py /path/to/parent_folder
"""

from __future__ import annotations

import argparse
from pathlib import Path


def contains_turning_off(path: Path) -> bool:
    """Search incrementally, so large cout files are not loaded into memory."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return any("Turning off" in line for line in handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Folder to search recursively")
    args = parser.parse_args()
    root = args.folder.expanduser().resolve()

    if not root.is_dir():
        parser.error(f"Not a folder: {root}")

    matched = sorted(path for path in root.rglob("runFor*evolve.cout*") if path.is_file())
    if not matched:
        print(f"No files matching runFor*evolve.cout* found under {root}")
        return 0

    incomplete = 0
    for cout_file in matched:
        if not contains_turning_off(cout_file):
            incomplete += 1
            print(f"{cout_file.parent} — 'Turning off' was NOT found (run is not finished).")

    print(f"\nChecked {len(matched)} file(s); {incomplete} run(s) not finished.")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
