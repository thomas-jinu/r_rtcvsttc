"""Post-process a TD (time-dependent) DMRG++ run into one CSV of operator values.

Usage:
  process one output:   <output.cout> <pump_table.csv>
  process a sweep:      --sweep-dir <folder> [--frequency-min F --frequency-max F]

Each output row is (t, site) and each operator gets a real and an imaginary column.
"""

import argparse
import csv
import json
import re
from pathlib import Path

# Operators printed by the TD run, as they appear in DMRG++ output.
TD_OPERATOR_LABELS = {
    "<P0|n|P0>",
    "<gs|n|gs>",
    "<P0|sz|P0>",
    "<gs|sz|gs>",
    "<P0|local_moment|P0>",
    "<gs|local_moment|gs>",
    "<P0|hole|P0>",
    "<gs|hole|gs>",
    "<P0|parity|P0>",
    "<gs|parity|gs>",
    "<P0|double|P0>",
    "<gs|double|gs>",
    "<P0|n*n|P0>",
    "<gs|n*n|gs>",
}

# One output line: site, (real, imag), time, operator label.
TD_OPERATOR_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<site>\d+)\s+
    \(
        (?P<real>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),
        (?P<imag>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \)\s+
    (?P<time>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+
    (?P<label><[^>]+>)
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Read the command line: one output file, or a sweep folder."""
    parser = argparse.ArgumentParser(
        description="Post-process TD DMRG++ outputs into operator CSV files.",
    )
    parser.add_argument("process_file", nargs="?", type=Path, help="Process an existing output file.")
    parser.add_argument("pump_file", nargs="?", type=Path, help="Path to the pump file.")
    parser.add_argument("--sweep-dir", type=Path, help="Parent directory containing TD frequency-run folders")
    parser.add_argument("--frequency-min", type=float, help="Inclusive lower frequency filter (requires --sweep-dir)")
    parser.add_argument("--frequency-max", type=float, help="Inclusive upper frequency filter (requires --sweep-dir)")
    args = parser.parse_args()

    if args.sweep_dir:
        if args.process_file or args.pump_file:
            parser.error("--sweep-dir cannot be combined with process_file or pump_file")
    elif not args.process_file or not args.pump_file:
        parser.error("supply process_file and pump_file, or --sweep-dir")
    elif args.frequency_min is not None or args.frequency_max is not None:
        parser.error("frequency filters require --sweep-dir")

    if args.frequency_min is not None and args.frequency_max is not None and args.frequency_min > args.frequency_max:
        parser.error("frequency-min must not exceed frequency-max")
    return args


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_pump_file(pump_file_path: Path) -> tuple[list[float], list[float]]:
    """Time and pump columns of a pump table."""
    time = []
    pump = []
    with open(pump_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            time.append(float(row["time"]))
            pump.append(float(row["pump"]))
    return time, pump


def read_operator_output(output_path: Path) -> dict[str, list[tuple[int, complex, float]]]:
    """Operator rows grouped by label, read from every matching line of the output."""
    operators = {label: [] for label in TD_OPERATOR_LABELS}
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            match = TD_OPERATOR_PATTERN.match(line)
            if match is None:
                continue
            label = match.group("label")
            if label not in operators:
                continue
            site = int(match.group("site"))
            value = complex(float(match.group("real")), float(match.group("imag")))
            time = float(match.group("time"))
            operators[label].append((site, value, time))
    return operators


def collect_td_data(output_path: Path) -> dict[str, dict[tuple[float, int], complex]]:
    """{label: {(time, site): value}}. A repeated (time, site) keeps its last value."""
    data = {label: {} for label in TD_OPERATOR_LABELS}
    for label, rows in read_operator_output(output_path).items():
        for site, value, time in rows:
            data[label][(float(time), int(site))] = value
    return data


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_td_csv(data: dict[str, dict[tuple[float, int], complex]], output_path: Path) -> None:
    """One wide CSV: a real and an imaginary column per operator, one row per (t, site)."""
    labels = sorted(data)
    all_keys = {key for operator_data in data.values() for key in operator_data}

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["t", "site_index"]
        for label in labels:
            header.extend([f"{label}_real", f"{label}_imag"])
        writer.writerow(header)

        for time, site in sorted(all_keys):
            row = [f"{time:.15g}", site]
            for label in labels:
                # Missing (time, site) entries are written as NaN.
                value = data[label].get((time, site), complex(float("nan")))
                row.extend([f"{value.real:.15g}", f"{value.imag:.15g}"])
            writer.writerow(row)


def process_td_results(output_path: Path, number_of_steps: int, time_axis) -> Path:
    """Read one TD output and write its CSV into the processed_data folder next to the run."""
    if not output_path.is_file():
        raise FileNotFoundError(f"Output file not found: {output_path}")
    if number_of_steps > len(time_axis):
        raise ValueError("number_of_steps cannot exceed the length of time_axis")

    data = collect_td_data(output_path)
    processed = output_path.parent.parent / "processed_data"
    processed.mkdir(parents=True, exist_ok=True)
    csv_path = processed / f"{output_path.parent.name}_{output_path.stem}_operators.csv"
    save_td_csv(data=data, output_path=csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def process_sweep(args: argparse.Namespace) -> int:
    """Process every TD folder in a sweep directory that falls inside the frequency range."""
    if not args.sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {args.sweep_dir}")

    matched = 0
    failed = 0
    for folder in sorted(args.sweep_dir.glob("td_N=*")):
        if not folder.is_dir():
            continue
        metadata = list(folder.glob("input_args_td_*.json"))
        if len(metadata) != 1:
            print(f"Skipping {folder}: expected one TD metadata file")
            continue
        try:
            with metadata[0].open(encoding="utf-8") as f:
                frequency = float(json.load(f)["pump_frequency"])
            if args.frequency_min is not None and frequency < args.frequency_min:
                continue
            if args.frequency_max is not None and frequency > args.frequency_max:
                continue
            matched += 1

            inputs = list(folder.glob("input_td_*.ain"))
            if len(inputs) != 1:
                raise ValueError("expected exactly one TD input file")
            output = folder / f"runFor{inputs[0].stem}.cout"
            time_axis, _ = read_pump_file(folder / "pump_table.csv")
            csv_path = process_td_results(output, len(time_axis), time_axis)

            destination = args.sweep_dir / "processed_data" / f"{folder.name}_operators.csv"
            csv_path.replace(destination)
            print(f"Frequency {frequency:g}: wrote {destination}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            failed += 1
            print(f"Failed to process {folder}: {error}")

    if matched == 0:
        raise ValueError("No TD runs matched the requested frequency range")
    if failed:
        print(f"Failed to process {failed} of {matched} matched TD runs")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Process one output file, or a whole sweep folder."""
    args = parse_args()

    if args.sweep_dir:
        return process_sweep(args)

    time_axis, _ = read_pump_file(args.pump_file)
    csv_path = process_td_results(args.process_file, len(time_axis), time_axis)
    print(f"Wrote post-processed data: {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
