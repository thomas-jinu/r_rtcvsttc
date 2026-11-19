"""Post-process a completed G< two-time Hubbard-chain calculation.

Reads the forward and backward DMRG++ outputs of every step and writes one CSV with
the correlator <P2|c'|P3> on the (t', t) grid. This file and the G> processor are kept
separate on purpose; the only difference is the operator label.
"""

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Operator measured by the G< runs (the input builder uses c for P1 and c' here).
TWOTIME_OPERATOR_LABEL = "<P2|c'|P3>"

# DMRG++ prints this line once the finite loops have finished.
TWOTIME_COLLECTION_MARKER = "FiniteLoops printing ends"

# Repeated (time, site) values must agree within this tolerance, otherwise the run is rejected.
OVERLAP_TOLERANCE = 1e-7

TWOTIME_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def operator_pattern(operator_label: str) -> re.Pattern[str]:
    """Regular expression for one output line: site, (re, im), time, operator label."""
    return re.compile(
        rf"^\s*(\d+)\s+"
        rf"\(\s*({TWOTIME_NUMBER_PATTERN})\s*,\s*({TWOTIME_NUMBER_PATTERN})\s*\)\s+"
        rf"({TWOTIME_NUMBER_PATTERN})\s+"
        rf"{re.escape(operator_label)}"
    )


# ---------------------------------------------------------------------------
# Command line and file lookup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(
        description="Post-process one two-time G< run into a CSV file.",
    )
    parser.add_argument("center_site", type=int, help="Center site index")
    parser.add_argument("gs_filename", type=Path, help="Ground-state file location")
    parser.add_argument("pump_file", type=Path, help="Path to the pump file")
    parser.add_argument(
        "process_folder",
        type=Path,
        help="Only process an existing twotime_RUN_NAME directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of output files to parse concurrently (default: CPU count).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def read_pump_file(pump_file_path: Path) -> tuple[list[float], list[float]]:
    """Read the time and pump columns of the pump table."""
    time = []
    pump = []
    with open(pump_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            time.append(float(row["time"]))
            pump.append(float(row["pump"]))
    return time, pump


def find_single_output(step_folder: Path, suffix: str, label: str) -> Path:
    """Return the one DMRG++ output (runFor<input>.cout) in step_folder that ends with suffix."""
    matches = sorted(step_folder.glob(f"runFor*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {label} output in {step_folder}, found {len(matches)}")
    return matches[0]


def find_evolve_output_forward(step_folder: Path) -> Path:
    """The forward-evolve output of one step."""
    return find_single_output(step_folder, "_evolve_forward.cout", "forward evolve")


def find_evolve_output_backward(step_folder: Path) -> Path:
    """The backward-evolve output of one step."""
    return find_single_output(step_folder, "_evolve_backward.cout", "backward evolve")


# ---------------------------------------------------------------------------
# Reading the outputs
# ---------------------------------------------------------------------------

def read_operator_output(
    output_path: Path,
    operator_label: str = TWOTIME_OPERATOR_LABEL,
) -> list[tuple[int, float, float, float]]:
    """Rows (site, real, imag, time) printed after the finite-loop marker.

    A (site, time) pair printed more than once must have the same value; otherwise it fails.
    """
    rows_by_key = {}
    collecting = False
    pattern = operator_pattern(operator_label)

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if TWOTIME_COLLECTION_MARKER in line:
                collecting = True
                continue
            if not collecting or operator_label not in line:
                continue

            match = pattern.search(line)
            if match is None:
                continue

            site_index = int(match.group(1))
            real_part = float(match.group(2))
            imaginary_part = float(match.group(3))
            time = float(match.group(4))

            key = (site_index, time)
            if key in rows_by_key:
                previous = complex(rows_by_key[key][1], rows_by_key[key][2])
                current = complex(real_part, imaginary_part)
                if abs(previous - current) > OVERLAP_TOLERANCE:
                    raise ValueError(
                        f"Repeated value differs in {output_path}: site {site_index}, "
                        f"time {time}: {previous} vs {current}"
                    )
            # Keep the last value; the check above guarantees it equals the earlier ones.
            rows_by_key[key] = [site_index, real_part, imaginary_part, time]

    return [
        (site, real_part, imaginary_part, time)
        for site, real_part, imaginary_part, time in rows_by_key.values()
    ]


# ---------------------------------------------------------------------------
# Collecting the steps
# ---------------------------------------------------------------------------

def collect_step_data(
    step: int,
    twotime_folder: Path,
    time_axis: list[float],
    center_site: int,
) -> dict[tuple[float, float, int, int], complex]:
    """Forward and backward values of one physical step, keyed by (t', t, center, site)."""
    step_folder = twotime_folder / f"step_{step:04d}"
    t_prime = float(time_axis[step + 1])
    data = {}

    # Forward and backward outputs can share the equal-time point; their values must agree.
    for output_path in (
        find_evolve_output_forward(step_folder),
        find_evolve_output_backward(step_folder),
    ):
        for site, real_part, imaginary_part, time in read_operator_output(output_path):
            key = (t_prime, float(time), int(center_site), int(site))
            value = complex(real_part, imaginary_part)
            if key in data and abs(data[key] - value) > OVERLAP_TOLERANCE:
                raise ValueError(
                    f"Forward and backward values differ at t'={t_prime}, t={time}, "
                    f"site {site}: {data[key]} vs {value}"
                )
            data[key] = value

    return data


def collect_twotime_data(
    twotime_folder: Path,
    number_of_steps: int,
    time_axis: list[float],
    center_site: int,
    workers: int,
) -> dict[tuple[float, float, int, int], complex]:
    """All steps, read concurrently and merged into one dictionary."""
    data = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        step_data_sets = executor.map(
            collect_step_data,
            range(number_of_steps),
            [twotime_folder] * number_of_steps,
            [time_axis] * number_of_steps,
            [center_site] * number_of_steps,
        )
        for step_data in step_data_sets:
            data.update(step_data)
    return data


# ---------------------------------------------------------------------------
# Writing the CSV
# ---------------------------------------------------------------------------

def save_twotime_csv(
    data: dict[tuple[float, float, int, int], complex],
    output_path: Path,
) -> None:
    """Write the (t', t, center, site) grid as one CSV, sorted by key."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["t_prime", "t", "center_site", "site_index", "real_part", "imaginary_part"])
        for (t_prime, time, center_site, site), value in sorted(data.items()):
            writer.writerow([
                f"{t_prime:.15g}",
                f"{time:.15g}",
                center_site,
                site,
                f"{value.real:.15g}",
                f"{value.imag:.15g}",
            ])


def process_twotime_results(
    base_directory: Path,
    run_name: str,
    number_of_steps: int,
    time_axis,
    center_site: int,
    workers: int,
) -> Path:
    """Collect one run folder and write its CSV into processed_data/."""
    twotime_folder = base_directory / f"{run_name}"

    if not twotime_folder.is_dir():
        raise FileNotFoundError(f"Two-time run folder not found: {twotime_folder}")
    if number_of_steps != len(time_axis) - 2:
        raise ValueError("number_of_steps must equal the physical pump-time count")

    missing_step_folders = [
        f"step_{step:04d}"
        for step in range(number_of_steps)
        if not (twotime_folder / f"step_{step:04d}").is_dir()
    ]
    if missing_step_folders:
        raise FileNotFoundError("Missing step folders: " + ", ".join(missing_step_folders))

    data = collect_twotime_data(
        twotime_folder=twotime_folder,
        number_of_steps=number_of_steps,
        time_axis=time_axis,
        center_site=center_site,
        workers=workers,
    )

    processed = base_directory / "processed_data"
    processed.mkdir(parents=True, exist_ok=True)
    # G< measures <P2|c'|P3>, so the file is named P2_cdag_P3 (G> files are P2_c_P3).
    csv_path = processed / f"{run_name}_P2_cdag_P3_centersite={center_site}.csv"
    save_twotime_csv(data=data, output_path=csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Post-process one two-time run folder."""
    args = parse_args()

    restart_path = args.gs_filename.resolve()
    time_axis, _pump_axis = read_pump_file(args.pump_file)
    if len(time_axis) < 3:
        raise ValueError("Pump file must contain physical times and both paddings")

    csv_path = process_twotime_results(
        base_directory=restart_path.parent,
        run_name=args.process_folder.name,
        number_of_steps=len(time_axis) - 2,
        time_axis=time_axis,
        center_site=args.center_site,
        workers=args.workers,
    )
    print(f"Wrote post-processed data: {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
