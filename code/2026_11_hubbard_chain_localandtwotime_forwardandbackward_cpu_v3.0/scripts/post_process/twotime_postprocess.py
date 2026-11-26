"""Shared CPU/GPU v2.4 G< and G> extraction and equal-time diagnostics.

Equal-time preparation differences are reported, not fatal. Retain the last
forward equal-time measurement explicitly; never average preparation sweeps.
Malformed/incomplete grids and conflicting non-equal-time duplicates stay fatal.
"""
import argparse
import csv
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

COLLECTION_MARKER = "FiniteLoops printing ends"  # End of input-table printing, NOT preparation.
OVERLAP_TOLERANCE = 1e-7  # Warning threshold at equal time; strict elsewhere.
TIME_TOLERANCE = 1e-9
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
METRICS = ("forward_spread", "backward_spread", "forward_backward_difference")


def operator_pattern(label):
    return re.compile(rf"^\s*(\d+)\s+\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)\s+({NUMBER})\s+{re.escape(label)}(?:\s|$)")


def same_time(a, b):
    return abs(a - b) <= TIME_TOLERANCE


def read_pump_file(path):
    with Path(path).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    times, pump = [float(r["time"]) for r in rows], [float(r["pump"]) for r in rows]
    if len(times) < 3 or not all(math.isfinite(v) for v in times + pump):
        raise ValueError("Pump table requires finite physical times and both padding entries")
    if any(b - a <= 2 * TIME_TOLERANCE for a, b in zip(times, times[1:])):
        raise ValueError("Pump times must be distinct and strictly increasing")
    return times, pump


def find_single_output(folder, suffix, label):
    matches = sorted(Path(folder).glob(f"runFor*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {label} output in {folder}, found {len(matches)}")
    return matches[0]


def read_samples(path, label):
    """Return all complex samples and the advertised site count if available."""
    pattern = operator_pattern(label)
    collecting = finished = False
    samples = {}
    sites = None
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            match = re.search(r"TotalNumberOfSites\s*=\s*(\d+)", line)
            if match:
                sites = int(match[1])
            if COLLECTION_MARKER in line:
                if collecting:
                    raise ValueError(f"Concatenated finite-loop logs in {path}")
                collecting = True
                continue
            finished |= "Turning off the engine." in line
            if not collecting or label not in line:
                continue
            match = pattern.match(line)
            if match is None:
                raise ValueError(f"Malformed {label} measurement in {path}:{line_number}")
            site, real, imag, time = int(match[1]), float(match[2]), float(match[3]), float(match[4])
            if not all(math.isfinite(x) for x in (real, imag, time)):
                raise ValueError(f"Nonfinite measurement in {path}:{line_number}")
            samples.setdefault((site, time), []).append(complex(real, imag))
    if not samples or not finished:
        raise ValueError(f"No measurements or unfinished DMRG output: {path}")
    return samples, sites


def read_operator_output(path, operator_label, equal_time=None):
    """Keep the last sample; default equal time is the first measured restart time."""
    samples, _ = read_samples(path, operator_label)
    if equal_time is None:
        equal_time = next(iter(samples))[1]
    rows = []
    for (site, time), values in samples.items():
        if equal_time is None or not same_time(time, equal_time):
            if any(abs(a - b) > OVERLAP_TOLERANCE for a, b in zip(values, values[1:])):
                raise ValueError(f"Repeated non-equal-time value differs in {path}: site {site}, time {time}")
        rows.append((site, values[-1].real, values[-1].imag, time))
    return rows


def spread(values):
    return max((abs(a - b) for a in values for b in values), default=0.0)


def canonical_samples(path, label, allowed_times, t_prime):
    raw, sites = read_samples(path, label)
    samples = {}
    for (site, time), values in raw.items():
        nearest = min(allowed_times, key=lambda t: abs(t - time))
        if not same_time(time, nearest):
            raise ValueError(f"Unexpected/padding time {time} in {path}")
        samples.setdefault((site, nearest), []).extend(values)
    for (site, time), values in samples.items():
        if time != t_prime and any(abs(a - b) > OVERLAP_TOLERANCE for a, b in zip(values, values[1:])):
            raise ValueError(f"Repeated non-equal-time value differs in {path}: site {site}, time {time}")
    return samples, sites


def collect_step(task):
    folder, step, time_axis, center_site, label, expected_sites = task
    t_prime = time_axis[step + 1]
    physical = time_axis[1:-1]
    forward_path = find_single_output(folder, "_evolve_forward.cout", "forward evolve")
    backward_path = find_single_output(folder, "_evolve_backward.cout", "backward evolve")
    forward, fn = canonical_samples(forward_path, label, physical[step:], t_prime)
    backward, bn = canonical_samples(backward_path, label, physical[:step + 1], t_prime)
    advertised = [n for n in (fn, bn, expected_sites) if n is not None]
    if advertised and len(set(advertised)) != 1:
        raise ValueError(f"Site counts disagree in {folder}: {advertised}")
    # Standard DMRG++ logs embed the input as base64, not a plain site-count line.
    # The parent validates a common count from all embedded inputs (or --sites).
    sites = advertised[0] if advertised else max(s for s, _ in forward.keys() | backward.keys()) + 1
    if sites < 1 or not 0 <= center_site < sites:
        raise ValueError(f"Invalid site count/center in {folder}")
    for direction, samples, times in (("forward", forward, physical[step:]), ("backward", backward, physical[:step+1])):
        expected = {(site, t) for site in range(sites) for t in times}
        if set(samples) != expected:
            missing, extra = sorted(expected - samples.keys()), sorted(samples.keys() - expected)
            raise ValueError(f"Incomplete {direction} grid in {folder}: missing={missing[:8]}, extra={extra[:8]}")
    data = {(t_prime, t, center_site, site): values[-1] for (site, t), values in backward.items()}
    # Forward explicitly owns equal time; overwrite backward only at that slice.
    for (site, t), values in forward.items():
        key = (t_prime, t, center_site, site)
        if key in data and t != t_prime:
            raise ValueError(f"Unexpected directional overlap in {folder}: site={site}, time={t}")
        data[key] = values[-1]
    diagnostics = []
    for site in range(sites):
        fv, bv = forward[(site, t_prime)], backward[(site, t_prime)]
        diagnostics.append((t_prime, site, spread(fv), spread(bv), abs(fv[-1] - bv[-1])))
    return data, diagnostics


def embedded_site_count(path):
    """Read the small input echo at the top of a standard solver log."""
    import base64
    with Path(path).open(encoding="utf-8") as file:
        for _ in range(20):
            line = file.readline()
            if not line:
                break
            if line.startswith("PsiApp::echoBase64:"):
                text = base64.b64decode(file.readline().strip(), validate=True).decode("utf-8")
                match = re.search(r"TotalNumberOfSites\s*=\s*(\d+)", text)
                if not match:
                    raise ValueError(f"Embedded input lacks TotalNumberOfSites in {path}")
                return int(match[1])
    return None


def save_twotime_csv(data, path):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t_prime", "t", "center_site", "site_index", "real_part", "imaginary_part"])
        for (tp, t, center, site), z in sorted(data.items()):
            writer.writerow([f"{tp:.15g}", f"{t:.15g}", center, site, f"{z.real:.15g}", f"{z.imag:.15g}"])


def save_diagnostics(rows, path):
    rows = sorted(rows)
    if not rows:
        raise ValueError("No equal-time diagnostics to summarize")
    lines = ["EQUAL-TIME DIAGNOSTICS SUMMARY", "Values measure numerical disagreement, not exact-error bounds.",
             "Physical CSV equal-time convention: last forward measurement; no sweep averaging.",
             f"Equal-time warning threshold: {OVERLAP_TOLERANCE:.8e} (processing continues).",
             "Extrema cover all t_prime and sites in this run; ties show the first sorted location.",
             "Quantity                         Minimum         t_prime Site  Maximum         t_prime Site  Min_ties Max_ties"]
    for column, name in enumerate(METRICS, 2):
        lo, hi = min(rows, key=lambda r: r[column]), max(rows, key=lambda r: r[column])
        min_ties = sum(r[column] == lo[column] for r in rows)
        max_ties = sum(r[column] == hi[column] for r in rows)
        lines.append(f"{name:<32} {lo[column]:.8e} {lo[0]:>9.9g} {lo[1]:>4}  {hi[column]:.8e} {hi[0]:>9.9g} {hi[1]:>4}  {min_ties:>8} {max_ties:>8}")
    count = sum(any(v > OVERLAP_TOLERANCE for v in row[2:]) for row in rows)
    lines.extend([f"Site/time rows exceeding warning threshold: {count}/{len(rows)}", "", "ALL EQUAL-TIME DIAGNOSTICS"])
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        file.write("\n".join(lines) + "\n")
        writer = csv.writer(file)
        writer.writerow(["t_prime", "site_index", *METRICS])
        for tp, site, *values in rows:
            writer.writerow([f"{tp:.15g}", site, *(f"{v:.8e}" for v in values)])
    if count:
        print(f"WARNING: equal-time disagreement exceeds {OVERLAP_TOLERANCE:g} in {count}/{len(rows)} site/time rows; processing continued. See {path}")


def process_twotime_results(base_directory, run_name, number_of_steps, time_axis, center_site, workers, operator_label, filename_token, sites=None):
    base_directory = Path(base_directory)
    folder = base_directory / run_name
    if number_of_steps != len(time_axis) - 2 or number_of_steps < 1:
        raise ValueError("number_of_steps must equal the positive physical pump-time count")
    if workers < 1 or not all(math.isfinite(t) for t in time_axis) or any(b-a <= 2*TIME_TOLERANCE for a,b in zip(time_axis,time_axis[1:])):
        raise ValueError("Invalid worker count or time axis")
    tasks = []
    counts = set()
    for step in range(number_of_steps):
        work = folder / f"step_{step:04d}"
        for suffix, direction in (("_evolve_forward.cout", "forward"), ("_evolve_backward.cout", "backward")):
            path = find_single_output(work, suffix, direction)
            count = embedded_site_count(path)
            if count is not None:
                counts.add(count)
        tasks.append((work, step, time_axis, center_site, operator_label, sites))
    if sites is not None:
        counts.add(sites)
    if len(counts) != 1:
        raise ValueError("Missing or inconsistent embedded site counts; provide --sites for legacy logs without an input echo")
    count = counts.pop()
    tasks = [(*task[:-1], count) for task in tasks]
    data, diagnostics = {}, []
    if workers == 1:
        results = map(collect_step, tasks)
        for step_data, step_diagnostics in results:
            data.update(step_data)
            diagnostics.extend(step_diagnostics)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for step_data, step_diagnostics in executor.map(collect_step, tasks):
                data.update(step_data)
                diagnostics.extend(step_diagnostics)
    processed = base_directory / "processed_data"
    processed.mkdir(parents=True, exist_ok=True)
    csv_path = processed / f"{run_name}_{filename_token}_centersite={center_site}.csv"
    report = csv_path.with_name(csv_path.stem + "_equal_time_diagnostics.txt")
    save_twotime_csv(data, csv_path)
    save_diagnostics(diagnostics, report)
    print(f"Wrote equal-time diagnostics: {report}")
    return csv_path


def main(label, filename_token):
    parser = argparse.ArgumentParser(description="Post-process a two-time run and report equal-time numerical disagreement.")
    parser.add_argument("center_site", type=int)
    parser.add_argument("gs_filename", type=Path)
    parser.add_argument("pump_file", type=Path)
    parser.add_argument("process_folder", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--sites", type=int, help="Site count for legacy logs without embedded inputs")
    args = parser.parse_args()
    if args.workers < 1 or (args.sites is not None and args.sites < 1):
        parser.error("workers and sites must be positive")
    times, _ = read_pump_file(args.pump_file)
    path = process_twotime_results(args.process_folder.resolve().parent, args.process_folder.name,
        len(times)-2, times, args.center_site, args.workers, label, filename_token, args.sites)
    print(f"Wrote post-processed data: {path}")
    return 0
