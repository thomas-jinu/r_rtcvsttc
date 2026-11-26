#!/usr/bin/env python3
"""Check a pair of forward and backward DMRG++ two-time output logs.

Run this optional, standalone tool after a calculation to inspect its logs:

    python3 audit_twotime_pair.py FORWARD_LOG BACKWARD_LOG

Reports default to processed_data/ beside the ground-state file for the standard
G branch/step directory layout. Use --output PATH to override the destination.

It reads the inputs embedded in both logs and checks that the runs agree on
site count and restart time, have the expected site/time measurements, follow
consistent time directions, and finished. The last time-table entry is treated
as padding. It reports missing, extra, and unexpected duplicate measurements,
repeated restart-time samples, and differences between forward and backward
values at the restart time. Use --label to select the measured operator and
--tolerance to flag numerical differences (the default is 1e-7).

The JSON report is diagnostic: it does not alter the logs, choose a value for
postprocessing, or establish physical accuracy. Structural errors produce a
nonzero exit status; numerical differences alone do not. Requires only the
Python standard library.
"""
import argparse
import base64
import json
import math
import re
from collections import defaultdict
from pathlib import Path

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def encode_sample(sample):
    loop, line, z = sample
    return {"loop": loop, "line": line, "real": z.real, "imag": z.imag}


def read_output(path, label):
    pattern = re.compile(
        rf"^\s*(\d+)\s+\(({NUMBER}),\s*({NUMBER})\)\s+({NUMBER})\s+{re.escape(label)}(?:\s|$)"
    )
    samples = defaultdict(list)
    loop = None
    collecting = False
    embedded = None
    expect_input = False
    finished = False
    truncations = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if expect_input:
                if embedded is not None:
                    raise ValueError("Multiple embedded inputs; concatenated logs are unsupported")
                embedded = base64.b64decode(line.strip(), validate=True).decode("utf-8")
                expect_input = False
            if line.startswith("PsiApp::echoBase64:"):
                expect_input = True
            if "FiniteLoops printing ends" in line:
                if collecting:
                    raise ValueError("Multiple finite-loop sections; concatenated logs are unsupported")
                collecting = True
            match = re.search(r"Finite loop number (\d+)", line)
            if match:
                loop = int(match[1])
            finished |= "Turning off the engine." in line
            match = re.search(r"Discarded weight \(Truncation error\): (\S+)", line)
            if match:
                truncations.append(float(match[1]))
            match = pattern.match(line) if collecting else None
            if match:
                if loop is None:
                    raise ValueError("Measurement without finite-loop context")
                site = int(match[1])
                z = complex(float(match[2]), float(match[3]))
                time = float(match[4])
                if not all(math.isfinite(x) for x in (z.real, z.imag, time)):
                    raise ValueError(f"Nonfinite measurement at line {line_number}")
                # Match input-grid times despite harmless floating-point print noise.
                samples[(site, round(time, 9))].append((loop, line_number, z))
    if embedded is None or not samples:
        raise ValueError(f"Missing embedded input or {label} measurements: {path}")
    table = re.search(r"matrix\s+AversusTime\s*=\s*\[(.*?)\];", embedded, re.S)
    sites = re.search(r"TotalNumberOfSites\s*=\s*(\d+)", embedded)
    restart = re.search(r'RestartFilename\s*=\s*"([^"]+)"', embedded)
    if not table or not sites or not restart:
        raise ValueError("Missing time table, site count, or restart filename")
    times = [round(float(m[0]), 9) for m in re.findall(
        rf"\[\s*({NUMBER})\s*,\s*({NUMBER})\s*\]", table[1])]
    if len(times) < 2 or len(set(times)) != len(times):
        raise ValueError("Time table must have distinct physical and padding entries")
    evolutions = []
    for vector in ("P2", "P3"):
        match = re.search(rf'string\s+{vector}\s*=\s*"TimeEvolve\{{([^}}]+)\}}', embedded)
        if not match:
            raise ValueError(f"Missing {vector} evolution")
        params = dict(item.strip().split("=", 1) for item in match[1].split(","))
        evolutions.append({"tau": float(params["tau"]), "start_loop": int(params["startLoop"])})
    if evolutions[0] != evolutions[1]:
        raise ValueError("P2 and P3 have different evolution parameters")
    meta = {"path": str(Path(path).resolve()), "sites": int(sites[1]),
            "restart": restart[1], "table_times": times,
            "embedded_input": embedded, "evolution": evolutions[0],
            "engine_finished": finished,
            "max_discarded_weight": max(truncations, default=None)}
    return meta, samples


def audit(forward, backward, label="<P2|c'|P3>", tolerance=1e-7):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")
    fm, fs = read_output(forward, label)
    bm, bs = read_output(backward, label)
    if fm["sites"] != bm["sites"] or fm["restart"] != bm["restart"]:
        raise ValueError("Outputs use different site counts or restart filenames")
    start = fm["table_times"][0]
    if start != bm["table_times"][0]:
        raise ValueError("Different restart times")
    errors = []
    directional = {}
    for direction, meta, samples in (("forward", fm, fs), ("backward", bm, bs)):
        times = meta["table_times"]
        sign = 1 if direction == "forward" else -1
        if sign * meta["evolution"]["tau"] <= 0 or any(
            sign * (b - a) <= 0 for a, b in zip(times, times[1:])
        ):
            errors.append(f"{direction}: inconsistent table order/tau")
        expected = {(site, t) for site in range(meta["sites"]) for t in times[:-1]}
        actual = set(samples)
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        duplicates = {key: values for key, values in samples.items() if len(values) > 1}
        unexpected = sorted(key for key in duplicates if key[1] != start)
        if missing or extra or unexpected or not meta["engine_finished"]:
            errors.append(f"{direction}: missing/extra keys, unexpected duplicates, or unfinished engine")
        spreads = []
        for (site, t), values in sorted(duplicates.items()):
            max_pair = max(abs(a[2] - b[2]) for a in values for b in values)
            max_adjacent = max(abs(a[2] - b[2]) for a, b in zip(values, values[1:]))
            spreads.append({"site": site, "time": t, "count": len(values),
                            "max_pairwise_difference": max_pair,
                            "max_consecutive_difference": max_adjacent,
                            "exceeds_diagnostic_tolerance": max_pair > tolerance,
                            "samples": [encode_sample(v) for v in values]})
        directional[direction] = {**meta, "physical_times": times[:-1],
            "padding_time": times[-1], "unique_measurements": len(samples),
            "missing_keys": missing, "extra_keys": extra,
            "unexpected_duplicate_keys": unexpected, "repeated_measurements": spreads}
    overlap = sorted(set(fs) & set(bs))
    expected_overlap = {(s, start) for s in range(fm["sites"])}
    if set(overlap) != expected_overlap:
        errors.append("Forward/backward overlap is not exactly the restart-time site set")
    comparisons = []
    for site, t in overlap:
        a, b = fs[(site, t)], bs[(site, t)]
        comparisons.append({"site": site, "time": t,
            "first_forward": encode_sample(a[0]), "first_backward": encode_sample(b[0]),
            "last_forward": encode_sample(a[-1]), "last_backward": encode_sample(b[-1]),
            "first_difference": abs(a[0][2] - b[0][2]),
            "last_difference": abs(a[-1][2] - b[-1][2])})
    return {"restart_time": start, "label": label, "diagnostic_tolerance": tolerance,
        "padding_convention": "One final padding entry per embedded directional time table",
        "accuracy_validated": False, "structural_errors": errors,
        "directions": directional, "equal_time_comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward", type=Path)
    parser.add_argument("backward", type=Path)
    parser.add_argument("--label", default="<P2|c'|P3>")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output", type=Path,
                        help="Override the default processed_data audit report path")
    args = parser.parse_args()
    report = audit(args.forward, args.backward, args.label, args.tolerance)
    if args.output is None:
        forward = args.forward.resolve()
        branch = next((p for p in forward.parents
                       if p.name.startswith(("G<_twotime_", "G>_twotime_"))), None)
        if branch is None:
            parser.error("Cannot identify a G branch directory; specify --output PATH")
        backward = args.backward.resolve()
        if branch not in backward.parents or forward.parent != backward.parent:
            parser.error("Default output requires logs from the same branch and step")
        args.output = (branch.parent / "processed_data" /
                       f"{branch.name}_{forward.parent.name}_audit.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Do not accidentally overwrite raw outputs or an existing diagnostic.
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Audit written: {args.output}")
    print(f"Restart time: {report['restart_time']}; structural errors: {len(report['structural_errors'])}")
    for name, d in report["directions"].items():
        print(f"{name}: {len(d['physical_times'])} physical times, padding={d['padding_time']}")
    values = report["equal_time_comparisons"]
    print(f"Max first/last equal-time directional differences: "
          f"{max((v['first_difference'] for v in values), default=0):.9g} / "
          f"{max((v['last_difference'] for v in values), default=0):.9g}")
    print("No equal-time value selected; physical accuracy remains unvalidated.")
    return 1 if report["structural_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
