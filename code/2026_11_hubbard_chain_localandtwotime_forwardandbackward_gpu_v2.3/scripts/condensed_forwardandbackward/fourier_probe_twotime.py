#!/usr/bin/env python3
"""Gaussian-probe transform of a two-time lesser correlator (Freericks & Kemper, Eq. 27).

This operates on the CSV produced by my_hubbard_chain_twotime_G<_parallel_process.py.
It integrates on the actual time grids with trapezoidal weights, independently
for each site pair. It does not perform a spatial transform or gauge correction.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

REQUIRED_COLUMNS = {"t_prime", "t", "center_site", "site_index", "real_part", "imaginary_part"}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Read the command line and check the ranges."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_csv", type=Path)
    p.add_argument("output_csv", type=Path)
    p.add_argument("--sigma", type=float, required=True, help="Gaussian probe width in time units")
    p.add_argument("--omega-min", type=float, required=True)
    p.add_argument("--omega-max", type=float, required=True)
    p.add_argument("--omega-count", type=int, required=True)
    p.add_argument("--probe-min", type=float, required=True, help="First probe center time")
    p.add_argument("--probe-max", type=float, required=True, help="Last probe center time")
    p.add_argument("--probe-count", type=int, required=True)
    p.add_argument(
        "--input-convention", choices=("correlator", "green"), required=True,
        help="correlator: input is the raw measured correlator; green: input includes the conventional i factor",
    )
    p.add_argument(
        "--frequency-sign", choices=("lesser", "greater"), required=True,
        help="lesser: exp(+i omega (t-t_prime)); greater: exp(-i omega (t-t_prime)). Required, so the sign is always chosen explicitly.",
    )
    a = p.parse_args()
    if a.sigma <= 0 or a.omega_count < 1 or a.probe_count < 1:
        p.error("sigma and sample counts must be positive")
    if a.omega_min > a.omega_max or a.probe_min > a.probe_max:
        p.error("minimum must not exceed maximum")
    return a


# ---------------------------------------------------------------------------
# Reading the grid
# ---------------------------------------------------------------------------

def weights(x):
    """Trapezoidal integration weights for an increasing time axis."""
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Each time axis must contain at least two distinct increasing times")
    result = np.empty_like(x)
    result[0] = (x[1] - x[0]) / 2
    result[-1] = (x[-1] - x[-2]) / 2
    result[1:-1] = (x[2:] - x[:-2]) / 2
    return result


def read_grids(input_csv: Path) -> dict:
    """{(center_site, site_index): {(t, t_prime): value}} from the two-time CSV."""
    groups = {}
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}")
        for row in reader:
            pair = (int(row["center_site"]), int(row["site_index"]))
            key = (float(row["t"]), float(row["t_prime"]))
            value = complex(float(row["real_part"]), float(row["imaginary_part"]))
            if not all(np.isfinite(v) for v in (*key, value.real, value.imag)):
                raise ValueError("Non-finite time or correlator value")
            data = groups.setdefault(pair, {})
            if key in data:
                raise ValueError(f"Duplicate time pair {key} for sites {pair}")
            data[key] = value
    if not groups:
        raise ValueError("Input CSV is empty")
    return groups


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def transform_pair(pair, data, probes, omega, sign, sigma, input_convention):
    """Yield one CSV row per (probe, omega) for one site pair."""
    t = np.array(sorted({key[0] for key in data}))
    tp = np.array(sorted({key[1] for key in data}))
    wt, wp = weights(t), weights(tp)
    if len(data) != len(t) * len(tp):
        raise ValueError(
            f"Incomplete rectangular time grid for sites {pair}: {len(data)} values, "
            f"expected {len(t) * len(tp)}; no missing values are assumed or interpolated"
        )
    matrix = np.array([[data[(x, y)] for y in tp] for x in t])
    if input_convention == "green":
        matrix *= -1j

    for t0 in probes:
        left = wt * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        right = wp * np.exp(-0.5 * ((tp - t0) / sigma) ** 2)
        for w in omega:
            value = (left * np.exp(sign * 1j * w * t)) @ matrix @ (right * np.exp(-sign * 1j * w * tp))
            yield (*pair, f"{t0:.15g}", f"{w:.15g}", f"{value.real:.15g}", f"{value.imag:.15g}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Read the two-time CSV, transform it, and write the probe CSV."""
    a = parse_arguments()
    groups = read_grids(a.input_csv)

    # The frequency kernel sign is chosen separately from the input prefactor convention.
    sign = 1 if a.frequency_sign == "lesser" else -1
    omega = np.linspace(a.omega_min, a.omega_max, a.omega_count)
    probes = np.linspace(a.probe_min, a.probe_max, a.probe_count)

    with a.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("center_site", "site_index", "probe_time", "omega", "real_part", "imaginary_part"))
        for pair, data in sorted(groups.items()):
            for row in transform_pair(pair, data, probes, omega, sign, a.sigma, a.input_convention):
                writer.writerow(row)
    print(f"Wrote {a.output_csv}; finite time range and probe truncation affect the result.")


if __name__ == "__main__":
    main()
