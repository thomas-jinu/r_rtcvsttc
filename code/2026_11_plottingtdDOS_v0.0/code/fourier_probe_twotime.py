#!/usr/bin/env python3
"""Gaussian-probe transform of a two-time lesser correlator (Freericks & Kemper, Eq. 27).

This operates on the CSV produced by my_hubbard_chain_twotime_G<_parallel_process.py.
It integrates on the actual time grids with trapezoidal weights, independently
for each site pair. The input is the raw correlator, without a Green-function
prefactor. It does not perform a spatial transform or gauge correction.

For a uniform time grid with spacing 0.1, trapezoidal weights are 0.1 at
interior points and 0.05 at each endpoint. The weights are computed from
both input time grids; each grid must be evenly spaced.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def weights(x):
    """Return trapezoidal weights for a strictly increasing, uniform grid."""
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Each time axis must contain at least two distinct increasing times")
    spacing = np.diff(x)
    if not np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-12):
        raise ValueError("Each time axis must be evenly spaced")
    # Endpoints contribute half an interval; interior points contribute one interval.
    result = np.empty_like(x)
    result[0] = (x[1] - x[0]) / 2
    result[-1] = (x[-1] - x[-2]) / 2
    result[1:-1] = (x[2:] - x[:-2]) / 2
    return result


def parse_args():
    """Parse and validate the command-line options."""
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
    p.add_argument('--correlator', choices=('lesser', 'greater'), required=True,
                   help='correlator type: lesser uses exp(+i omega (t-t_prime)); greater uses exp(-i omega (t-t_prime))')
    a = p.parse_args()
    if a.sigma <= 0 or a.omega_count < 1 or a.probe_count < 1:
        p.error("sigma and sample counts must be positive")
    if a.omega_min > a.omega_max or a.probe_min > a.probe_max:
        p.error("minimum must not exceed maximum")
    return a


def read_correlators(path):
    """Read values grouped by site pair and then by (t, t_prime)."""
    groups = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"t_prime", "t", "center_site", "site_index", "real_part", "imaginary_part"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
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


def prepare_grid(pair, data):
    """Build a rectangular time grid and correlator matrix for one site pair."""
    # Matrix rows correspond to t; columns correspond to t_prime.
    t = np.array(sorted({key[0] for key in data}))
    tp = np.array(sorted({key[1] for key in data}))
    wt, wp = weights(t), weights(tp)
    if len(data) != len(t) * len(tp):
        raise ValueError(f"Incomplete rectangular time grid for sites {pair}: {len(data)} values, expected {len(t) * len(tp)}; no missing values are assumed or interpolated")
    matrix = np.array([[data[(x, y)] for y in tp] for x in t])
    return t, tp, wt, wp, matrix


def transform_grid(t, tp, wt, wp, matrix, probes, omega, sigma, sign):
    """Yield (probe time, frequency, transform) for one site pair."""
    for t0 in probes:
        # Apply the Gaussian envelope and quadrature weights on both axes.
        left = wt * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        right = wp * np.exp(-0.5 * ((tp - t0) / sigma) ** 2)
        for w in omega:
            # Factor the frequency kernel across the two time axes; matrix
            # multiplication then performs the weighted double integral.
            value = (left * np.exp(sign * 1j * w * t)) @ matrix @ (right * np.exp(-sign * 1j * w * tp))
            yield t0, w, value


def transform_lesser(t, tp, wt, wp, matrix, probes, omega, sigma):
    """Transform using the lesser-frequency kernel exp(+i omega (t - t_prime))."""
    return transform_grid(t, tp, wt, wp, matrix, probes, omega, sigma, sign=1)


def transform_greater(t, tp, wt, wp, matrix, probes, omega, sigma):
    """Transform using the greater-frequency kernel exp(-i omega (t - t_prime))."""
    return transform_grid(t, tp, wt, wp, matrix, probes, omega, sigma, sign=-1)


def write_transforms(path, groups, probes, omega, sigma, correlator):
    """Transform each site pair and write the results to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("center_site", "site_index", "probe_time", "omega", "real_part", "imaginary_part"))
        transform = transform_lesser if correlator == "lesser" else transform_greater
        for pair, data in sorted(groups.items()):
            grid = prepare_grid(pair, data)
            for t0, w, value in transform(*grid, probes, omega, sigma):
                writer.writerow((*pair, f"{t0:.15g}", f"{w:.15g}", f"{value.real:.15g}", f"{value.imag:.15g}"))


def main():
    a = parse_args()
    groups = read_correlators(a.input_csv)
    omega = np.linspace(a.omega_min, a.omega_max, a.omega_count)
    probes = np.linspace(a.probe_min, a.probe_max, a.probe_count)
    write_transforms(a.output_csv, groups, probes, omega, a.sigma, a.correlator)
    print(f"Wrote {a.output_csv}; finite time range and probe truncation affect the result.")


if __name__ == "__main__":
    main()
