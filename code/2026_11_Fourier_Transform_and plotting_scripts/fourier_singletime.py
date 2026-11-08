"""One-sided Fourier transform of a fixed-t_prime, fixed-site time-series CSV.

This is a single-time transform, NOT a double-time probe integral or a
retarded spectral function. No Hermitian completion or extra factor of -i
is applied. Choose the Fourier exponent appropriate for your convention.
"""

import argparse
import cmath
import csv
import math
from pathlib import Path


def load_center_series(csv_path: Path, center_site: int) -> tuple[float, list[float], list[complex]]:
    """Return fixed t_prime, sorted relative times tau=t-t_prime, and G values."""
    samples: dict[float, complex] = {}
    t_primes: set[float] = set()
    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"t_prime", "t", "center_site", "site_index", "real_part", "imaginary_part"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if int(row["center_site"]) != center_site or int(row["site_index"]) != center_site:
                continue
            t_prime = float(row["t_prime"])
            t_primes.add(t_prime)
            time = float(row["t"])
            if time in samples:
                raise ValueError(f"Duplicate center-site sample at t={time}")
            samples[time] = complex(float(row["real_part"]), float(row["imaginary_part"]))
    if not samples:
        raise ValueError(f"No center-site samples for site {center_site}")
    if len(t_primes) != 1:
        raise ValueError("CSV contains multiple t_prime values: select a single slice first")
    t_prime = t_primes.pop()
    times = sorted(samples)
    tau = [time - t_prime for time in times]
    if tau[0] < -1e-9:
        raise ValueError("This function expects t >= t_prime (a one-sided time series)")
    if len(tau) < 2:
        raise ValueError("At least two time samples are required")
    return t_prime, tau, [samples[time] for time in times]


def fourier_singletime(
    tau: list[float],
    signal: list[complex],
    omega_axis: list[float],
    *,
    exponent_sign: int = -1,
    gaussian_sigma: float | None = None,
) -> list[complex]:
    """Calculate F(w) = int dtau G(tau) exp(sign*i*w*tau) with trapezoids.

    gaussian_sigma, if provided, multiplies G by exp(-tau**2/(2*sigma**2));
    it does not normalize the window. No factors of -i or 1/pi are added.
    """
    if len(tau) != len(signal) or len(tau) < 2:
        raise ValueError("tau and signal need equal lengths of at least two")
    if exponent_sign not in (-1, 1):
        raise ValueError("exponent_sign must be -1 or +1")
    if gaussian_sigma is not None and (gaussian_sigma <= 0 or not math.isfinite(gaussian_sigma)):
        raise ValueError("gaussian_sigma must be positive and finite")
    if not all(math.isfinite(t) for t in tau) or not all(b > a for a, b in zip(tau, tau[1:])):
        raise ValueError("tau must be finite and strictly increasing")
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in signal):
        raise ValueError("signal contains a non-finite value")
    weights = [
        math.exp(-t*t/(2*gaussian_sigma**2)) if gaussian_sigma is not None else 1.0
        for t in tau
    ]
    result = []
    for omega in omega_axis:
        if not math.isfinite(omega):
            raise ValueError("omega values must be finite")
        values = [v*w*cmath.exp(exponent_sign*1j*omega*t)
                  for t, v, w in zip(tau, signal, weights)]
        result.append(sum(
            (b-a)*(va+vb)/2 for a, b, va, vb
            in zip(tau, tau[1:], values, values[1:])
        ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--center-site", type=int, default=3)
    parser.add_argument("--omega-min", type=float, default=-20.0)
    parser.add_argument("--omega-max", type=float, default=20.0)
    parser.add_argument("--omega-step", type=float, default=0.1)
    parser.add_argument("--exponent-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--gaussian-sigma", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not math.isfinite(args.omega_step) or args.omega_step <= 0 or args.omega_max <= args.omega_min:
        parser.error("omega-step must be positive and omega-max > omega-min")
    n = math.ceil((args.omega_max - args.omega_min)/args.omega_step)
    omega_axis = [args.omega_min + i*args.omega_step for i in range(n)]
    t_prime, tau, signal = load_center_series(args.csv_file, args.center_site)
    transformed = fourier_singletime(
        tau, signal, omega_axis,
        exponent_sign=args.exponent_sign, gaussian_sigma=args.gaussian_sigma,
    )
    output = args.output or args.csv_file.with_name(args.csv_file.stem + "_fourier.csv")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["omega", "t_prime", "center_site", "fourier_real", "fourier_imag"])
        for omega, value in zip(omega_axis, transformed):
            writer.writerow([f"{omega:.15g}", f"{t_prime:.15g}", args.center_site,
                             f"{value.real:.15g}", f"{value.imag:.15g}"])
    print(f"Wrote {len(transformed)} frequency samples to {output}")


if __name__ == "__main__":
    main()
