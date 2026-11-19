#!/usr/bin/env python3
"""Gaussian-windowed one-sided relative-time transform of a raw correlator.

For a fixed initial time t_prime, extract C(t_prime + tau, t_prime) and compute
    A(omega) = -i integral_0^infinity d tau exp(sign * i omega tau)
               * exp(-tau**2 / (2 sigma**2)) C(tau)
The greater kernel uses sign=-1 and the lesser kernel uses sign=+1,
matching the corresponding two-time transform conventions. These signs differ
from the greater-kernel convention in Lai and Zhu, PRL 122, 207401, Eq. (3).
    I(omega) = -Im A(omega) / pi.
The integral is truncated at the last available time and uses trapezoidal
weights. Input values are raw correlators, without an i prefactor. The input
must contain tau=0 and a uniformly spaced positive-time slice; no values are
interpolated. Frequencies are in inverse time units (hbar = 1); sigma is in
time units. The Gaussian replaces the paper's exponential lifetime broadening,
so this is not exactly the lifetime-broadened spectrum of Eq. (3).

Eq. (3) describes an absorption correlator with a core-hole Hamiltonian acting
between the operators. Ordinary greater-correlator data without that evolution
gives a related one-sided transform, not the paper's core-hole XAS. A single
input is not automatically summed over spins or sites.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from fourier_probe_twotime import read_correlators, weights


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--correlator', choices=('lesser', 'greater'), required=True,
                        help='greater uses exp(-i omega tau); lesser uses exp(+i omega tau)')
    parser.add_argument('--initial-time', type=float, required=True,
                        help='Fixed t_prime at which the correlator starts')
    parser.add_argument('--sigma', type=float, required=True,
                        help='Gaussian relative-time window width in time units')
    parser.add_argument('--omega-min', type=float, required=True)
    parser.add_argument('--omega-max', type=float, required=True)
    parser.add_argument('--omega-count', type=int, required=True)
    args = parser.parse_args()
    if not all(np.isfinite(x) for x in (args.initial_time, args.sigma, args.omega_min, args.omega_max)):
        parser.error('time, sigma, and frequency bounds must be finite')
    if args.sigma <= 0 or args.omega_count < 1 or args.omega_min > args.omega_max:
        parser.error('sigma and count must be positive, and minimum <= maximum')
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error('output must not overwrite the input')
    return args


def relative_time_slice(pair, data, initial_time):
    """Extract the forward-time slice at an existing t_prime without interpolation."""
    available = np.array(sorted({tp for _, tp in data}))
    matches = available[np.isclose(available, initial_time, rtol=0, atol=1e-9)]
    if len(matches) != 1:
        raise ValueError(f'Sites {pair}: initial time {initial_time} must match exactly one t_prime in the data')
    t0 = matches[0]
    points = sorted((t - t0, value) for (t, tp), value in data.items()
                    if tp == t0 and t >= t0 - 1e-9)
    if not points or abs(points[0][0]) > 1e-9:
        raise ValueError(f'Sites {pair}: the slice must contain the equal-time point tau=0')
    tau = np.array([x for x, _ in points])
    tau[0] = 0.0  # Suppress small CSV rounding differences at equal time.
    values = np.array([value for _, value in points])
    return tau, values


def transform_relative_time(tau, values, omega, sigma, sign):
    """Return the Gaussian-windowed response and intensity for a finite slice."""
    weighted = weights(tau) * np.exp(-0.5 * (tau / sigma) ** 2) * values
    response = np.array([-1j * np.dot(np.exp(sign * 1j * w * tau), weighted) for w in omega])
    return response, -response.imag / np.pi


def transform_greater(tau, values, omega, sigma):
    """Use the greater kernel exp(-i omega tau)."""
    return transform_relative_time(tau, values, omega, sigma, sign=-1)


def transform_lesser(tau, values, omega, sigma):
    """Use the lesser kernel exp(+i omega tau)."""
    return transform_relative_time(tau, values, omega, sigma, sign=1)


def main():
    args = parse_args()
    groups = read_correlators(args.input_csv)
    omega = np.linspace(args.omega_min, args.omega_max, args.omega_count)
    transform = transform_lesser if args.correlator == 'lesser' else transform_greater
    with args.output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('center_site', 'site_index', 'initial_time', 'omega',
                         'response_real', 'response_imag', 'intensity'))
        for pair, data in sorted(groups.items()):
            tau, values = relative_time_slice(pair, data, args.initial_time)
            response, intensity = transform(tau, values, omega, args.sigma)
            for w, a, intensity_value in zip(omega, response, intensity):
                writer.writerow((*pair, args.initial_time, f'{w:.15g}', f'{a.real:.15g}',
                                 f'{a.imag:.15g}', f'{intensity_value:.15g}'))
    print(f'Wrote {args.output_csv}; results are truncated at the available relative time.')


if __name__ == '__main__':
    main()
