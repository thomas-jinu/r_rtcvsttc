"""One-sided Gaussian-windowed relative-time transforms of raw correlators.

Inspired by Lai and Zhu (2019), Eq. 3: the greater correlator uses
exp(+i omega tau), tau >= 0. The paper uses exponential lifetime broadening;
here a Gaussian window is used instead. DOS additionally includes the raw
lesser correlator with exp(-i omega tau). No core-hole physics is introduced.
"""
import numpy as np

from fourier_probe_twotime import read_grids, weights


def transform_rtc(inputs, probes, omega, sigma, input_convention='correlator'):
    """Return complex integrals keyed by (center, site, probe, omega).

    A probe between samples is interpolated in both time axes. Integration
    starts exactly at t=T and ends at the last sampled t. Input grids must
    match. Green inputs use the conventional +i (lesser), -i (greater)
    prefactors, which are removed before transforming.
    """
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    if input_convention not in ('correlator', 'green'):
        raise ValueError('input_convention must be correlator or green')
    result, reference = {}, None
    for path, kind in inputs:
        if kind not in ('lesser', 'greater'):
            raise ValueError(f'Unknown correlator kind: {kind}')
        groups = read_grids(path)
        grid = {pair: set(data) for pair, data in groups.items()}
        if reference is None:
            reference = grid
        elif grid != reference:
            raise ValueError(f'Input site/time grids do not match: {path}')
        for pair, data in sorted(groups.items()):
            t = np.array(sorted({key[0] for key in data}))
            tp = np.array(sorted({key[1] for key in data}))
            if len(data) != len(t) * len(tp):
                raise ValueError(f'Incomplete rectangular time grid for sites {pair}')
            weights(t)
            weights(tp)
            matrix = np.array([[data[(x, y)] for y in tp] for x in t])
            if input_convention == 'green':
                matrix *= -1j if kind == 'lesser' else 1j
            for T in probes:
                if not tp[0] <= T <= tp[-1] or not t[0] <= T < t[-1]:
                    raise ValueError(f'Probe {T:g} must lie within t_prime and before the last t for sites {pair}')
                column = np.array([np.interp(T, tp, row) for row in matrix])
                sample_t = np.concatenate(([T], t[t > T]))
                sample = np.interp(sample_t, t, column)
                tau = sample_t - T
                window = np.exp(-0.5 * (tau / sigma)**2)
                sign = 1 if kind == 'greater' else -1
                values = np.exp(sign * 1j * np.outer(omega, tau)) @ (weights(sample_t) * window * sample)
                for w, value in zip(omega, values):
                    key = (*pair, float(T), float(w))
                    result[key] = result.get(key, 0j) + value
    if reference is None:
        raise ValueError('At least one input is required')
    return result
