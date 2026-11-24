#!/usr/bin/env python3
"""Relative-time transform at fixed t_prime = T (probe time).

F(T, omega) = integral dt exp(sign*i*omega*(t-T)) C(t, T).
No window or Fourier normalization is applied. Values at intermediate T are
linearly interpolated along t_prime; extrapolation is not allowed.
"""
import numpy as np

from fourier_probe_twotime import read_grids, weights


def transform_pair(pair, data, probes, omega, sign, input_convention):
    t = np.array(sorted({key[0] for key in data}))
    tp = np.array(sorted({key[1] for key in data}))
    wt = weights(t)
    weights(tp)  # Validate the second axis as well.
    if len(data) != len(t) * len(tp):
        raise ValueError(f'Incomplete rectangular time grid for sites {pair}')
    if input_convention not in ('correlator', 'green'):
        raise ValueError('Unknown input convention')
    matrix = np.array([[data[(x, y)] for y in tp] for x in t])
    if input_convention == 'green':
        matrix *= -1j
    for T in probes:
        if not tp[0] <= T <= tp[-1]:
            raise ValueError(f'Probe time {T:g} is outside t_prime range [{tp[0]:g}, {tp[-1]:g}] for sites {pair}')
        # np.interp handles complex data and uses exact values at grid points.
        column = np.array([np.interp(T, tp, row) for row in matrix])
        values = np.exp(sign * 1j * np.outer(omega, t - T)) @ (wt * column)
        for w, value in zip(omega, values):
            yield (*pair, float(T), float(w), value)


def transform_inputs(inputs, probes, omega, input_convention):
    """Transform and sum lesser/greater inputs, requiring identical input grids."""
    result = {}
    reference = None
    for path, kind in inputs:
        if kind not in ('lesser', 'greater'):
            raise ValueError(f'Unknown frequency sign: {kind}')
        groups = read_grids(path)
        grid = {pair: set(data) for pair, data in groups.items()}
        if reference is None:
            reference = grid
        elif grid != reference:
            raise ValueError(f'Input site/time grids do not match: {path}')
        sign = 1 if kind == 'lesser' else -1
        for pair, data in sorted(groups.items()):
            for center, site, T, w, value in transform_pair(pair, data, probes, omega, sign, input_convention):
                key = (center, site, T, w)
                result[key] = result.get(key, 0j) + value
    return result
