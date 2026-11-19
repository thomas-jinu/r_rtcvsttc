"""Fixed-probe two-time and relative-time responses for raw correlator CSVs.

The Gaussian double-time and one-sided transforms use the same Fourier sign
for each correlator type. They have different windows, factors, and
normalizations; their outputs are not directly identical.
"""
from pathlib import Path
import numpy as np

from fourier_probe_twotime import read_correlators, prepare_grid, transform_grid
from fourier_relative_time import relative_time_slice, transform_relative_time


def _get_pair(path, pair):
    path = Path(path)
    groups = read_correlators(path)
    if pair not in groups:
        raise ValueError(f'{path}: site pair {pair} not found; available: {sorted(groups)}')
    return groups[pair]


def _dt(axis, description):
    differences = np.diff(axis)
    if len(differences) < 1 or np.any(differences <= 0):
        raise ValueError(f'{description} needs at least two increasing times')
    if not np.allclose(differences, differences[0], rtol=1e-7, atol=1e-9):
        raise ValueError(f'{description} must be uniformly sampled')
    return float(differences.max())


def fixed_probe_response(twotime_inputs, singletime_inputs, site_pair, probe_time,
                         frequencies, sigma, relative_sigma, shifts, relative_signs=None):
    """Compute individual and summed results on a common shifted-frequency grid.

    Each inputs mapping has keys 'lesser' and/or 'greater' and values Path.
    Relative inputs may be combined singletime files, individual step files,
    or two-time files with a forward slice. Both mappings must select the same
    correlator types. Frequencies are display frequencies; the transform is
    evaluated at display frequency + per-kind shift. Optional relative_signs
    overrides the original one-sided kernel sign independently for each kind.
    """
    kinds = set(twotime_inputs)
    if not kinds or kinds != set(singletime_inputs) or not kinds <= {'lesser', 'greater'}:
        raise ValueError('Select matching nonempty lesser/greater inputs for both methods')
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all():
        raise ValueError('Frequencies must be a nonempty finite one-dimensional array')
    if not np.isfinite([probe_time, sigma, relative_sigma]).all() or sigma <= 0 or relative_sigma <= 0:
        raise ValueError('Probe time and widths must be finite, and widths must be positive')

    two_components = {}
    relative_components = {}
    allowed = []
    time_slices = {}
    for kind in sorted(kinds):
        shift = float(shifts[kind])
        if not np.isfinite(shift):
            raise ValueError(f'{kind}: frequency shift must be finite')
        requested = frequencies + shift
        data = _get_pair(twotime_inputs[kind], site_pair)
        t, tp, wt, wp, matrix = prepare_grid(site_pair, data)
        nyquist = np.pi / max(_dt(t, f'{kind} t'), _dt(tp, f'{kind} t_prime'))
        if not (t[0] <= probe_time <= t[-1] and tp[0] <= probe_time <= tp[-1]):
            raise ValueError(f'{kind}: probe center lies outside the two-time grid')
        sign = 1 if kind == 'lesser' else -1
        two_components[kind] = np.array([value for _, _, value in
            transform_grid(t, tp, wt, wp, matrix, [probe_time], requested, sigma, sign)])

        singletime_data = _get_pair(singletime_inputs[kind], site_pair)
        tau, values = relative_time_slice(site_pair, singletime_data, probe_time)
        nyquist_relative = np.pi / _dt(tau, f'{kind} relative time')
        sign_relative = (1 if kind == 'lesser' else -1) if relative_signs is None else relative_signs[kind]
        if sign_relative not in (-1, 1):
            raise ValueError(f'{kind}: relative Fourier sign must be -1 or +1')
        response, intensity = transform_relative_time(tau, values, requested, relative_sigma, sign_relative)
        relative_components[kind] = {'response': response, 'intensity': intensity}
        time_slices[kind] = {'tau': tau, 'correlator': values}
        limit = min(nyquist, nyquist_relative)
        allowed.append((-limit - shift, limit - shift))

    return {'frequencies': frequencies, 'probe_time': probe_time,
            'two_time': sum(two_components.values()), 'two_time_components': two_components,
            'relative_response': sum((x['response'] for x in relative_components.values())),
            'relative_intensity': sum((x['intensity'] for x in relative_components.values())),
            'relative_components': relative_components, 'time_slices': time_slices,
            'nyquist_display_bounds': (max(x[0] for x in allowed), min(x[1] for x in allowed))}
