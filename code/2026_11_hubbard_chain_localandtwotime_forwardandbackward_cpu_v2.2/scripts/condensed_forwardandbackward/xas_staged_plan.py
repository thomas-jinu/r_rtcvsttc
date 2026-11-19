#!/usr/bin/env python3
"""Create a non-executable plan for staged two-branch XAS evolution.

This does not produce DMRG++ inputs or run a calculation. Restart-vector
mapping, Hamiltonian rebuilding and two-branch targeting must be validated
against the actual DMRG++ build before an executable workflow can be generated.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--time-count', type=int, required=True)
    p.add_argument('--time-step', type=float, required=True)
    p.add_argument('--core-hole-potential', type=float, required=True, help='Attractive strength Vch >= 0; excited branch uses H - Vch*n_m')
    p.add_argument('--site', type=int, required=True, help='Core-hole site index')
    p.add_argument('--output', type=Path, required=True, help='JSON plan; must not already exist')
    a = p.parse_args()
    if a.time_count < 1 or not math.isfinite(a.time_step) or a.time_step <= 0:
        p.error('time-count and finite time-step must be positive')
    if not math.isfinite(a.core_hole_potential) or a.core_hole_potential < 0 or a.site < 0:
        p.error('core-hole-potential must be finite and nonnegative; site must be nonnegative')
    probes = []
    for probe in range(a.time_count):
        stages = []
        for direction in (1, -1):
            for current in range(probe, a.time_count - 1 if direction == 1 else 0, direction):
                target = current + direction
                for branch in ('reference', 'excited'):
                    stages.append({'branch': branch, 'direction': 'forward' if direction == 1 else 'backward',
                                   'restart_from_probe': current == probe and branch == 'reference',
                                   'start_index': current, 'end_index': target,
                                   'start_time': current * a.time_step, 'end_time': target * a.time_step,
                                   'potential_shift_at_site': -a.core_hole_potential if branch == 'excited' else 0.0,
                                   'measure_after': branch == 'excited'})
        probes.append({'probe_index': probe, 'probe_time': probe * a.time_step, 'stages': stages})
    plan = {'status': 'non-executable design only', 'site': a.site, 'time_count': a.time_count,
            'time_step': a.time_step, 'core_hole_potential': a.core_hole_potential,
            'backward_convention': 'H - Vch*n_m applied on the excited branch in both directions; physical interpretation unvalidated',
            'evolution_stage_count': sum(len(p['stages']) for p in probes), 'probes': probes}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
        f.write('\n')
    print(f'Wrote non-executable plan to {a.output}: {plan["evolution_stage_count"]} staged evolutions')


if __name__ == '__main__':
    main()
