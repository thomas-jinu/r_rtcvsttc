"""ED for two frozen-Hamiltonian intervals, A(t)=0.4t, Vch=4."""
import numpy as np
import exact_diagonalization as ed


def hamiltonian_at(states, vch, phase):
    h = ed.hamiltonian(states, vch).astype(complex)
    index = {state: i for i, state in enumerate(states)}
    for col, state in enumerate(states):
        for spin_offset in (0, ed.L):
            for j in range(ed.L - 1):
                for target, source, direction in ((j + 1 + spin_offset, j + spin_offset, 1), (j + spin_offset, j + 1 + spin_offset, -1)):
                    a = ed.annihilate(state, source)
                    if a is None:
                        continue
                    c = ed.create(a[0], target)
                    if c is not None:
                        h[index[c[0]], col] -= a[1] * c[1] * (np.exp(1j * direction * phase) - 1)
    return h


def step(h, state, dt):
    energies, vectors = np.linalg.eigh(h)
    return vectors @ (np.exp(-1j * energies * dt) * (vectors.conj().T @ state))


for direction in (1, -1):
    ref, excited = ed.gs, ed.created
    for i in range(2):
        time = direction * i * 0.5
        dt = direction * 0.5
        ref = step(hamiltonian_at(ed.initial, 0, 0.4 * time), ref, dt)
        excited = step(hamiltonian_at(ed.excited, 4, 0.4 * time), excited, dt)
    value = np.vdot(ed.apply_creation(ref), excited)
    print(f't={direction:g}, frozen H: {value.real:.12g} {value.imag:+.12g}i')
