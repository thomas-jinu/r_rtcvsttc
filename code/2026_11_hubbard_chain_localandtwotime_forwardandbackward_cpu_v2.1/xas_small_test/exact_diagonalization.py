"""Four-site exact diagonalization for the staged XAS restart test."""
import numpy as np

L, site, interaction, dt = 4, 1, 8.0, 0.5

def basis(nup, ndown):
    return [s for s in range(1 << (2 * L)) if (s & 15).bit_count() == nup and (s >> L).bit_count() == ndown]

def annihilate(s, orbital):
    if not (s >> orbital) & 1:
        return None
    return s ^ (1 << orbital), (-1) ** ((s & ((1 << orbital) - 1)).bit_count())

def create(s, orbital):
    if (s >> orbital) & 1:
        return None
    return s ^ (1 << orbital), (-1) ** ((s & ((1 << orbital) - 1)).bit_count())

def hamiltonian(states, vch):
    index = {state: i for i, state in enumerate(states)}
    h = np.zeros((len(states), len(states)))
    for col, state in enumerate(states):
        h[col, col] = interaction * sum(((state >> j) & 1) * ((state >> (j + L)) & 1) for j in range(L))
        h[col, col] -= vch * (((state >> site) & 1) + ((state >> (site + L)) & 1))
        for spin_offset in (0, L):
            for j in range(L - 1):
                for target, source in ((j + spin_offset, j + 1 + spin_offset), (j + 1 + spin_offset, j + spin_offset)):
                    a = annihilate(state, source)
                    if a is None:
                        continue
                    c = create(a[0], target)
                    if c is not None:
                        h[index[c[0]], col] -= a[1] * c[1]
    return h

def evolve(h, state):
    energies, vectors = np.linalg.eigh(h)
    return vectors @ (np.exp(-1j * energies * dt) * (vectors.conj().T @ state))

initial, excited = basis(2, 2), basis(3, 2)
h0 = hamiltonian(initial, 0.0)
e, vec = np.linalg.eigh(h0)
gs = vec[:, 0]
excited_index = {s: i for i, s in enumerate(excited)}

def apply_creation(state):
    result = np.zeros(len(excited), dtype=complex)
    for i, s in enumerate(initial):
        c = create(s, site)
        if c is not None:
            result[excited_index[c[0]]] += c[1] * state[i]
    return result

reference = evolve(h0, gs)
created = apply_creation(gs)
print(f'Ground-state energy: {e[0]:.12g}')
for vch in (0.0, 4.0):
    value = np.vdot(apply_creation(reference), evolve(hamiltonian(excited, vch), created))
    print(f'Vch={vch:g}, t={dt:g}: {value.real:.12g} {value.imag:+.12g}i')
