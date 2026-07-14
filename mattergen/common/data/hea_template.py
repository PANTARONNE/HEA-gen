# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared utilities for the fixed-geometry HEA (high-entropy alloy) task.

All training structures and all generated structures are anchored to a single
canonical FCC(111) grid built at ``DEFAULT_A`` with ``ase.build.fcc111``. Only
the atom types vary; positions and cell are fixed. After generation, the real
lattice constant is recovered from the generated composition via Vegard's law.

The construction here mirrors ``build_hea_surface.py`` so that the diffusion
model, the training data, and the final written CIFs all share the exact same
geometry convention.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Template definition (must match build_hea_surface.py defaults).
# ---------------------------------------------------------------------------
DEFAULT_A = 3.6  # canonical FCC lattice constant (Angstrom) used to build the grid
SLAB_SIZE = (4, 4, 4)  # fcc111 repeats -> 64 sites
VACUUM_TOTAL = 15.0  # total vacuum thickness (Angstrom); ase gets vacuum=VACUUM_TOTAL/2

# FCC-equivalent lattice constants (Angstrom) for Vegard's-law estimation.
# Copied from build_hea_surface.py FCC_LATTICE_CONSTANTS, restricted to the pool.
FCC_LATTICE_CONSTANTS = {
    "Fe": 3.59,
    "Co": 3.54,
    "Ni": 3.52,
    "Cu": 3.61,
    "Zn": 3.94,
    "Ga": 4.51,
    "Mo": 3.86,
    "Sn": 4.89,
    "W": 3.93,
}

# One-based atomic numbers of the 9 pool elements (In excluded).
HEA_ELEMENTS = ["Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Mo", "Sn", "W"]


def build_template_slab(a: float = DEFAULT_A):
    """Return an ase.Atoms FCC(111) slab template with the given lattice constant.

    The z-direction has a fixed absolute vacuum (VACUUM_TOTAL), matching
    build_hea_surface.py.
    """
    from ase.build import fcc111

    slab = fcc111(symbol="Cu", size=SLAB_SIZE, a=a, vacuum=VACUUM_TOTAL / 2.0)
    slab.set_pbc(True)
    return slab


def _site_keys(scaled_positions: np.ndarray) -> tuple[list[tuple], int]:
    """Compute geometry-invariant site keys for matching atoms to grid sites.

    Key = (layer_rank_by_z, round(x % 1, 3), round(y % 1, 3)). The in-plane
    (x, y) fractional coordinates of an fcc111 grid are independent of the
    lattice constant and of the (fixed) vacuum, so this key is stable across all
    structures. Coordinates are wrapped into [0, 1) before rounding to avoid
    floating-point boundary issues (e.g. 0.9999 vs 0.0).
    """
    sp = scaled_positions % 1.0

    def _wrap(v: float) -> float:
        # Round to 3 decimals FIRST, then wrap into [0, 1). Rounding can push a
        # value like 0.9995 up to 1.0; the trailing `% 1.0` maps it back to 0.0
        # so it matches the canonical grid (which only has coords in [0, 1)).
        return round(round(float(v), 3) % 1.0, 3)

    z = np.round(sp[:, 2], 3)
    unique_z = sorted(set(z.tolist()))
    layer_of = {zz: i for i, zz in enumerate(unique_z)}
    keys = [
        (layer_of[round(float(zz), 3)], _wrap(x), _wrap(y))
        for x, y, zz in sp
    ]
    return keys, len(unique_z)


def canonical_template():
    """Build the canonical grid and return (pos_frac, cell, key_to_index).

    pos_frac: (64, 3) fractional coordinates of the DEFAULT_A grid.
    cell:     (3, 3) lattice matrix of the DEFAULT_A grid.
    key_to_index: dict mapping each site key to its canonical site index.
    """
    slab = build_template_slab(DEFAULT_A)
    pos_frac = slab.get_scaled_positions()
    cell = np.asarray(slab.cell.array, dtype=float)
    keys, n_layers = _site_keys(pos_frac)
    assert n_layers == 4, f"Expected 4 layers, got {n_layers}"
    key_to_index = {k: i for i, k in enumerate(keys)}
    assert len(key_to_index) == len(pos_frac), "Canonical site keys are not unique"
    return pos_frac.astype(np.float32), cell.astype(np.float32), key_to_index


def map_atoms_to_canonical(
    scaled_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    key_to_index: dict,
) -> np.ndarray:
    """Reorder `atomic_numbers` onto the canonical site order.

    Each atom is matched to a canonical site by its geometry-invariant key.
    Returns an (n_sites,) array of atomic numbers in canonical site order, or
    raises ValueError if the mapping is not bijective (structure not on grid).
    """
    keys, n_layers = _site_keys(scaled_positions)
    if n_layers != 4:
        raise ValueError(f"Structure has {n_layers} layers, expected 4")
    out = np.full(len(key_to_index), -1, dtype=np.int64)
    for k, z in zip(keys, atomic_numbers):
        idx = key_to_index.get(k, -1)
        if idx < 0:
            raise ValueError(f"Atom key {k} not found on canonical grid")
        out[idx] = int(z)
    if (out < 0).any():
        raise ValueError("Mapping did not cover all canonical sites (not bijective)")
    return out


def vegard_lattice_constant(symbols) -> float:
    """Estimate lattice constant via ratio-weighted Vegard's law from a list of
    element symbols (one per atom)."""
    symbols = list(symbols)
    n = len(symbols)
    a_vals = np.array([FCC_LATTICE_CONSTANTS[s] for s in symbols])
    return float(a_vals.sum() / n)
