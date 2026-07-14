# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Prepare the HEA dataset cache for atom-type-only diffusion.

Reads all CIFs under the raw structure directory, drops any structure that
contains an out-of-pool element (In), maps every atom onto the canonical
DEFAULT_A FCC(111) grid (so all structures share identical positions/cell and
only differ in atom types), and writes the numpy cache expected by
`CrystalDataset.from_cache_path` under `<cache>/hea/{train,val}`.

No primitive/Niggli reduction is applied: the full 64-site supercell is kept.

Usage:
    python -m mattergen.scripts.prepare_hea_dataset \
        --raw-dir datasets/randomStructures \
        --cache-folder datasets/cache \
        --dataset-name hea --val-fraction 0.1 --seed 42
"""

import argparse
import os
from glob import glob

import numpy as np
from ase.io import read

from mattergen.common.data.hea_template import (
    HEA_ELEMENTS,
    canonical_template,
    map_atoms_to_canonical,
)

# Inlined from mattergen.common.data.dataset.CORE_STRUCTURE_FILE_NAMES to keep
# this script importable without heavy deps (pymatgen/torch).
CORE_STRUCTURE_FILE_NAMES = {
    "pos": "pos.npy",
    "cell": "cell.npy",
    "atomic_numbers": "atomic_numbers.npy",
    "num_atoms": "num_atoms.npy",
    "structure_id": "structure_id.npy",
}

ALLOWED = set(HEA_ELEMENTS)


def _write_split(cache_dir, pos_std, cell_std, records):
    """Write one split (train or val) to `.npy` files.

    records: list of (structure_id, atomic_numbers_canonical(64,)).
    All structures share the same pos_std (64,3) and cell_std (3,3).
    """
    os.makedirs(cache_dir, exist_ok=True)
    n = len(records)
    # pos: stack the canonical grid n times -> (n*64, 3)
    pos = np.tile(pos_std, (n, 1)).astype(np.float32)
    # cell: (n, 3, 3)
    cell = np.tile(cell_std[None], (n, 1, 1)).astype(np.float32)
    atomic_numbers = np.concatenate([r[1] for r in records]).astype(np.int64)
    num_atoms = np.full(n, pos_std.shape[0], dtype=np.int64)
    structure_id = np.array([r[0] for r in records])

    np.save(f"{cache_dir}/{CORE_STRUCTURE_FILE_NAMES['pos']}", pos)
    np.save(f"{cache_dir}/{CORE_STRUCTURE_FILE_NAMES['cell']}", cell)
    np.save(f"{cache_dir}/{CORE_STRUCTURE_FILE_NAMES['atomic_numbers']}", atomic_numbers)
    np.save(f"{cache_dir}/{CORE_STRUCTURE_FILE_NAMES['num_atoms']}", num_atoms)
    np.save(f"{cache_dir}/{CORE_STRUCTURE_FILE_NAMES['structure_id']}", structure_id)
    print(f"[done] wrote {n} structures to {cache_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="datasets/randomStructures",
                   help="Directory containing the raw HEA CIF files.")
    p.add_argument("--cache-folder", default="datasets/cache",
                   help="Root cache folder. Output goes to <cache>/<name>/{train,val}.")
    p.add_argument("--dataset-name", default="hea")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    pos_std, cell_std, key_to_index = canonical_template()
    print(f"[info] canonical grid: {pos_std.shape[0]} sites, "
          f"cell diag {np.round(np.linalg.norm(cell_std, axis=1), 4)}")

    files = sorted(glob(f"{args.raw_dir}/*.cif"))
    print(f"[info] found {len(files)} CIF files")

    records = []
    skipped_in, skipped_map = 0, 0
    for f in files:
        atoms = read(f)
        symbols = atoms.get_chemical_symbols()
        if any(s not in ALLOWED for s in symbols):
            skipped_in += 1
            continue
        try:
            an_canonical = map_atoms_to_canonical(
                atoms.get_scaled_positions(),
                np.array(atoms.get_atomic_numbers()),
                key_to_index,
            )
        except ValueError as e:
            skipped_map += 1
            print(f"[warn] skip {os.path.basename(f)}: {e}")
            continue
        sid = os.path.splitext(os.path.basename(f))[0]
        records.append((sid, an_canonical))

    print(f"[info] usable: {len(records)}, skipped(out-of-pool): {skipped_in}, "
          f"skipped(mapping): {skipped_map}")

    # Shuffle + split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(records))
    n_val = max(1, int(round(len(records) * args.val_fraction)))
    val_idx = set(idx[:n_val].tolist())
    train_records = [records[i] for i in range(len(records)) if i not in val_idx]
    val_records = [records[i] for i in range(len(records)) if i in val_idx]

    base = f"{args.cache_folder}/{args.dataset_name}"
    _write_split(f"{base}/train", pos_std, cell_std, train_records)
    _write_split(f"{base}/val", pos_std, cell_std, val_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
