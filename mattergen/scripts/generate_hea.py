# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate HEA structures: fixed FCC(111) geometry, only atom types are sampled.

Pipeline:
    1. Load a model trained with the atom-only diffusion config.
    2. Inject the HEA element mask so only the 9 pool elements can be produced.
    3. Sample atom types on the fixed canonical DEFAULT_A grid (positions/cell
       stay constant throughout denoising).
    4. Post-process each structure with Vegard's law: compute the real lattice
       constant from the generated composition, rebuild the FCC(111) slab at
       that constant, and place the generated elements on it. Write CIFs.

Usage:
    python -m mattergen.scripts.generate_hea \
        --model_path outputs/singlerun/<date>/<time> \
        --num_structures 100 --batch_size 64 --output_dir outputs/hea
"""

import os
from pathlib import Path

import fire
import numpy as np
import torch
from ase.io import write
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mattergen.common.data.hea_template import (
    SLAB_SIZE,
    VACUUM_TOTAL,
    build_template_slab,
    vegard_lattice_constant,
)
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.data_utils import get_element_symbol
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH, get_device

# Inject the HEA element mask so sampling only ever produces the 9 pool elements.
ELEMENT_MASK_OVERRIDE = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_to_hea_elements',_partial_:True}"
)


def _build_and_write(atomic_numbers: np.ndarray, out_path: Path, index: int) -> str:
    """Rebuild the FCC(111) slab at the Vegard lattice constant and write a CIF."""
    symbols = [get_element_symbol(int(z)) for z in atomic_numbers]
    a = vegard_lattice_constant(symbols)
    slab = build_template_slab(a=a)
    assert len(slab) == len(symbols), (
        f"site count mismatch: slab {len(slab)} vs generated {len(symbols)}"
    )
    slab.set_chemical_symbols(symbols)
    formula = slab.get_chemical_formula()
    fname = out_path / f"gen_{index:04d}_{formula}.cif"
    write(str(fname), slab)
    return str(fname)


def main(
    model_path: str,
    output_dir: str = "outputs/hea",
    num_structures: int = 100,
    batch_size: int = 64,
    checkpoint_epoch: str = "last",
    sampling_config_name: str = "atom_only",
    sampling_config_path: str | None = None,
    record_trajectories: bool = False,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Load model with HEA element mask injected ---
    ckpt_info = MatterGenCheckpointInfo(
        model_path=Path(model_path).resolve(),
        load_epoch=checkpoint_epoch,
        config_overrides=[ELEMENT_MASK_OVERRIDE],
        strict_checkpoint_loading=True,
    )
    model = load_model_diffusion(ckpt_info).to(get_device())
    model.eval()

    # --- Build sampler + fixed-geometry template loader ---
    cfg_path = Path(sampling_config_path) if sampling_config_path else DEFAULT_SAMPLING_CONFIG_PATH
    import hydra

    with hydra.initialize_config_dir(os.path.abspath(str(cfg_path))):
        sampling_cfg = hydra.compose(
            config_name=sampling_config_name,
            overrides=[
                f"+condition_loader_partial.num_structures={num_structures}",
                f"+condition_loader_partial.batch_size={batch_size}",
            ],
        )
    print("Sampling config:\n", OmegaConf.to_yaml(sampling_cfg, resolve=True))

    condition_loader = instantiate(sampling_cfg.condition_loader_partial)()
    sampler = instantiate(sampling_cfg.sampler_partial)(pl_module=model)

    # --- Sample atom types on the fixed grid ---
    written = []
    idx = 0
    for conditioning_data, mask in condition_loader:
        conditioning_data = conditioning_data.to(get_device())
        sample, mean = sampler.sample(conditioning_data, mask)
        # `mean` is the final denoised batch (no noise on last step).
        result = mean.to("cpu")
        an = result["atomic_numbers"].reshape(-1).numpy()
        num_atoms = result["num_atoms"].reshape(-1).numpy()
        offsets = np.concatenate([[0], np.cumsum(num_atoms[:-1])])
        for i, (o, n) in enumerate(zip(offsets, num_atoms)):
            written.append(_build_and_write(an[o : o + n], out_path, idx))
            idx += 1

    print(f"[done] wrote {len(written)} structures to {out_path}")
    return written


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()
