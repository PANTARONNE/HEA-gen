# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Warm-start an atom-only HEA model from the pretrained mattergen_base backbone.

The HEA atom-only model and mattergen_base share the *same* GemNetTDenoiser
backbone (hidden_dim=512, num_blocks=4, mask atom-type diffusion). They differ
only in the corruption process (HEA diffuses atom types only; positions/cell are
frozen) and in the loss - neither of which are network weights. So most of
mattergen_base's state_dict maps 1:1 onto a freshly built atom-only model.

This script:
  1. Builds an atom-only lightning module from the `hea_finetune` config
     (identical architecture to what training will instantiate).
  2. Loads the pretrained mattergen_base checkpoint's state_dict.
  3. Copies over every parameter whose name AND shape match (non-strict), leaving
     any mismatched/missing tensors - e.g. cell/pos output heads that atom-only
     never uses - at their fresh init.
  4. Saves the result as a Lightning-style checkpoint at
     <output_dir>/checkpoints/epoch=0-step=0.ckpt, so that a subsequent
        mattergen-train --config-name=hea_finetune
     with OUTPUT_DIR=<output_dir> and auto_resume=true starts from the warm
     backbone.

Usage
-----
    # 1. Download mattergen_base (HF: microsoft/mattergen) so that you have a
    #    directory containing config.yaml and checkpoints/last.ckpt. Pass EITHER
    #    the .ckpt file directly, or the model dir (last.ckpt is auto-found).
    python inject_pretrained_weights.py \
        --pretrained /path/to/mattergen_base/checkpoints/last.ckpt \
        --output-dir outputs/hea_finetune_run

    # 2. Fine-tune, resuming from the injected checkpoint:
    OUTPUT_DIR=outputs/hea_finetune_run \
        mattergen-train --config-name=hea_finetune trainer.devices=2

Run this on the machine/env where you'll fine-tune (conda env `mattergen`).
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.diffusion.lightning_module import DiffusionLightningModule


def _resolve_ckpt(path: Path) -> Path:
    """Accept either a .ckpt file or a directory; find last.ckpt in the latter."""
    if path.is_file():
        return path
    if path.is_dir():
        candidates = list(path.rglob("last.ckpt")) or list(path.rglob("*.ckpt"))
        assert candidates, f"No .ckpt file found under {path}"
        return candidates[0]
    raise FileNotFoundError(f"Pretrained path does not exist: {path}")


def build_atom_only_module(config_name: str) -> DiffusionLightningModule:
    """Instantiate the lightning module exactly as training would."""
    with initialize_config_dir(str((MODELS_PROJECT_ROOT / "conf").absolute()), version_base="1.1"):
        cfg = compose(config_name=config_name)
    module = instantiate(cfg.lightning_module)
    assert isinstance(module, DiffusionLightningModule)
    return module


def inject(pretrained: Path, output_dir: Path, config_name: str) -> Path:
    module = build_atom_only_module(config_name)
    scratch_dict: OrderedDict = module.state_dict()

    ckpt_path = _resolve_ckpt(pretrained)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pretrained_dict: OrderedDict = ckpt["state_dict"]

    # Copy only tensors present in BOTH and with identical shape. This skips
    # heads the atom-only model doesn't have (or that changed shape) while
    # transferring the whole shared GemNet-T backbone.
    matched, shape_mismatch, only_in_scratch = [], [], []
    for name, tensor in scratch_dict.items():
        if name in pretrained_dict:
            if pretrained_dict[name].shape == tensor.shape:
                scratch_dict[name] = pretrained_dict[name]
                matched.append(name)
            else:
                shape_mismatch.append(name)
        else:
            only_in_scratch.append(name)
    only_in_pretrained = [k for k in pretrained_dict if k not in scratch_dict]

    module.load_state_dict(scratch_dict, strict=True)

    print(f"Pretrained checkpoint : {ckpt_path}")
    print(f"Matched (transferred) : {len(matched)} tensors")
    print(f"Shape mismatch (kept init) : {len(shape_mismatch)} -> {shape_mismatch}")
    print(f"Only in atom-only (kept init) : {len(only_in_scratch)} -> {only_in_scratch}")
    print(f"Only in pretrained (dropped) : {len(only_in_pretrained)} -> {only_in_pretrained}")
    assert matched, "No weights matched - check that the pretrained ckpt is mattergen_base."

    # Write a minimal Lightning checkpoint that trainer.fit(ckpt_path=...) / auto_resume
    # can restore weights from. epoch/global_step start at 0 so the full fine-tuning
    # schedule runs. We intentionally do NOT copy optimizer state - fine-tuning starts
    # a fresh optimizer at the new (small) LR.
    out_ckpt_dir = output_dir / "checkpoints"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_ckpt_dir / "epoch=0-step=0.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "epoch": 0,
            "global_step": 0,
            "pytorch-lightning_version": __import__("pytorch_lightning").__version__,
            "loops": {},
            "callbacks": {},
            "optimizer_states": [],
            "lr_schedulers": [],
        },
        out_path,
    )
    print(f"\nWarm-start checkpoint written to: {out_path}")
    print(f"Fine-tune with:\n  OUTPUT_DIR={output_dir} mattergen-train --config-name={config_name}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pretrained",
        required=True,
        type=Path,
        help="Path to mattergen_base checkpoint (.ckpt file) or its model directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Fine-tuning run dir; the warm-start checkpoint is written to <dir>/checkpoints/.",
    )
    parser.add_argument(
        "--config-name",
        default="hea_finetune",
        help="Hydra config used to build the atom-only model (default: hea_finetune).",
    )
    args = parser.parse_args()
    inject(args.pretrained.expanduser(), args.output_dir.expanduser(), args.config_name)


if __name__ == "__main__":
    main()
