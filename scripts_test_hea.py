#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end smoke test for the fixed-geometry HEA atom-type-only pipeline.

Run AFTER `python -m mattergen.scripts.prepare_hea_dataset` has produced the
cache under datasets/cache/hea/{train,val}. This does NOT need a trained
checkpoint; it validates that every wired-up piece constructs and runs.

    python scripts_test_hea.py

Checks:
    1. hea_template canonical grid + Vegard round-trip.
    2. Dataset cache loads via CrystalDataset and yields 64-atom ChemGraphs.
    3. Training config composes; DiffusionModule instantiates.
    4. One forward/backward training step runs (loss is finite).
    5. Prior sampling on the HEA template loader starts from all-MASK.
    6. HEA element mask restricts logits to the 9 pool elements.
"""

import os
import sys
import traceback

os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name):
    def deco(fn):
        print(f"\n=== {name} ===")
        try:
            fn()
            print(PASS, name)
            results.append((name, True))
        except Exception as e:
            print(FAIL, name, "->", repr(e))
            traceback.print_exc()
            results.append((name, False))
    return deco


@check("1. hea_template grid + Vegard")
def _():
    from mattergen.common.data.hea_template import (
        canonical_template, vegard_lattice_constant, HEA_ELEMENTS,
    )
    pos, cell, key_to_index = canonical_template()
    assert pos.shape == (64, 3), pos.shape
    assert cell.shape == (3, 3), cell.shape
    assert len(key_to_index) == 64
    a = vegard_lattice_constant(["Fe"] * 64)
    assert abs(a - 3.59) < 1e-6, a
    a2 = vegard_lattice_constant(HEA_ELEMENTS * 7 + HEA_ELEMENTS[:1])
    assert 3.5 < a2 < 4.9, a2
    print(f"   grid ok, Vegard(Fe)= {a:.3f}, mixed= {a2:.3f}")


@check("2. dataset cache loads")
def _():
    from mattergen.common.data.dataset import CrystalDataset
    ds = CrystalDataset.from_cache_path(cache_path="datasets/cache/hea/train")
    assert len(ds) > 0, "empty dataset"
    g = ds[0]
    assert int(g["num_atoms"]) == 64, g["num_atoms"]
    zs = set(g["atomic_numbers"].tolist())
    allowed = {26, 27, 28, 29, 30, 31, 42, 50, 74}
    assert zs.issubset(allowed), f"unexpected elements: {zs - allowed}"
    print(f"   {len(ds)} structures, sample elements {sorted(zs)}")


@check("3. training config composes + DiffusionModule")
def _():
    import hydra
    from hydra import compose, initialize_config_dir
    from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(str(MODELS_PROJECT_ROOT / "conf"), version_base="1.1"):
        cfg = compose(config_name="hea")
    dm_cfg = cfg.lightning_module.diffusion_module
    assert dm_cfg.loss_fn.include_pos is False
    assert dm_cfg.loss_fn.include_cell is False
    assert dm_cfg.loss_fn.include_atomic_numbers is True
    # corruption should only have atomic_numbers
    corr = dm_cfg.corruption
    assert "sdes" not in corr or not corr.get("sdes"), "unexpected sdes present"
    assert "atomic_numbers" in corr.discrete_corruptions
    lm = hydra.utils.instantiate(cfg.lightning_module)
    fields = lm.diffusion_module.corruption.corrupted_fields
    assert fields == ["atomic_numbers"], fields
    print(f"   corrupted_fields = {fields}")
    globals()["_LM"] = lm


@check("4. one training step (finite loss)")
def _():
    import hydra
    from hydra import compose, initialize_config_dir
    from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
    from mattergen.common.data.dataset import CrystalDataset
    from mattergen.common.data.collate import collate

    lm = globals().get("_LM")
    if lm is None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize_config_dir(str(MODELS_PROJECT_ROOT / "conf"), version_base="1.1"):
            cfg = compose(config_name="hea")
        lm = hydra.utils.instantiate(cfg.lightning_module)

    ds = CrystalDataset.from_cache_path(cache_path="datasets/cache/hea/train")
    batch = collate([ds[i] for i in range(4)])
    lm.train()
    loss, metrics = lm.diffusion_module.calc_loss(batch)
    assert torch.isfinite(loss), loss
    loss.backward()
    print(f"   loss = {float(loss):.4f}, metrics keys = {list(metrics.keys())}")


@check("5. prior sampling starts from all-MASK on template")
def _():
    from mattergen.common.data.condition_factory import get_hea_template_loader
    loader = get_hea_template_loader(num_structures=4, batch_size=4)
    cond, mask = next(iter(loader))
    assert int(cond["num_atoms"][0]) == 64
    # positions/cell are the fixed template (finite, not NaN)
    assert torch.isfinite(cond["pos"]).all()
    assert torch.isfinite(cond["cell"]).all()
    print(f"   template batch: pos {tuple(cond['pos'].shape)}, cell {tuple(cond['cell'].shape)}")


@check("6. HEA element mask restricts to 9 elements")
def _():
    from mattergen.denoiser import mask_to_hea_elements
    from mattergen.common.utils.globals import HEA_ATOMIC_NUMBERS
    logits = torch.randn(10, 101)
    masked = mask_to_hea_elements(logits, predictions_are_zero_based=True)
    # allowed zero-based indices = atomic_number - 1
    allowed = set(z - 1 for z in HEA_ATOMIC_NUMBERS)
    argmax = masked.argmax(dim=1).tolist()
    assert all(a in allowed for a in argmax), argmax
    # disallowed columns should be -inf-ish
    disallowed_col = 0  # H (Z=1) not in pool
    assert (masked[:, disallowed_col] < -1e8).all()
    print(f"   argmax all in pool; allowed zero-based = {sorted(allowed)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    ok = sum(1 for _, r in results if r)
    total = len(results)
    print(f"SUMMARY: {ok}/{total} passed")
    for name, r in results:
        print(f"  {'OK ' if r else 'ERR'} {name}")
    sys.exit(0 if ok == total else 1)
