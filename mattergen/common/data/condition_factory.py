# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import NumAtomsCrystalDataset
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.transform import SetProperty, Transform
from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_utils import create_chem_graph_from_composition
from mattergen.diffusion.data.batched_data import BatchedData

ConditionLoader = Iterable[tuple[BatchedData, dict[str, torch.Tensor]] | None]


def _collate_fn(
    batch: Sequence[ChemGraph],
    collate_fn: Callable[[Sequence[ChemGraph]], BatchedData],
) -> tuple[BatchedData, None]:
    return collate_fn(batch), None


def get_number_of_atoms_condition_loader(
    num_atoms_distribution: str,
    num_samples: int,
    batch_size: int,
    shuffle: bool = True,
    transforms: list[Transform] | None = None,
    properties: TargetProperty | None = None,
) -> ConditionLoader:
    transforms = transforms or []
    if properties is not None:
        for k, v in properties.items():
            transforms.append(SetProperty(k, v))
    assert (
        num_atoms_distribution in NUM_ATOMS_DISTRIBUTIONS
    ), f"Invalid num_atoms_distribution: {num_atoms_distribution}"
    dataset = NumAtomsCrystalDataset.from_num_atoms_distribution(
        num_atoms_distribution=NUM_ATOMS_DISTRIBUTIONS[num_atoms_distribution],
        num_samples=num_samples,
        transforms=transforms,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=shuffle,
    )


def get_composition_data_loader(
    target_compositions_dict: list[dict[str, float]],
    num_structures_to_generate_per_composition: int,
    batch_size: int,
) -> ConditionLoader:
    """
    Given a list of target compositions, generate a dataset of chemgraphs
    where each chemgraph contains atoms corresponding to the target composition
    without positions or cell information.
    Returns a torch dataloader equipped with the correct collate function containing such dataset.
    """

    dataset_ = []
    for compostion in target_compositions_dict:
        chemgraphs = [
            create_chem_graph_from_composition(compostion)
        ] * num_structures_to_generate_per_composition
        dataset_.extend(chemgraphs)

    dataset = ChemGraphlistDataset(dataset_)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=False,
    )


def get_hea_template_loader(
    num_structures: int,
    batch_size: int,
) -> ConditionLoader:
    """Condition loader for fixed-geometry HEA generation.

    Every conditioning sample carries the *same* canonical DEFAULT_A FCC(111)
    grid (identical positions and cell as the training data). The atom types are
    placeholders that the D3PM prior overwrites with MASK tokens at sampling
    time; only atom types are then denoised, while positions and cell stay fixed.
    """
    from mattergen.common.data.hea_template import canonical_template

    pos_frac, cell, _ = canonical_template()
    pos_t = torch.from_numpy(pos_frac).float()  # (n_sites, 3)
    cell_t = torch.from_numpy(cell).float().unsqueeze(0)  # (1, 3, 3)
    n_sites = pos_t.shape[0]

    def _make_template() -> ChemGraph:
        num_atoms = torch.tensor(n_sites, dtype=torch.long)
        return ChemGraph(
            # Placeholder atom types; replaced by the MASK token in the prior.
            atomic_numbers=torch.zeros(n_sites, dtype=torch.long),
            num_atoms=num_atoms,
            num_nodes=num_atoms,
            pos=pos_t.clone(),
            cell=cell_t.clone(),
        )

    dataset = ChemGraphlistDataset([_make_template() for _ in range(num_structures)])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=False,
    )


class ChemGraphlistDataset(Dataset):
    def __init__(self, data: list[ChemGraph]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> ChemGraph:
        return self.data[index]
