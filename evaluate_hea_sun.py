#!/usr/bin/env python
"""Evaluate structural U/N and composition-screened S/U/N metrics for HEA CIFs.

The stability screen intentionally follows the four criteria used by
``hea_dataset.py`` in HEA-tools.  Uniqueness and novelty use the ordered
structure-matching convention from MatterGen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


R_GAS = 8.314  # J mol^-1 K^-1
ENTROPY_MIN_OVER_R = 1.5
SIZE_DELTA_MAX_PCT = 6.6
HMIX_MIN_KJ_MOL = -15.0
HMIX_MAX_KJ_MOL = 5.0
VEC_MIN = 8.0

ATOMIC_RADII_PM = {
    "Fe": 126.0,
    "Co": 125.0,
    "Ni": 124.0,
    "Cu": 128.0,
    "Zn": 134.0,
    "Ga": 135.0,
    "In": 167.0,
    "Mo": 139.0,
    "W": 139.0,
    "Sn": 151.0,
}

VALENCE_ELECTRONS = {
    "Fe": 8.0,
    "Co": 9.0,
    "Ni": 10.0,
    "Cu": 11.0,
    "Zn": 12.0,
    "Ga": 3.0,
    "In": 3.0,
    "Mo": 6.0,
    "W": 6.0,
    "Sn": 4.0,
}

# Binary mixing enthalpies [kJ/mol], copied from HEA-tools/hea_dataset.py.
BINARY_HMIX = {
    ("Co", "Fe"): -0.06904243624999928,
    ("Co", "Ni"): -0.02639796000000061,
    ("Co", "Cu"): 0.05296564250000024,
    ("Co", "Zn"): -0.06440434124999994,
    ("Co", "Ga"): -0.2825187887500009,
    ("Co", "In"): -0.03301798562499947,
    ("Co", "Mo"): -0.044348996250000994,
    ("Co", "W"): -0.08396261250000059,
    ("Co", "Sn"): -0.14228259083333347,
    ("Fe", "Ni"): -0.09145708749999937,
    ("Cu", "Fe"): 0.06767582333333325,
    ("Fe", "Zn"): -0.03887403857142857,
    ("Fe", "Ga"): -0.23574756187500068,
    ("Fe", "In"): 0.08910415999999977,
    ("Fe", "Mo"): -0.0030923849999998274,
    ("Fe", "W"): -0.023666749166667483,
    ("Fe", "Sn"): -0.0393573491666667,
    ("Cu", "Ni"): -0.0018628412500003577,
    ("Ni", "Zn"): -0.2547110099999994,
    ("Ga", "Ni"): -0.4091984924999991,
    ("In", "Ni"): -0.1919506283333329,
    ("Mo", "Ni"): -0.0923721724999993,
    ("Ni", "W"): -0.10733122500000025,
    ("Ni", "Sn"): -0.2849230889999994,
    ("Cu", "Zn"): -0.11121758769230758,
    ("Cu", "Ga"): -0.11059130346153814,
    ("Cu", "In"): -0.015922263333333575,
    ("Cu", "Mo"): 0.07922715333333367,
    ("Cu", "W"): 0.1272672366666671,
    ("Cu", "Sn"): -0.05965081250000015,
    ("Ga", "Zn"): 0.014705678750000098,
    ("In", "Zn"): 0.01643934833333353,
    ("Mo", "Zn"): -0.04813484562499992,
    ("W", "Zn"): 0.05189458250000012,
    ("Sn", "Zn"): 0.02516417500000001,
    ("Ga", "In"): 0.02130772708333299,
    ("Ga", "Mo"): -0.17489470199999957,
    ("Ga", "W"): -0.08640789800000022,
    ("Ga", "Sn"): 0.035810431666666726,
    ("In", "Mo"): 0.022434562000000113,
    ("In", "W"): 0.1469121500000007,
    ("In", "Sn"): 0.001599915625000392,
    ("Mo", "W"): -0.017974743333333265,
    ("Mo", "Sn"): -0.03018114499999945,
    ("Sn", "W"): 0.16359640333333422,
}


def ordered_structure_matcher() -> StructureMatcher:
    """Return MatterGen's DefaultOrderedStructureMatcher configuration."""
    return StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )


def cif_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.cif"))
    if not paths:
        raise ValueError(f"No CIF files found in {directory}")
    return paths


def load_cifs(paths: Iterable[Path]) -> list[Structure]:
    structures = []
    for path in paths:
        try:
            structures.append(Structure.from_file(path))
        except Exception as exc:
            raise ValueError(f"Could not parse CIF {path}: {exc}") from exc
    return structures


def integer_composition(structure: Structure) -> tuple[tuple[str, int], ...]:
    amounts = structure.composition.get_el_amt_dict()
    result = []
    for element, amount in amounts.items():
        rounded = round(amount)
        if not math.isclose(amount, rounded, abs_tol=1e-6):
            raise ValueError(
                f"Expected an ordered integer composition, got {element}={amount}"
            )
        result.append((element, int(rounded)))
    return tuple(sorted(result))


def composition_label(composition: tuple[tuple[str, int], ...]) -> str:
    return "".join(f"{element}{count}" for element, count in composition)


def stable_screen(composition: tuple[tuple[str, int], ...]) -> dict[str, Any]:
    total = sum(count for _, count in composition)
    fractions = {element: count / total for element, count in composition}

    s_mix_over_r = -sum(x * math.log(x) for x in fractions.values() if x > 0)
    entropy_ok = s_mix_over_r > ENTROPY_MIN_OVER_R

    missing_radii = sorted(set(fractions) - set(ATOMIC_RADII_PM))
    if missing_radii:
        delta_pct = None
        size_ok = False
    else:
        mean_radius = sum(fractions[e] * ATOMIC_RADII_PM[e] for e in fractions)
        delta_pct = 100 * math.sqrt(
            sum(
                fractions[e] * (1 - ATOMIC_RADII_PM[e] / mean_radius) ** 2
                for e in fractions
            )
        )
        size_ok = delta_pct <= SIZE_DELTA_MAX_PCT

    h_mix = 0.0
    missing_pairs = []
    elements = list(fractions)
    for i, element_i in enumerate(elements):
        for element_j in elements[i + 1 :]:
            pair = tuple(sorted((element_i, element_j)))
            if pair not in BINARY_HMIX:
                missing_pairs.append("-".join(pair))
            else:
                h_mix += (
                    4
                    * BINARY_HMIX[pair]
                    * fractions[element_i]
                    * fractions[element_j]
                )
    hmix_ok = not missing_pairs and HMIX_MIN_KJ_MOL <= h_mix <= HMIX_MAX_KJ_MOL

    missing_vec = sorted(set(fractions) - set(VALENCE_ELECTRONS))
    if missing_vec:
        vec = None
        vec_ok = False
    else:
        vec = sum(fractions[e] * VALENCE_ELECTRONS[e] for e in fractions)
        vec_ok = vec >= VEC_MIN

    stable = entropy_ok and size_ok and hmix_ok and vec_ok
    return {
        "s_mix_over_R": s_mix_over_r,
        "s_mix_J_mol_K": s_mix_over_r * R_GAS,
        "entropy_ok": entropy_ok,
        "delta_pct": delta_pct,
        "size_ok": size_ok,
        "h_mix_kJ_mol": None if missing_pairs else h_mix,
        "hmix_ok": hmix_ok,
        "vec": vec,
        "vec_ok": vec_ok,
        "stable": stable,
        "missing_radii": missing_radii,
        "missing_hmix_pairs": missing_pairs,
        "missing_vec": missing_vec,
    }


def structural_unique_mask(
    structures: list[Structure], matcher: StructureMatcher
) -> list[bool]:
    """Mark the first representative of each matching group as unique."""
    representatives: dict[str, list[Structure]] = defaultdict(list)
    mask = []
    for structure in structures:
        key = structure.composition.reduced_formula
        is_unique = not any(matcher.fit(structure, other) for other in representatives[key])
        mask.append(is_unique)
        if is_unique:
            representatives[key].append(structure)
    return mask


def structural_novelty(
    generated: list[Structure],
    references: list[Structure],
    reference_paths: list[Path],
    matcher: StructureMatcher,
) -> tuple[list[bool], list[list[str]]]:
    references_by_formula: dict[str, list[tuple[Structure, Path]]] = defaultdict(list)
    for structure, path in zip(references, reference_paths):
        references_by_formula[structure.composition.reduced_formula].append((structure, path))

    novel_mask = []
    matches = []
    for structure in generated:
        candidates = references_by_formula.get(structure.composition.reduced_formula, [])
        matched_paths = [
            path.name for reference, path in candidates if matcher.fit(structure, reference)
        ]
        matches.append(matched_paths)
        novel_mask.append(not matched_paths)
    return novel_mask, matches


def fraction(mask: Iterable[bool]) -> float:
    values = list(mask)
    return sum(values) / len(values)


def percent(value: float) -> float:
    return round(100 * value, 6)


def evaluate(train_dir: Path, generated_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_paths = cif_paths(train_dir)
    generated_paths = cif_paths(generated_dir)
    train_structures = load_cifs(train_paths)
    generated_structures = load_cifs(generated_paths)

    matcher = ordered_structure_matcher()
    is_unique = structural_unique_mask(generated_structures, matcher)
    is_novel, reference_matches = structural_novelty(
        generated_structures, train_structures, train_paths, matcher
    )

    train_compositions = {integer_composition(s) for s in train_structures}
    generated_compositions = [integer_composition(s) for s in generated_structures]
    composition_counts = Counter(generated_compositions)
    seen_compositions = set()
    rows = []

    for i, (path, composition) in enumerate(zip(generated_paths, generated_compositions)):
        screen = stable_screen(composition)
        composition_unique = composition not in seen_compositions
        seen_compositions.add(composition)
        composition_novel = composition not in train_compositions
        unique_novel = is_unique[i] and is_novel[i]
        sun = screen["stable"] and unique_novel
        rows.append(
            {
                "file": path.name,
                "composition": composition_label(composition),
                "reduced_formula": generated_structures[i].composition.reduced_formula,
                "is_unique": is_unique[i],
                "is_novel": is_novel[i],
                "is_unique_novel": unique_novel,
                "is_stable": screen["stable"],
                "is_SUN": sun,
                "s_mix_over_R": screen["s_mix_over_R"],
                "s_mix_J_mol_K": screen["s_mix_J_mol_K"],
                "entropy_ok": screen["entropy_ok"],
                "delta_pct": screen["delta_pct"],
                "size_ok": screen["size_ok"],
                "h_mix_kJ_mol": screen["h_mix_kJ_mol"],
                "hmix_ok": screen["hmix_ok"],
                "vec": screen["vec"],
                "vec_ok": screen["vec_ok"],
                "matching_training_files": ";".join(reference_matches[i]),
                "is_composition_unique": composition_unique,
                "is_composition_novel": composition_novel,
                "generated_composition_multiplicity": composition_counts[composition],
                "missing_radii": ";".join(screen["missing_radii"]),
                "missing_hmix_pairs": ";".join(screen["missing_hmix_pairs"]),
                "missing_vec": ";".join(screen["missing_vec"]),
            }
        )

    stable_mask = [row["is_stable"] for row in rows]
    unique_novel_mask = [row["is_unique_novel"] for row in rows]
    sun_mask = [row["is_SUN"] for row in rows]
    comp_unique_mask = [row["is_composition_unique"] for row in rows]
    comp_novel_mask = [row["is_composition_novel"] for row in rows]

    def metric(mask: list[bool]) -> dict[str, float | int]:
        return {"count": sum(mask), "fraction": fraction(mask), "percent": percent(fraction(mask))}

    summary = {
        "inputs": {
            "training_directory": str(train_dir.resolve()),
            "generated_directory": str(generated_dir.resolve()),
            "training_cif_count": len(train_paths),
            "generated_cif_count": len(generated_paths),
        },
        "definitions": {
            "unique": "First representative among generated CIFs under ordered StructureMatcher.",
            "novel": "No ordered StructureMatcher match in the training CIFs.",
            "stable": "All four HEA composition criteria pass.",
            "SUN": "stable AND unique AND novel; denominator is all generated CIFs.",
            "matcher": {
                "ltol": 0.2,
                "stol": 0.3,
                "angle_tol": 5,
                "primitive_cell": True,
                "scale": True,
                "attempt_supercell": False,
                "allow_subset": False,
            },
            "stability_thresholds": {
                "s_mix_over_R": f"> {ENTROPY_MIN_OVER_R}",
                "delta_pct": f"<= {SIZE_DELTA_MAX_PCT}",
                "h_mix_kJ_mol": f"[{HMIX_MIN_KJ_MOL}, {HMIX_MAX_KJ_MOL}]",
                "vec": f">= {VEC_MIN}",
            },
        },
        "structure_metrics": {
            "unique": metric(is_unique),
            "novel": metric(is_novel),
            "unique_novel": metric(unique_novel_mask),
            "stable": metric(stable_mask),
            "SUN": metric(sun_mask),
        },
        "composition_metrics_supplement": {
            "unique": metric(comp_unique_mask),
            "novel": metric(comp_novel_mask),
            "unique_novel": metric(
                [u and n for u, n in zip(comp_unique_mask, comp_novel_mask)]
            ),
        },
        "stability_failures": {
            "entropy": sum(not row["entropy_ok"] for row in rows),
            "atomic_size": sum(not row["size_ok"] for row in rows),
            "mixing_enthalpy": sum(not row["hmix_ok"] for row in rows),
            "VEC": sum(not row["vec_ok"] for row in rows),
        },
    }
    return summary, rows


def write_results(
    summary: dict[str, Any], rows: list[dict[str, Any]], json_path: Path, csv_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("datasets/randomStructures"))
    parser.add_argument("--generated-dir", type=Path, default=Path("outputs/hea"))
    parser.add_argument(
        "--json-output", type=Path, default=Path("outputs/hea_sun_summary.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("outputs/hea_sun_per_structure.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, rows = evaluate(args.train_dir, args.generated_dir)
    write_results(summary, rows, args.json_output, args.csv_output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.json_output} and {args.csv_output}")


if __name__ == "__main__":
    main()
