#!/usr/bin/env python3
# %%
"""
Batch replace the metal atom in VASP structures and relax with PFP.

This file is intentionally written in notebook-friendly cells.  In Jupyter,
run each ``# %%`` block in order, then call ``run_batch(CONFIG)``.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# %%
DEFAULT_ELEMENT_LIST = """
Ag Au Al Bi Cd Co_pv Cr_pv Cu Fe_pv Ga Hf Hg In_d Ir Mn_pv Mo_sv Nb_pv
Ni_pv Os_pv Pb Pd Pt Rh_pv Ru_pv Sb Sc_sv Sn_d Sr_sv Ta Tc_pv Ti_pv
V_sv W_sv Y_sv Zn Zr_sv
"""

DEFAULT_STRUCTURES = ("M.vasp", "O.vasp", "HO.vasp", "HOO.vasp")


class BatchConfig:
    def __init__(
        self,
        input_dir=Path("."),
        output_dir=Path("."),
        structure_files=DEFAULT_STRUCTURES,
        from_element="Sn",
        elements=(),
        element_list_file=None,
        fmax=0.05,
        steps=None,
        optimizer="LBFGS",
        model_version="v8.0.0",
        calc_mode="CRYSTAL_U0",
        overwrite=False,
        write_logs=True,
        continue_on_error=True,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.structure_files = tuple(structure_files)
        self.from_element = from_element
        self.elements = tuple(elements)
        self.element_list_file = Path(element_list_file) if element_list_file is not None else None
        self.fmax = fmax
        self.steps = steps
        self.optimizer = optimizer
        self.model_version = model_version
        self.calc_mode = calc_mode
        self.overwrite = overwrite
        self.write_logs = write_logs
        self.continue_on_error = continue_on_error


# %%
def normalize_element_name(name: str) -> str:
    """Convert VASP PAW labels such as Fe_pv, Mo_sv, Sn_d into element symbols."""
    token = name.strip()
    if not token:
        raise ValueError("empty element name")
    symbol = re.sub(r"_[A-Za-z0-9]+$", "", token)
    return symbol[:1].upper() + symbol[1:].lower()


def parse_element_text(text: str) -> Tuple[str, ...]:
    elements = []  # type: List[str]
    for raw in re.split(r"[\s,]+", text.strip()):
        if raw:
            element = normalize_element_name(raw)
            if element not in elements:
                elements.append(element)
    return tuple(elements)


def load_elements(config: BatchConfig) -> Tuple[str, ...]:
    if config.elements:
        return tuple(normalize_element_name(x) for x in config.elements)
    if config.element_list_file is not None:
        return parse_element_text(config.element_list_file.read_text())
    return parse_element_text(DEFAULT_ELEMENT_LIST)


def validate_inputs(config: BatchConfig) -> None:
    if not config.structure_files:
        raise ValueError("No structure files were specified.")
    missing = [name for name in config.structure_files if not (config.input_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing VASP template files in {config.input_dir}: {joined}")


# %%
def _load_matlantis_classes():
    try:
        from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
        from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
    except ImportError as exc:
        raise ImportError(
            "pfp_api_client is not available. Run this script in a Matlantis environment "
            "or activate a Python environment where the Matlantis/PFP client is installed "
            "and authenticated."
        ) from exc

    return ASECalculator, Estimator, EstimatorCalcMode


def _resolve_estimator_calc_mode(calc_mode):
    _, _, EstimatorCalcMode = _load_matlantis_classes()
    if isinstance(calc_mode, EstimatorCalcMode):
        return calc_mode

    normalized = str(calc_mode).upper()
    if hasattr(EstimatorCalcMode, normalized):
        return getattr(EstimatorCalcMode, normalized)

    available = sorted(name for name in dir(EstimatorCalcMode) if name.isupper())
    raise ValueError(
        f"Unsupported Matlantis/PFP calc_mode {calc_mode!r}. "
        f"Available modes include: {', '.join(available)}"
    )


def build_calculator(calc_mode, model_version: str):
    """Build the Matlantis/PFP ASE calculator used for all relaxations.

    This follows the calculator definition used in the reference script:
    ``Estimator(calc_mode=calc_mode, model_version=model_version)`` wrapped by
    ``ASECalculator``.

    This must be executed inside a Matlantis/PFP-enabled Python environment.
    The default ``CRYSTAL_U0`` mode is suitable for periodic crystal/slab
    structures written as VASP POSCAR/CONTCAR files.
    """
    ASECalculator, Estimator, _ = _load_matlantis_classes()
    calc_mode = _resolve_estimator_calc_mode(calc_mode)
    estimator = Estimator(calc_mode=calc_mode, model_version=model_version)
    return ASECalculator(estimator)


def get_optimizer_class(name: str):
    try:
        from ase.optimize import BFGS, FIRE, LBFGS
    except ImportError as exc:
        raise ImportError("ASE is required. Install ase in the active Python environment.") from exc

    optimizers = {
        "BFGS": BFGS,
        "FIRE": FIRE,
        "LBFGS": LBFGS,
    }
    key = name.upper()
    if key not in optimizers:
        raise ValueError(f"Unsupported optimizer {name!r}; choose one of {sorted(optimizers)}")
    return optimizers[key]


# %%
def replace_element(atoms, from_element: str, to_element: str):
    symbols = atoms.get_chemical_symbols()
    replaced = 0
    for index, symbol in enumerate(symbols):
        if symbol == from_element:
            symbols[index] = to_element
            replaced += 1
    if replaced == 0:
        raise ValueError(f"Structure does not contain {from_element}")
    atoms.set_chemical_symbols(symbols)
    return atoms, replaced


def relax_one_structure(
    template_path: Path,
    output_path: Path,
    calculator,
    config: BatchConfig,
    element: str,
) -> Dict[str, Any]:
    try:
        from ase.io import read, write
    except ImportError as exc:
        raise ImportError("ASE is required. Install ase in the active Python environment.") from exc

    atoms = read(template_path, format="vasp")
    atoms, replaced = replace_element(atoms, config.from_element, element)
    atoms.calc = calculator

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    log_file = str(output_path.parent / f"{stem}.opt.log") if config.write_logs else None
    trajectory = str(output_path.parent / f"{stem}.traj") if config.write_logs else None

    optimizer_cls = get_optimizer_class(config.optimizer)
    optimizer = optimizer_cls(atoms, logfile=log_file, trajectory=trajectory)
    if config.steps is None:
        optimizer_converged = bool(optimizer.run(fmax=config.fmax))
    else:
        optimizer_converged = bool(optimizer.run(fmax=config.fmax, steps=config.steps))

    energy = atoms.get_potential_energy()
    max_force = max((sum(force_i * force_i for force_i in force) ** 0.5) for force in atoms.get_forces())
    write(output_path, atoms, format="vasp", vasp5=True, direct=True, sort=False)

    return {
        "element": element,
        "structure": template_path.name,
        "output": str(output_path),
        "replaced_atoms": replaced,
        "energy_eV": energy,
        "max_force_eV_A": max_force,
        "converged": optimizer_converged and max_force <= config.fmax,
        "status": "ok",
        "error": "",
    }


def failed_result(element: str, structure_name: str, output_path: Path, error: Exception) -> Dict[str, Any]:
    return {
        "element": element,
        "structure": structure_name,
        "output": str(output_path),
        "replaced_atoms": "",
        "energy_eV": "",
        "max_force_eV_A": "",
        "converged": False,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }


def run_batch(config: BatchConfig) -> List[Dict[str, Any]]:
    config.input_dir = Path(config.input_dir)
    config.output_dir = Path(config.output_dir)
    validate_inputs(config)

    elements = load_elements(config)
    calculator = build_calculator(config.calc_mode, str(config.model_version))
    results = []  # type: List[Dict[str, Any]]

    for element in elements:
        element_dir = config.output_dir / element
        print(f"[ELEMENT] {element}")
        element_dir.mkdir(parents=True, exist_ok=True)
        for structure_name in config.structure_files:
            template_path = config.input_dir / structure_name
            output_path = element_dir / structure_name
            if output_path.exists() and not config.overwrite:
                print(f"  [SKIP] {structure_name}: {output_path} exists. Use --overwrite to regenerate.")
                continue

            try:
                result = relax_one_structure(template_path, output_path, calculator, config, element)
            except Exception as exc:
                result = failed_result(element, structure_name, output_path, exc)
                results.append(result)
                print(f"  [FAIL] {structure_name}: {result['error']}")
                write_summary(config.output_dir / "pfp_optimization_summary.csv", results)
                if not config.continue_on_error:
                    raise
                continue

            results.append(result)
            print(
                f"  [DONE] {structure_name}: "
                f"E={result['energy_eV']:.6f} eV, "
                f"Fmax={result['max_force_eV_A']:.4f} eV/A"
            )
            write_summary(config.output_dir / "pfp_optimization_summary.csv", results)

    write_summary(config.output_dir / "pfp_optimization_summary.csv", results)
    return results


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# %%
# Jupyter example:
#
# CONFIG = BatchConfig(
#     input_dir=Path("."),
#     output_dir=Path("."),
#     from_element="Sn",
#     fmax=0.05,
#     steps=None,
#     optimizer="LBFGS",
#     model_version="v3.0.0",
#     calc_mode="CRYSTAL_U0",
#     overwrite=False,
# )
# results = run_batch(CONFIG)


# %%
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--structure-files", nargs="*", default=DEFAULT_STRUCTURES)
    parser.add_argument("--from-element", default="Sn")
    parser.add_argument("--elements", nargs="*", default=())
    parser.add_argument("--element-list-file", type=Path)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--optimizer", default="LBFGS", choices=("FIRE", "BFGS", "LBFGS"))
    parser.add_argument("--model-version", default=os.environ.get("MATLANTIS_MODEL_VERSION", "v3.0.0"))
    parser.add_argument("--calc-mode", default=os.environ.get("MATLANTIS_CALC_MODE", "CRYSTAL_U0"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-logs", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def _running_from_ipykernel() -> bool:
    return Path(sys.argv[0]).name == "ipykernel_launcher.py"


def main(argv=None):
    if argv is None and _running_from_ipykernel():
        argv = []
    args = build_arg_parser().parse_args(argv)
    config = BatchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        structure_files=tuple(args.structure_files),
        from_element=args.from_element,
        elements=tuple(args.elements),
        element_list_file=args.element_list_file,
        fmax=args.fmax,
        steps=args.steps,
        optimizer=args.optimizer,
        model_version=args.model_version,
        calc_mode=args.calc_mode,
        overwrite=args.overwrite,
        write_logs=not args.no_logs,
        continue_on_error=not args.stop_on_error,
    )
    return run_batch(config)


if __name__ == "__main__":
    main()
