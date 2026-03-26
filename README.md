# Single-Atom Computational Materials Toolkit

This repository collects small Python utilities for computational materials science workflows around single-atom and oxide catalyst studies. The codebase is script-oriented rather than packaged as a single library, and most tools are intended to be run directly inside their own folders.

The previous version is the same user's [geo_angle_distance](https://github.com/alekseyevich180/geo_angle_distance) repository.

The current repository focuses on four areas:

- VASP workflow helpers for generating `INCAR`/`POTCAR` inputs and post-processing adsorption-energy data.
- Octahedral structure manipulation for angle analysis and geometry rotation based on `POSCAR`-style files.
- Volcano-plot generation for catalytic activity analysis from tabulated energetic descriptors.
- Gaussian process regression experiments for fitting structure-property trends.

## Repository Layout

### `supercomputer_scripts/`
Utilities for VASP input preparation and OER-related energy analysis.

- `make_incar.py`: builds `INCAR` files from a shared `par` configuration and the element counts in a `POSCAR`.
- `make_potcar.py`: helper for POTCAR-related setup.
- `oer_overpotential.py`: computes OER reaction-energy steps from a table of total energies.
- `run.sh`: example batch script for job execution.
- `Fe_pv_3_test/`: sample calculation layout with `POSCAR`, `POTCAR`, and a local `incar_generator.py`.

Typical usage:

```bash
python supercomputer_scripts/oer_overpotential.py energy.csv --out results.csv
python supercomputer_scripts/make_incar.py --dir Fe_pv_3
```

### `octachedral_rot/`
Tools for octahedral geometry analysis and controlled rotation of oxygen environments.

- `rot_reo.py`: rotates selected O atoms in octahedral structures using command-line arguments or a local `par` file.
- `angle_get.py`: extracts structural angle information.
- `share/`: reusable helpers for angle calculation, atom reordering, SSH/job helpers, and geometry utilities.
- `*.vasp`: example structures for metals such as Co, Cu, Fe, and Ni.

Typical usage:

```bash
python octachedral_rot/rot_reo.py --poscar octachedral_rot/Fe.vasp --angle-start 0 --angle-end 90 --angle-step 5
```

### `volcano_plots/`
Plotting scripts for descriptor-activity analysis and volcano-curve generation.

- `plot_volcano.py`: loads tabulated adsorption-energy data, filters entries, derives reaction descriptors, and generates volcano plots.
- `plot_potential.py`, `combine_potential.py`, `plot_all.py`, `plot_manager.py`: related plotting workflows.
- `plot_config.py`: central plotting configuration.
- `*.dat`: example input datasets for different XC functionals and correction schemes.

Typical usage:

```bash
python volcano_plots/plot_volcano.py
```

### `ml/`
Machine-learning experiments for Gaussian process regression and trend fitting.

- `Scikit_framework/`: scikit-learn based GPR scripts.
- `Gpy_framework/`: GPy-based exploratory notebooks/scripts.
- `gaussian_pure/`: lightweight GPR test setup.
- `datafit_test/`: per-element fitting cases (`Co`, `Cu`, `Fe`, `Ir`, `Mn`, `Ni`, `Ru`, `Sn`, `Ti`) with local `gpr_config.py`, data files, test scripts, and generated figures.

Typical usage:

```bash
python ml/datafit_test/Cu/test.py
python ml/Scikit_framework/gaussian.py
```

## Configuration Conventions

Different subprojects use different configuration styles:

- `par`: key-value configuration files used by several VASP and structure-generation scripts.
- `gpr_config.py`: Python configuration modules for GPR experiments.
- `plot_config.py`: centralized visual and filtering settings for volcano plots.

Most scripts assume they are executed from their own directory or with paths adjusted accordingly.

## Dependencies

The exact requirements depend on which scripts you use, but the repository commonly relies on:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `pymatgen`
- `GPy` for the `ml/Gpy_framework` experiments

You can install a practical baseline with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn pymatgen
```

Add `GPy` only if you need the GPy-based experiments.

## Notes

- This repository is not yet organized as a single installable Python package.
- Some scripts and inline comments are still bilingual or Chinese-language; the execution logic is unchanged.
- Several folders include generated figures or test data alongside source scripts.
