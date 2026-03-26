# Computational Materials Science Tools

This repository contains a collection of scripts and tools for computational materials science, primarily focused on VASP calculations, structure manipulation, and data analysis using machine learning techniques.

## Directory Structure

### `generate_incar/`
Tools for automatically generating VASP `INCAR` files.
- **`incar_generator.py`**: The main script to generate `INCAR` files based on `POSCAR` composition and a global configuration file (`par`).
    - **Usage**:
      ```bash
      python incar_generator.py --poscar POSCAR > INCAR
      # or
      python incar_generator.py --poscar POSCAR --out INCAR
      ```
    - Supports features like DFT+U, Spin, SOC, and various phase modes (oxide/metal) via the `par` configuration file.

### `octachedral_rot/`
Scripts for manipulating octahedral structures, specifically rotating the O-M-O angle or rotating around a specific axis.
- **`rot_reo.py`**: Rotates an Ir-O6 (or similar) octahedron.
    - **Usage**:
      ```bash
      python rot_reo.py --poscar Ir.vasp --angle-start 0 --angle-end 90 --angle-step 10
      ```
    - Configurable via command line arguments or a `par` file.

### `volcano/`
Tools for generating volcano plots, commonly used in electrocatalysis to analyze activity trends.
- **`plot_volcano.py`**: Reads data files and generates volcano plots (Activity vs. Descriptor).
    - Features include:
        - Filtering data based on specific criteria.
        - Calculating potentials from ΔG values.
        - Fitting trend lines for left and right legs of the volcano.
        - customizable plotting via `CONFIG`.

### `marchine_scripts/`
A collection of scripts for machine learning and data analysis in materials science.
- Includes scripts for:
    - extracting descriptors using `matminer` (`matminer_descriptors_distribute_forgcolab.py`).
    - Gaussian Process Regression (GPR) examples (`dx_gem_singleobjectivebayesopt.py`).
    - Other data analysis demos.

### `octachedral_gaussian/`
Scripts related to Gaussian Process Regression (GPR) and potentially fitting Gaussian functions.
- **`gaussian.py`**: Example script performing GPR on data, including outlier detection and kernel optimization.

### `carbon_chain_rot/`
Tools related to carbon chain structures, possibly including rotation or optimization tasks.
- Contains a `par` file for configuring Bayesian Optimization settings, suggesting it's used for structure optimization or property search.

## Requirements

The scripts in this repository depend on various Python libraries, including:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `matminer` (for some scripts in `marchine_scripts`)
- `pymatgen` (implied by usage of VASP files and matminer)

## Configuration

Many scripts use a `par` file (key=value format) for configuration. This allows for centralized control of parameters without modifying the code.
