#!/usr/bin/env python3
"""
Compute OER reaction energy steps (delta_E1–delta_E4) from DFT total energies,
using H2O and H2 as reference species (energy-based, not free energy).

Formulas:
  delta_E1 = E_HO  - E_M    - (E_H2O - 0.5 * E_H2)
  delta_E2 = E_O   - E_HO   - (E_H2O -     E_H2)
  delta_E3 = E_HOO - E_O    - (E_H2O -     E_H2)
  delta_E4 = E_M + E_H2O - E_HOO - 0.5 * E_H2

Supports both HOO and HOO2 as *OOH intermediates.
Duplicate (element, phase) entries are resolved by keeping the LAST occurrence.

Usage:
  python oer_deltaE.py energy.txt --out oer.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute OER energy steps (delta_E) using DFT total energies."
    )
    p.add_argument("energy_file", type=Path, help="Input table with columns: element, phase, <energy>")
    p.add_argument("--energy-column", default="TOTEN(eV)",
                   help="Name of the energy column (e.g., 'TOTEN(eV)')")
    p.add_argument("--sep", default=None,
                   help="CSV separator (auto-detected if not specified)")
    p.add_argument("--out", default="results.csv",
                   help="Output CSV filename")
    p.add_argument("--E_H2O", default=-14.21651, type=float,
                   help="Total energy of isolated H2O molecule (eV)")
    p.add_argument("--E_H2", default=-6.77026, type=float,
                   help="Total energy of isolated H2 molecule (eV)")
    return p.parse_args()

def main():
    args = parse_args()
    energy_file = args.energy_file

    if not energy_file.is_file():
        raise SystemExit(f"Error: File '{energy_file}' does not exist.")

    # Read input
    try:
        df = pd.read_csv(energy_file, sep=args.sep, engine="python")
    except Exception as e:
        raise SystemExit(f"Error reading file: {e}")

    df.columns = [c.strip() for c in df.columns]
    required_cols = {"element", "phase", args.energy_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Handle duplicates: keep LAST occurrence for each (element, phase)
    df = df.drop_duplicates(subset=["element", "phase"], keep="last")

    # Pivot to wide format
    wide = df.pivot_table(
        index="element",
        columns="phase",
        values=args.energy_column,
        aggfunc="last"  # redundant after drop_duplicates, but safe
    )

    # Support HOO2 as fallback for HOO
    if "HOO" not in wide.columns and "HOO2" in wide.columns:
        wide["HOO"] = wide["HOO2"]

    # Ensure all needed columns exist
    for col in ["M", "HO", "O", "HOO", "HOO2"]:
        if col not in wide.columns:
            wide[col] = np.nan

    # Rename to canonical energy names
    wide = wide.rename(columns={
        "M": "E_M", "HO": "E_HO", "O": "E_O", "HOO": "E_HOO", "HOO2": "E_HOO2"
    })

    # Filter valid elements: must have M, HO, O, and (HOO or HOO2)
    core_mask = (
        wide[["E_M", "E_HO", "E_O"]].notna().all(axis=1) &
        (wide["E_HOO"].notna() | wide["E_HOO2"].notna())
    )
    core = wide[core_mask].copy()
    all_elements = set(wide.index)
    skipped = all_elements - set(core.index)
    if skipped:
        print(f"⚠️ Skipped elements (missing intermediates): {sorted(skipped)}", file=sys.stderr)

    if core.empty:
        raise SystemExit("No valid elements found with required intermediates (M, HO, O, and HOO/HOO2).")

    E_H2O = args.E_H2O
    E_H2 = args.E_H2

    results = []
    for elem, row in core.iterrows():
        out = {"element": elem}

        # Store raw energies
        for key in ["E_M", "E_HO", "E_O", "E_HOO", "E_HOO2"]:
            val = row.get(key, np.nan)
            if pd.notna(val):
                out[key] = val

        # Common delta_E1 and delta_E2 (same for both paths)
        E_M, E_HO, E_O = row["E_M"], row["E_HO"], row["E_O"]
        dE1 = E_HO - E_M - (E_H2O - 0.5 * E_H2)
        dE2 = E_O - E_M - (E_H2O - E_H2)
        out.update({"delta_E1(eV)": dE1, "delta_E2(eV)": dE2})

        # Pathway 1: HOO
        if pd.notna(row["E_HOO"]):
            E_HOO = row["E_HOO"]
            dE3_HOO = E_HOO - E_M - (2*E_H2O - 1.5*E_H2)
            dE4_HOO = E_M + E_H2O - E_HOO - 0.5 * E_H2
            dE_HOO_minus_HO = dE3_HOO - dE1
            out.update({
                "delta_E3_HOO(eV)": dE3_HOO,
                "delta_E4_HOO(eV)": dE4_HOO,
                "delta_E_HOO−HO(eV)": dE_HOO_minus_HO,
            })

        # Pathway 2: HOO2
        if pd.notna(row["E_HOO2"]):
            E_HOO2 = row["E_HOO2"]
            dE3_HOO2 = E_HOO2 - E_M - (2*E_H2O - 1.5*E_H2)
            dE4_HOO2 = E_M + E_H2O - E_HOO2 - 0.5 * E_H2
            dE_HOO2_minus_HO = dE3_HOO2 - dE1
            out.update({
                "delta_E3_HOO2(eV)": dE3_HOO2,
                "delta_E4_HOO2(eV)": dE4_HOO2,
                "delta_E_HOO2−HO(eV)": dE_HOO2_minus_HO,
            })

        results.append(out)

    # Build output DataFrame
    res = pd.DataFrame(results)

    # Define column order for clean output
    col_order = ["element", "E_M", "E_HO", "E_O"]
    if "E_HOO" in res.columns:
        col_order.append("E_HOO")
    if "E_HOO2" in res.columns:
        col_order.append("E_HOO2")
    col_order += [
        "delta_E1(eV)", "delta_E2(eV)",
        "delta_E3_HOO(eV)", "delta_E4_HOO(eV)",
        "delta_E3_HOO2(eV)", "delta_E4_HOO2(eV)",
        "delta_E_HOO−HO(eV)", "delta_E_HOO2−HO(eV)"
    ]
    col_order = [c for c in col_order if c in res.columns]

    res = res[col_order]
    res.to_csv(args.out, index=False)

    # Print to terminal
    display_cols = [c for c in [
        "element", "delta_E1(eV)", "delta_E2(eV)",
        "delta_E3_HOO(eV)", "delta_E4_HOO(eV)",
        "delta_E3_HOO2(eV)", "delta_E4_HOO2(eV)",
        "delta_E_HOO−HO(eV)", "delta_E_HOO2−HO(eV)"
    ] if c in res.columns]

    with pd.option_context(
        "display.max_rows", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format
    ):
        print(res[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()