import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_config import CONFIG

# Fallback defaults if CONFIG is missing the keys.
DEFAULT_ALLOWED_LABELS = [
    "Ag", "Au", "Bi", "Cd", "Co",
    "Cr_pv", "Cu", "Fe", "Ga", "Hg", "In_d", "Ir",
    "Mn_pv", "Mo_sv", "Ni", "Pb", "Pd", "Pt", "Rh",
    "Ru_pv", "Sb", "Sn_d", "Sr_sv", "Zn",
]


def load_with_dynamic_columns(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_fields = lines[0].split()
    max_len = max(len(line.split()) for line in lines)
    column_names = header_fields + [f"extra_{i}" for i in range(len(header_fields), max_len)]
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=0,
        names=column_names,
        engine="python",
    )
    return df


def filter_allowed(df: pd.DataFrame) -> pd.DataFrame:
    allowed_labels = CONFIG.get("allowed_elements", DEFAULT_ALLOWED_LABELS)
    allowed_bases = {label.split("_")[0] for label in allowed_labels}
    preferred_variant = {label.split("_")[0]: label for label in allowed_labels if "_" in label}

    df = df.copy()
    df["__base"] = df["element"].astype(str).str.split("_").str[0]
    df = df[df["__base"].isin(allowed_bases)]
    if df.empty:
        return df

    deduped_rows = []
    for base in df["__base"].unique():
        subset = df[df["__base"] == base]
        preferred = preferred_variant.get(base)
        if preferred and preferred in subset["element"].values:
            chosen = subset[subset["element"] == preferred].iloc[0]
        else:
            chosen = subset.iloc[0]
        deduped_rows.append(chosen)

    deduped = pd.DataFrame(deduped_rows).drop(columns="__base")
    deduped["element"] = deduped["element"].astype(str).str.split("_").str[0]
    return deduped


def plot_oer_potential(file_path: str, output_dir: str | None = None):
    """
    Plot OER free-energy steps using columns 7, 8, 9 (1-based) as ΔG1, ΔG2, ΔG3.
    ΔG4 is computed to close the 4e- cycle: ΔG4 = 4*1.23 - (ΔG1+ΔG2+ΔG3).
    """
    try:
        df = load_with_dynamic_columns(file_path)

        # Filter by 13th column (index 12) between 2.0 and 4.0 to stay consistent with other plots
        filter_idx = 12
        if filter_idx >= len(df.columns):
            print(f"{file_path}: not enough columns for filtering (need index 12).")
            return
        filt_col = df.columns[filter_idx]
        df[filt_col] = pd.to_numeric(df[filt_col], errors="coerce")
        df = df[(df[filt_col] >= 2.0) & (df[filt_col] <= 4.0)].copy()
        if df.empty:
            print(f"{file_path}: no rows after filter 2.0-4.0 on {filt_col}.")
            return

        # Keep allowed elements and deduplicate variants
        df = filter_allowed(df)
        if df.empty:
            print(f"{file_path}: no allowed elements after filtering list.")
            return

        col_indices = [6, 7, 8]  # zero-based for columns 7, 8, 9
        if any(idx >= len(df.columns) for idx in col_indices):
            print(f"{file_path}: not enough columns to read ΔG1-3 (need indices 6,7,8).")
            return
        dg_cols = [df.columns[i] for i in col_indices]
        df[dg_cols] = df[dg_cols].apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=dg_cols, inplace=True)
        if df.empty:
            print(f"{file_path}: no valid ΔG1-3 after numeric cleanup.")
            return

        cfg = CONFIG.get("potential", {})
        stage_labels = cfg.get("stage_labels", ["*+2H2O", "OH*", "O*", "OOH*", "O2"])
        ylabel = cfg.get("ylabel", "ΔG (eV)")
        title_prefix = cfg.get("title_prefix", "OER Potential")
        text_fs = cfg.get("text_fontsize", 10)

        target_dir = Path(output_dir) if output_dir else Path(".")
        os.makedirs(target_dir, exist_ok=True)

        for _, row in df.iterrows():
            element = str(row["element"]).split("_")[0]
            dg1, dg2, dg3 = (float(row[c]) for c in dg_cols)
            dg4 = 4 * 1.23 - (dg1 + dg2 + dg3)
            steps = [0.0, dg1, dg1 + dg2, dg1 + dg2 + dg3, dg1 + dg2 + dg3 + dg4]
            deltas = [dg1, dg2, dg3, dg4]
            pds_idx = int(np.argmax(deltas))

            xs = list(range(len(steps)))
            plt.figure(figsize=cfg.get("figsize", (9, 6)))

            # horizontal steps
            for i in range(len(steps) - 1):
                plt.plot(
                    [xs[i], xs[i + 1]],
                    [steps[i], steps[i]],
                    color=cfg.get("line_color", "tab:blue"),
                    linewidth=cfg.get("line_width", 2.0),
                )
                # vertical arrow for each ΔGi
                y0, y1 = steps[i], steps[i + 1]
                arrowprops = dict(
                    arrowstyle="-|>",
                    color=cfg.get("arrow_color", "black"),
                    lw=cfg.get("arrow_width", 1.0),
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=cfg.get("arrow_head_width", 6),
                )
                plt.annotate(
                    "",
                    xy=(xs[i], y1),
                    xytext=(xs[i], y0),
                    arrowprops=arrowprops,
                )
                dy = deltas[i]
                label_lines = [f"ΔG{i+1}", f"{dy:.2f}"]
                if i == pds_idx:
                    label_lines.append("PDS")
                plt.text(
                    xs[i] + 0.05,
                    (y0 + y1) / 2,
                    "\n".join(label_lines),
                    fontsize=text_fs,
                    color=cfg.get("pds_color", "red") if i == pds_idx else "black",
                    va="center",
                )

            plt.xticks(xs, stage_labels, rotation=0)
            plt.ylabel(ylabel, fontsize=cfg.get("axes_label_fontsize", 11))
            plt.title(f"{title_prefix} - {element}", fontsize=cfg.get("title_fontsize", 13))
            grid_cfg = cfg.get("grid", {"axis": "y", "linestyle": "--", "linewidth": 0.5})
            plt.grid(True, **grid_cfg)
            plt.ylim(bottom=min(0, min(steps) - 0.2))

            out_path = target_dir / f"{element}_oer_potential.png"
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"Saved OER potential plot: {out_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except IndexError:
        print("Error: The file does not have enough columns. Please check the column indices.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    DATA_FILE = r"c:\Users\yingkaiwu\Desktop\single-atom\volcano\pbe-d3.dat"
    plot_oer_potential(DATA_FILE)
