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
    df = pd.read_csv(
        file_path,
        sep=r"\s+|,",
        header=0,
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
    Plot OER free-energy steps from HOO-only cumulative adsorption energies.
    ΔG4 is computed to close the 4e- cycle: ΔG4 = 4.43 - delta_E3_HOO.
    """
    data_path = Path(file_path)
    data_name = data_path.stem  # folder/name prefix without suffix
    data_title = data_path.name
    try:
        df = load_with_dynamic_columns(file_path)

        # Filter by HOO-HO energy between 2.0 and 4.0 to stay consistent with other plots.
        filt_col = "delta_E_HOO-HO(eV)"
        if filt_col not in df.columns:
            print(f"{file_path}: required filter column not found: {filt_col}.")
            return
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

        dg_cols = ["delta_E1(eV)", "delta_E2(eV)", "delta_E3_HOO(eV)"]
        missing_dg_cols = [col for col in dg_cols if col not in df.columns]
        if missing_dg_cols:
            print(f"{file_path}: required HOO energy columns not found: {missing_dg_cols}.")
            return
        df[dg_cols] = df[dg_cols].apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=dg_cols, inplace=True)
        if df.empty:
            print(f"{file_path}: no valid HOO energies after numeric cleanup.")
            return

        cfg = CONFIG.get("potential", {})
        stage_labels = cfg.get("stage_labels", ["*+2H2O", "OH*", "O*", "OOH*", "O2"])
        ylabel = cfg.get("ylabel", "ΔG (eV)")
        title_prefix = cfg.get("title_prefix", "OER Potential")
        text_fs = cfg.get("text_fontsize", 10)
        line_color = cfg.get("line_color", "tab:blue")
        line_width = cfg.get("line_width", 2.0)
        tick_fs = cfg.get("tick_label_fontsize", 11)
        show_grid = cfg.get("show_grid", True)
        facecolor = cfg.get("facecolor", None)

        base_dir = Path(output_dir) if output_dir else Path(".")
        target_dir = base_dir / data_name
        os.makedirs(target_dir, exist_ok=True)

        for _, row in df.iterrows():
            element = str(row["element"]).split("_")[0]
            de1, de2, de3 = (float(row[c]) for c in dg_cols)
            dg1 = de1
            dg2 = de2 - de1
            dg3 = de3 - de2
            dg4 = 4.43 - de3
            steps = [0.0, de1, de2, de3, 4.43]
            deltas = [dg1, dg2, dg3, dg4]
            pds_idx = int(np.argmax(deltas))

            n_states = len(steps)
            x_centers = np.arange(n_states)
            x_left = x_centers - 0.5
            x_right = x_centers + 0.5

            plt.figure(figsize=cfg.get("figsize", (9, 6)))
            ax = plt.gca()
            if facecolor is not None:
                ax.set_facecolor(facecolor)

            # horizontal plateaus centered on ticks
            for i in range(n_states):
                plt.plot(
                    [x_left[i], x_right[i]],
                    [steps[i], steps[i]],
                    color=line_color,
                    linewidth=line_width,
                )
                if i < n_states - 1:
                    y0, y1 = steps[i], steps[i + 1]
                    # vertical connector between plateaus
                    vcolor = cfg.get("pds_color", "red") if i == pds_idx else line_color
                    plt.plot(
                        [x_right[i], x_right[i]],
                        [y0, y1],
                        color=vcolor,
                        linewidth=line_width,
                    )
                    dy = deltas[i]
                    label_lines = [rf"$\Delta G_{{{i+1}}}$", f"{dy:.2f}"]
                    if i == pds_idx:
                        label_lines.append("PDS")
                    plt.text(
                        x_right[i] + 0.05,
                        (y0 + y1) / 2,
                        "\n".join(label_lines),
                        fontsize=text_fs,
                        color=cfg.get("pds_color", "red") if i == pds_idx else "black",
                        va="center",
                    )

            tick_positions = x_centers
            # Trim/extend labels to match tick count
            if len(stage_labels) < len(tick_positions):
                labels = stage_labels + [""] * (len(tick_positions) - len(stage_labels))
            else:
                labels = stage_labels[: len(tick_positions)]
            plt.xticks(tick_positions, labels, rotation=0, fontsize=tick_fs)
            ax.tick_params(axis="y", labelsize=tick_fs)
            plt.xlim(-0.5, n_states - 0.5)
            plt.ylabel(ylabel, fontsize=cfg.get("axes_label_fontsize", 11))
            plt.title(f"{title_prefix} ({data_title}) - {element}", fontsize=cfg.get("title_fontsize", 13))
            grid_cfg = cfg.get("grid", {"axis": "y", "linestyle": "--", "linewidth": 0.5})
            if show_grid:
                plt.grid(True, **grid_cfg)
            plt.ylim(bottom=min(0, min(steps) - 0.2))

            out_path = target_dir / f"{element}_oer_potential.png"
            plt.tight_layout()
            dpi = cfg.get("dpi", None)
            plt.savefig(out_path, dpi=dpi)
            plt.close()
            print(f"Saved OER potential plot: {out_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    DATA_FILE = r"c:\Users\yingkaiwu\Desktop\single-atom\volcano\pbe-d3.dat"
    plot_oer_potential(DATA_FILE)
