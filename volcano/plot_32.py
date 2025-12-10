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

def load_with_dynamic_columns(file_path):
    # Build column names to capture extra fields beyond the header.
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header_fields = lines[0].split()
    max_len = max(len(line.split()) for line in lines)
    column_names = header_fields + [f"extra_{i}" for i in range(len(header_fields), max_len)]

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=0,
        names=column_names,
        engine='python',
        skiprows=[1],
    )
    return df


def plot_three_lines(file_path, output_dir=None):
    """
    Reads data, filters, and plots scatter + two fitted lines (with R^2) on one chart.
    - x axis: column 7 (1-based) -> index 6
    - y axes: columns 8, 9 (1-based) -> indices 7, 8
    - Lines: fit y8 vs x and y9 vs x; report R^2 for each.
    """
    df = load_with_dynamic_columns(file_path)

    # Filter by 13th column (index 12) between 2.0 and 4.0
    filter_col_idx = 12
    if filter_col_idx >= len(df.columns):
        print(f"Not enough columns in {file_path} for filtering. Found {len(df.columns)} columns.")
        return
    filter_col = df.columns[filter_col_idx]
    df[filter_col] = pd.to_numeric(df[filter_col], errors='coerce')
    df = df[(df[filter_col] >= 2.0) & (df[filter_col] <= 4.0)].copy()
    if df.empty:
        print("No data remained after filtering; nothing to plot.")
        return

    # Keep only allowed elements; prefer specific variants when provided
    allowed_labels = CONFIG.get("allowed_elements", DEFAULT_ALLOWED_LABELS)
    allowed_bases = {label.split("_")[0] for label in allowed_labels}
    preferred_variant = {label.split("_")[0]: label for label in allowed_labels if "_" in label}

    df["__base"] = df["element"].astype(str).str.split("_").str[0]
    df = df[df["__base"].isin(allowed_bases)]
    if df.empty:
        print("No allowed elements remained after base-name filtering; nothing to plot.")
        return

    x_col_idx, y1_idx, y2_idx = 6, 7, 8
    if max(x_col_idx, y1_idx, y2_idx) >= len(df.columns):
        print(f"Not enough columns in {file_path}. Found {len(df.columns)} columns.")
        return

    x_col = df.columns[x_col_idx]
    y1_col = df.columns[y1_idx]
    y2_col = df.columns[y2_idx]

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y1_col] = pd.to_numeric(df[y1_col], errors='coerce')
    df[y2_col] = pd.to_numeric(df[y2_col], errors='coerce')

    df.dropna(subset=[x_col, y1_col, y2_col], inplace=True)
    if df.empty:
        print("No valid data to plot after cleaning.")
        return

    # Deduplicate by base element, preferring the specified variant when provided
    deduped_rows = []
    for base in df["__base"].unique():
        subset = df[df["__base"] == base]
        preferred = preferred_variant.get(base)
        if preferred and preferred in subset["element"].values:
            chosen = subset[subset["element"] == preferred].iloc[0]
        else:
            chosen = subset.iloc[0]
        deduped_rows.append(chosen)
    df = pd.DataFrame(deduped_rows)
    df["element"] = df["element"].astype(str).str.split("_").str[0]
    df.drop(columns="__base", errors="ignore", inplace=True)

    x = df[x_col]
    y1 = df[y1_col]
    y2 = df[y2_col]

    # Fit lines and compute R^2
    def fit_line_and_r2(x_vals, y_vals):
        m, b = np.polyfit(x_vals, y_vals, 1)
        y_pred = m * x_vals + b
        ss_res = np.sum((y_vals - y_pred) ** 2)
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
        return m, b, r2

    m1, b1, r2_1 = fit_line_and_r2(x, y1)
    m2, b2, r2_2 = fit_line_and_r2(x, y2)

    x_min, x_max = x.min(), x.max()
    x_range = np.linspace(x_min, x_max, 100)

    cfg = CONFIG.get("plot32", {})
    colors = cfg.get("colors", {"y1": "tab:orange", "y2": "tab:purple"})
    markers = cfg.get("markers", {"y1": "o", "y2": "^"})

    plt.figure(figsize=cfg.get("figsize", (10, 7)))
    scatter_alpha = cfg.get("scatter_alpha", 0.7)
    scatter_size = cfg.get("scatter_size", 30)
    plt.scatter(x, y1, alpha=scatter_alpha, s=scatter_size, marker=markers.get("y1", "o"), color=colors["y1"], label=f"{y1_col} data")
    plt.scatter(x, y2, alpha=scatter_alpha, s=scatter_size, marker=markers.get("y2", "^"), color=colors["y2"], label=f"{y2_col} data")

    # Labels: follow volcano rules (base element name, optional offsets), plotted separately for the two series
    grouped = df.assign(_base=df["element"].astype(str).str.split("_").str[0])
    label_offsets = cfg.get("label_offsets", {})
    annot_fs = cfg.get("annotation_fontsize", 8)

    y1_labels = grouped.groupby("_base")[[x_col, y1_col]].mean().reset_index()
    for _, row in y1_labels.iterrows():
        label = row["_base"]
        x_val = row[x_col]
        y_val = row[y1_col]
        dx, dy = label_offsets.get(label, (0, 0))
        plt.annotate(
            label,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=annot_fs,
            color=colors.get("y1"),
        )

    y2_labels = grouped.groupby("_base")[[x_col, y2_col]].mean().reset_index()
    for _, row in y2_labels.iterrows():
        label = row["_base"]
        x_val = row[x_col]
        y_val = row[y2_col]
        dx, dy = label_offsets.get(label, (0, 0))
        plt.annotate(
            label,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=annot_fs,
            color=colors.get("y2"),
        )

    line_style = cfg.get("line_style", "--")
    line_width = cfg.get("line_width", 1.3)
    plt.plot(x_range, m1 * x_range + b1, color=colors["y1"], linestyle=line_style, linewidth=line_width,
             label=f"{y1_col} fit: y={m1:.3f}x+{b1:.3f}, R^2={r2_1:.3f}")
    plt.plot(x_range, m2 * x_range + b2, color=colors["y2"], linestyle=line_style, linewidth=line_width,
             label=f"{y2_col} fit: y={m2:.3f}x+{b2:.3f}, R^2={r2_2:.3f}")

    label_fs = cfg.get("axes_label_fontsize", None)
    title_fs = cfg.get("title_fontsize", None)
    legend_fs = cfg.get("legend_fontsize", None)
    plt.xlabel(f"Column 7: {x_col} (eV)", fontsize=label_fs)
    plt.ylabel(cfg.get("ylabel", "Energy (eV)"), fontsize=label_fs)
    plt.title(cfg.get("title", "Scatter with Two Fitted Lines"), fontsize=title_fs)
    grid_cfg = cfg.get("grid", {"linestyle": "--", "linewidth": 0.5, "which": "both"})
    plt.grid(True, **grid_cfg)
    if legend_fs:
        leg = plt.legend()
        if leg:
            for text in leg.get_texts():
                text.set_fontsize(legend_fs)
    else:
        plt.legend()
    plt.legend()

    target_dir = Path(output_dir) if output_dir else Path('.')
    os.makedirs(target_dir, exist_ok=True)
    out_path = target_dir / "plot_32.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    DATA_FILE = r'c:\Users\yingkaiwu\Desktop\single-atom\volcano\pbe-d3-spin.dat'
    plot_three_lines(DATA_FILE)
