import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_config import CONFIG


def load_with_dynamic_columns(file_path):
    # Build column names to capture extra fields beyond the header.
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header_fields = lines[0].split()
    max_len = max(len(line.split()) for line in lines)
    column_names = header_fields + [f'extra_{i}' for i in range(len(header_fields), max_len)]

    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=0,
        names=column_names,
        engine='python',
        skiprows=[1],
    )
    return df


def plot_three_lines(file_path, output_dir=None):
    """
    Reads data, filters, and plots scatter + two fitted lines (with R²) on one chart.
    - x axis: column 7 (1-based) -> index 6
    - y axes: columns 8, 9 (1-based) -> indices 7, 8
    - Lines: fit y8 vs x and y9 vs x; report R² for each.
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

    x = df[x_col]
    y1 = df[y1_col]
    y2 = df[y2_col]

    # Fit lines and compute R²
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
    plt.scatter(x, y1, alpha=scatter_alpha, s=scatter_size, marker=markers.get("y1", "o"), color=colors["y1"], label=f'{y1_col} data')
    plt.scatter(x, y2, alpha=scatter_alpha, s=scatter_size, marker=markers.get("y2", "^"), color=colors["y2"], label=f'{y2_col} data')

    line_style = cfg.get("line_style", "--")
    line_width = cfg.get("line_width", 1.3)
    plt.plot(x_range, m1 * x_range + b1, color=colors["y1"], linestyle=line_style, linewidth=line_width,
             label=f'{y1_col} fit: y={m1:.3f}x+{b1:.3f}, R²={r2_1:.3f}')
    plt.plot(x_range, m2 * x_range + b2, color=colors["y2"], linestyle=line_style, linewidth=line_width,
             label=f'{y2_col} fit: y={m2:.3f}x+{b2:.3f}, R²={r2_2:.3f}')

    label_fs = cfg.get("axes_label_fontsize", None)
    title_fs = cfg.get("title_fontsize", None)
    legend_fs = cfg.get("legend_fontsize", None)
    plt.xlabel(f'Column 7: {x_col} (eV)', fontsize=label_fs)
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

    target_dir = Path(output_dir) if output_dir else Path(".")
    os.makedirs(target_dir, exist_ok=True)
    out_path = target_dir / "plot_32.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-d3.dat'
    plot_three_lines(DATA_FILE)
