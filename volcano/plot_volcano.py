import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_config import CONFIG


def filter_and_plot_data(file_path, output_dir=None):
    """
    Reads data, filters, deduplicates elements, and plots a volcano chart.

    - Filters rows where column 13 (index 12) is between 2.0 and 4.0.
    - Plots column 19 vs. column 22 (1-based indices) if available.
    - Trend lines: fit left/right separately, find their true intersection, and draw each segment split at that point.
    """
    try:
        # --- Load with dynamic column names to capture extra fields ---
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
            engine='python'
        )

        # --- Filtering ---
        filter_column_index = 12
        filter_column_name = df.columns[filter_column_index]
        df[filter_column_name] = pd.to_numeric(df[filter_column_name], errors='coerce')

        filtered_df = df[
            (df[filter_column_name] >= 2.0) & (df[filter_column_name] <= 4.0)
        ].copy()

        print(f"Successfully loaded data from {file_path}.")
        print(f"Original rows: {len(df)}, Rows after filtering: {len(filtered_df)}")

        if filtered_df.empty:
            col_stats = pd.to_numeric(df[filter_column_name], errors='coerce')
            print(f"No data remains after filtering. Column '{filter_column_name}' range: "
                  f"min={col_stats.min()}, max={col_stats.max()}")
            return

        # --- Deduplicate elements (prefer plain names; Sn->Sn_d, Ru->Ru_pv) ---
        def dedupe_by_element(df_in):
            if df_in.empty:
                return df_in
            preferred_special = {'Sn': 'Sn_d', 'Ru': 'Ru_pv'}
            deduped_rows = []
            base_names = df_in['element'].apply(lambda x: str(x).split('_')[0]).unique()
            for base in base_names:
                subset = df_in[df_in['element'].str.startswith(base)]
                preferred_name = preferred_special.get(base, base)
                if preferred_name in subset['element'].values:
                    chosen = subset[subset['element'] == preferred_name].iloc[0]
                else:
                    chosen = subset.iloc[0]
                deduped_rows.append(chosen)
            return pd.DataFrame(deduped_rows)

        filtered_df = dedupe_by_element(filtered_df)

        # --- Plot columns ---
        x_col_index = 16
        y_col_index = 20
        if x_col_index >= len(filtered_df.columns) or y_col_index >= len(filtered_df.columns):
            print(f"Error: Requested plot columns (19, 22) exceed available columns ({len(filtered_df.columns)}).")
            print("Available columns (0-based):")
            for i, col in enumerate(filtered_df.columns):
                print(f"  {i}: {col}")
            return

        x_col_name = filtered_df.columns[x_col_index]
        y_col_name = filtered_df.columns[y_col_index]

        filtered_df[x_col_name] = pd.to_numeric(filtered_df[x_col_name], errors='coerce')
        filtered_df[y_col_name] = pd.to_numeric(filtered_df[y_col_name], errors='coerce')
        filtered_df.dropna(subset=[x_col_name, y_col_name], inplace=True)

        if filtered_df.empty:
            print("No valid data for plotting after cleaning non-numeric values.")
            return

        print(f"Plotting '{y_col_name}' vs. '{x_col_name}'.")

        # --- Choose initial split seed (Pt x if present, else 2.0) ---
        base_elements = filtered_df['element'].astype(str).str.split('_').str[0]
        pt_mask = base_elements == 'Pt'
        if pt_mask.any():
            split_seed = float(filtered_df.loc[pt_mask, x_col_name].iloc[0])
            print(f"Using Pt x-position as seed: split_seed = {split_seed}")
        else:
            split_seed = 2.0
            print("Pt not found after filtering; using split_seed = 2.0")

        left_df = filtered_df[filtered_df[x_col_name] < split_seed]
        right_df = filtered_df[filtered_df[x_col_name] >= split_seed]

        # --- Fit lines and find true intersection ---
        def fit_line(subset):
            if len(subset) < 2:
                return None
            x_vals = subset[x_col_name]
            y_vals = subset[y_col_name]
            m, b = np.polyfit(x_vals, y_vals, 1)
            return m, b

        left_fit = fit_line(left_df)
        right_fit = fit_line(right_df)

        x_axis_min, x_axis_max = cfg.get("x_axis_limits", (0.0, 3.0))

        cfg = CONFIG.get("volcano", {})
        plt.figure(figsize=cfg.get("figsize", (10, 7)))
        plt.scatter(
            filtered_df[x_col_name],
            filtered_df[y_col_name],
            alpha=cfg.get("scatter_alpha", 0.7),
            color=cfg.get("scatter_color", "tab:blue"),
            marker=cfg.get("scatter_marker", "o"),
            s=cfg.get("scatter_size", 30),
        )

        if left_fit and right_fit and not left_df.empty and not right_df.empty:
            m_l, b_l = left_fit
            m_r, b_r = right_fit
            if m_l != m_r:
                x_int = (b_r - b_l) / (m_l - m_r)
                x_int = max(x_axis_min, min(x_axis_max, x_int))
                y_int = m_l * x_int + b_l

                samples = cfg.get("trend_samples", 50)
                x_left_range = np.linspace(x_axis_min, x_int, samples)
                x_right_range = np.linspace(x_int, x_axis_max, samples)

                line_style = cfg.get("trend_line_style", "--")
                line_width = cfg.get("trend_line_width", 1.3)
                plt.plot(
                    x_left_range,
                    m_l * x_left_range + b_l,
                    color=cfg.get("trend_left_color", "orange"),
                    linestyle=line_style,
                    linewidth=line_width,
                    label=f'{x_col_name} < {x_int:.2f} trend',
                )
                plt.plot(
                    x_right_range,
                    m_r * x_right_range + b_r,
                    color=cfg.get("trend_right_color", "purple"),
                    linestyle=line_style,
                    linewidth=line_width,
                    label=f'{x_col_name} >= {x_int:.2f} trend',
                )
                plt.scatter(
                    [x_int],
                    [y_int],
                    color=cfg.get("split_line_color", "gray"),
                    s=cfg.get("split_marker_size", 12),
                    zorder=5,
                )
                plt.axvline(
                    x_int,
                    color=cfg.get("split_line_color", "gray"),
                    linestyle=cfg.get("split_line_style", ":"),
                    linewidth=cfg.get("split_line_width", 1),
                )

        # --- Labels with slight manual offsets for crowded ones ---
        grouped = filtered_df.assign(_base=filtered_df['element'].astype(str).str.split('_').str[0])
        label_positions = grouped.groupby('_base')[[x_col_name, y_col_name]].mean().reset_index()

        label_offsets = cfg.get("label_offsets", {})
        annot_fs = cfg.get("annotation_fontsize", 8)
        for _, row in label_positions.iterrows():
            label = row['_base']
            x_val, y_val = row[x_col_name], row[y_col_name]
            dx, dy = label_offsets.get(label, (0, 0))
            plt.annotate(
                label,
                (x_val, y_val),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=annot_fs
            )

        label_fs = cfg.get("axes_label_fontsize", None)
        title_fs = cfg.get("title_fontsize", None)
        legend_fs = cfg.get("legend_fontsize", None)
        plt.xlabel(f"{cfg.get('xlabel_prefix', 'Descriptor')}: {x_col_name} (eV)", fontsize=label_fs)
        plt.ylabel(cfg.get("ylabel", f"Activity: {y_col_name} (eV)"), fontsize=label_fs)
        plt.title(cfg.get("title", "Volcano Plot"), fontsize=title_fs)
        grid_cfg = cfg.get("grid", {"linestyle": "--", "linewidth": 0.5, "which": "both"})
        plt.grid(True, **grid_cfg)
        if "xlim" in cfg:
            plt.xlim(*cfg["xlim"])
        if "ylim" in cfg:
            plt.ylim(*cfg["ylim"])
        if legend_fs:
            leg = plt.legend()
            if leg:
                for text in leg.get_texts():
                    text.set_fontsize(legend_fs)

        target_dir = Path(output_dir) if output_dir else Path(".")
        os.makedirs(target_dir, exist_ok=True)
        output_filename = target_dir / 'volcano_plot.png'
        plt.savefig(output_filename)

        print(f"\nPlot successfully generated and saved as '{output_filename}'.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except IndexError:
        print("Error: The file does not have enough columns. Please check the column indices.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-d3.dat'
    filter_and_plot_data(DATA_FILE)
