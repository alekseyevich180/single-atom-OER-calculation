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

        # --- Keep only allowed elements and deduplicate by base (use preferred variant when specified) ---
        allowed_labels = CONFIG.get("allowed_elements", DEFAULT_ALLOWED_LABELS)
        allowed_bases = {label.split("_")[0] for label in allowed_labels}
        preferred_variant = {label.split("_")[0]: label for label in allowed_labels if "_" in label}

        def keep_allowed_and_dedupe(df_in):
            if df_in.empty:
                return df_in
            df_in = df_in.copy()
            df_in["__base"] = df_in["element"].astype(str).str.split("_").str[0]
            df_in = df_in[df_in["__base"].isin(allowed_bases)]
            if df_in.empty:
                return df_in
            deduped_rows = []
            for base in df_in["__base"].unique():
                subset = df_in[df_in["__base"] == base]
                preferred = preferred_variant.get(base)
                if preferred and preferred in subset["element"].values:
                    chosen = subset[subset["element"] == preferred].iloc[0]
                else:
                    chosen = subset.iloc[0]
                deduped_rows.append(chosen)
            deduped = pd.DataFrame(deduped_rows).drop(columns="__base")
            deduped["element"] = deduped["element"].astype(str).str.split("_").str[0]
            return deduped

        filtered_df = keep_allowed_and_dedupe(filtered_df)
        if filtered_df.empty:
            print("No allowed elements remain after filtering; nothing to plot.")
            return

        # --- Build ΔG1-4 from columns 7/8/9 (1-based); no cumulative energies ---
        energy_indices = [6, 7, 8]  # zero-based for columns 7,8,9
        if any(idx >= len(filtered_df.columns) for idx in energy_indices):
            print(f"Error: Need columns 7,8,9 (indices 6,7,8) to build ΔG1-4, but only {len(filtered_df.columns)} columns present.")
            return
        dg_cols = [filtered_df.columns[i] for i in energy_indices]
        filtered_df[dg_cols] = filtered_df[dg_cols].apply(pd.to_numeric, errors="coerce")
        filtered_df.dropna(subset=dg_cols, inplace=True)
        if filtered_df.empty:
            print("No valid rows after parsing columns 7,8,9.")
            return

        cfg = CONFIG.get("volcano", {})
        g0_base = cfg.get("G0_base", 4.43)
        potential_shift = cfg.get("potential_shift", 1.11)  # potential = max(ΔG) - shift

        filtered_df["dG1"] = filtered_df[dg_cols[0]]
        filtered_df["dG2"] = filtered_df[dg_cols[1]] - filtered_df[dg_cols[0]]
        filtered_df["dG3"] = filtered_df[dg_cols[2]] - filtered_df[dg_cols[1]]
        filtered_df["dG4"] = g0_base - filtered_df[dg_cols[2]]

        filtered_df["limiting_dG"] = filtered_df[["dG1", "dG2", "dG3", "dG4"]].max(axis=1)
        # Potential definition: max(ΔG1-4) - potential_shift
        filtered_df["potential"] = filtered_df["limiting_dG"] - potential_shift
        filtered_df["potential_neg"] = -filtered_df["potential"]  # invert for volcano plotting
        print(f"Computed potential as max(ΔG1-4) - {potential_shift} eV; using its negative for plotting.")

        descriptor_col = cfg.get("descriptor_column", "dG2")
        activity_col = cfg.get("activity_column", "potential_neg")
        missing = [col for col in (descriptor_col, activity_col) if col not in filtered_df.columns]
        if missing:
            print(f"Configured columns not found: {missing}. Available derived columns: dG1-4, limiting_dG, potential, potential_neg.")
            return

        x_col_name = descriptor_col
        y_col_name = activity_col
        filtered_df.dropna(subset=[x_col_name, y_col_name], inplace=True)

        if filtered_df.empty:
            print("No valid data for plotting after cleaning non-numeric values.")
            return

        print(f"Plotting '{y_col_name}' vs. '{x_col_name}'.")

        # --- Choose initial split seed (configurable element or fallback) ---
        split_element = cfg.get("split_seed_element", "Pd")
        split_default = cfg.get("split_seed_default", 2.0)
        base_elements = filtered_df['element'].astype(str).str.split('_').str[0]
        seed_mask = base_elements == split_element
        if seed_mask.any():
            split_seed = float(filtered_df.loc[seed_mask, x_col_name].iloc[0])
            print(f"Using {split_element} x-position as seed: split_seed = {split_seed}")
        else:
            split_seed = split_default
            print(f"{split_element} not found after filtering; using split_seed = {split_default}")

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
        plt.figure(figsize=cfg.get("figsize", (10, 7)))
        left_color = cfg.get("scatter_left_color") or cfg.get("trend_left_color", "tab:blue")
        right_color = cfg.get("scatter_right_color") or cfg.get("trend_right_color", "tab:purple")
        scatter_alpha = cfg.get("scatter_alpha", 0.7)
        scatter_marker = cfg.get("scatter_marker", "o")
        scatter_size = cfg.get("scatter_size", 30)
        if not left_df.empty:
            plt.scatter(
                left_df[x_col_name],
                left_df[y_col_name],
                alpha=scatter_alpha,
                color=left_color,
                marker=scatter_marker,
                s=scatter_size,
                label=cfg.get("left_label", f"{x_col_name} < split"),
            )
        if not right_df.empty:
            plt.scatter(
                right_df[x_col_name],
                right_df[y_col_name],
                alpha=scatter_alpha,
                color=right_color,
                marker=scatter_marker,
                s=scatter_size,
                label=cfg.get("right_label", f"{x_col_name} >= split"),
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
        x_label_text = cfg.get("xlabel_override") or f"{cfg.get('xlabel_prefix', 'Descriptor')}: {x_col_name} (eV)"
        plt.xlabel(x_label_text, fontsize=label_fs)
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
        dpi = cfg.get("dpi", None)
        plt.savefig(output_filename, dpi=dpi)
        plt.close()

        print(f"\nPlot successfully generated and saved as '{output_filename}'.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except IndexError:
        print("Error: The file does not have enough columns. Please check the column indices.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-spinoff.dat'
    filter_and_plot_data(DATA_FILE)
