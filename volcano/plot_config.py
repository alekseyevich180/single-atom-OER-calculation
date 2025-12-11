CONFIG = {
    # Elements allowed for plotting; labels use the base name (suffixes are for selection preference only).
    "allowed_elements": [
        "Ag", "Au", "Bi", "Cd", "Co",
        "Cr_pv", "Cu", "Fe", "Ga", "Hg", "In_d", "Ir",
        "Mn_pv", "Mo_sv", "Ni", "Pb", "Pd", "Pt", "Rh",
        "Ru_pv", "Sb", "Sn_d", "Sr_sv", "Zn",
    ],
    "volcano": {
        "figsize": (6.1, 6),
        "dpi": 600,
        "scatter_alpha": 0.7,
        "scatter_color": "tab:blue",  # fallback; individual colors below override when set
        "scatter_left_color": "#5a7fbf",
        "scatter_right_color": "#9f66a8",
        "scatter_marker": "o",
        "scatter_size": 30,
        "G0_base": 4.43,
        "potential_shift": 1.11,        # potential = max(ΔG1-4) - shift
        "descriptor_column": "dG2",     # x-axis; derived from columns 7,8,9
        "activity_column": "potential_neg", # y-axis; negative potential for volcano shape
        "split_seed_element": "Pd",     # use this element's x as initial split if present
        "split_seed_default": 2.0,      # fallback split seed if element not found
        "xlabel_override": r"$\Delta E_{\mathrm{O*}}$ - $\Delta E_{\mathrm{HO*}}$ (eV)",        # optional custom x-axis label
        "trend_line_style": "--",
        "trend_line_width": 1.3,
        "trend_left_color": "#5a7fbf",
        "trend_right_color": "#9f66a8",
        "split_line_color": "gray",
        "split_line_style": ":",
        "split_line_width": 1,
        "split_marker_size": 12,
        "trend_samples": 50,
        "x_axis_limits": (0.0, 3.0),
        "grid": {"linestyle": "--", "linewidth": 0.5, "which": "both"},
        "xlim": (0, 3),
        "ylim": (-2.5, -0.5),
        "axes_label_fontsize": 15,
        "title_fontsize": 13,
        "legend_fontsize": 10,
        "annotation_fontsize": 10,
        "label_offsets": {
            "Ir": (10, 6),
            "Cr": (-10, 6),
            "Mn": (10, -6),
            "Pb": (-10, -6),
        },
        "title": "Volcano Plot",
        "xlabel_prefix": "Descriptor",
        "ylabel": r"- $\eta_{\mathrm{OER}}$ (V)",
    },
    "plot32": {
        "figsize": (6, 6),
        "dpi": 600,
        "scatter_alpha": 0.7,
        "x_col_index": 6,  # configurable x-axis column (0-based, default column 7)
        "colors": {
            "y1": "#5ec6ce",
            "y2": "#d9675c",
        },
        "markers": {
            "y1": "o",
            "y2": "^",
        },
        "scatter_size": 30,
        "line_style": "--",
        "line_width": 1.3,
        "grid": {"linestyle": "--", "linewidth": 0.5, "which": "both"},
        "title": r"The relation between $\Delta E_{\mathrm{O*}}$ - $\Delta E_{\mathrm{HO*}}$ and $\Delta E_{\mathrm{HOO*}}$ - $\Delta E_{\mathrm{HO*}}$",
        "ylabel": r"$\Delta E_{\mathrm{ads}}$ (eV)",
        "xlabel_override": r"$\Delta E_{\mathrm{HO*}}$ (eV)",  # optional custom x-axis label
        "axes_label_fontsize": 15,
        "title_fontsize": 13,
        "legend_fontsize": 10,
        "annotation_fontsize": 12,
        # Optional custom legend text; fit labels support .format(y, m, b, r2)
        "legend_labels": {
            "y1_data": "ΔEO - ΔEHO (eV) data",
            "y2_data": "ΔEHOO - ΔEHO (eV) data",
            "y1_fit": "y={m:.3f}x+{b:.3f}, R²={r2:.3f}",
            "y2_fit": "y={m:.3f}x+{b:.3f}, R²={r2:.3f}",
        },
        "label_offsets": {
            "Ir": (10, 6),
            "Cr": (-10, 6),
            "Mn": (10, -6),
            "Pb": (-10, -6),
        },
    },
    "potential": {
        "figsize": (3, 2),
        "dpi": 600,
        "line_color": "tab:blue",
        "line_width": 2.5,
        "arrow_color": "black",
        "arrow_width": 1.0,
        "arrow_head_width": 6,
        "arrow_head_length": 6,
        "pds_color": "red",
        "stage_labels": ["*+H$_2$O", "OH*", "O*", "OOH*", "O$_2$"],
        "ylabel": r"$\Delta E$ (eV)",
        "title_prefix": "OER Potential",
        "axes_label_fontsize": 8,
        "title_fontsize": 8,
        "text_fontsize": 6,
        "tick_label_fontsize": 7,
        "grid": {"axis": "y", "linestyle": "--", "linewidth": 0.5},
        "show_grid": False,
        "facecolor": "white",
    },
}
