# Combined trend plotting for all elements in datafit_test

import importlib.util
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

# Fixed palette for multi-element plots (6 original + 3 recommended)
PALETTE = [
    "#4c6f8c",
    "#e68a2e",
    "#d9675c",
    "#9f66a8",
    "#1b9e77",
    "#75c267",
    "#5ec6ce",
    "#5a7fbf",
    "#6a64ab",
]


def remove_outliers(X, y, cfg, method="zscore"):
    """Remove outliers using z-score, IQR, or MAD."""
    if method == "zscore":
        y_zscores = zscore(y)
        mask = np.abs(y_zscores) < cfg.Z_THRESHOLD
    elif method == "iqr":
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower = q1 - cfg.IQR_MULTIPLIER * iqr
        upper = q3 + cfg.IQR_MULTIPLIER * iqr
        mask = (y >= lower) & (y <= upper)
    elif method == "mad":
        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))
        if mad_y == 0:
            mask = np.ones_like(y, dtype=bool)
        else:
            mask = np.abs(y - median_y) / (1.4826 * mad_y) < cfg.MAD_THRESHOLD
    else:
        raise ValueError("Invalid method. Choose 'zscore', 'iqr', or 'mad'.")
    return X[mask], y[mask]


def get_trend_line_for_element(element_name):
    """Load one element config and return predicted trend (X_pred, y_pred)."""
    print(f"--- Processing element: {element_name} ---")

    config_path = os.path.join(element_name, "gpr_config.py")
    if not os.path.exists(config_path):
        print(f"Error: missing config {config_path}. Skipping.")
        return None, None

    spec = importlib.util.spec_from_file_location(f"gpr_config_{element_name}", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    data_path = os.path.join(element_name, cfg.FILE_NAME)
    try:
        data = np.loadtxt(data_path, delimiter=",", skiprows=cfg.SKIP_ROWS, usecols=cfg.USE_COLS)
    except FileNotFoundError:
        print(f"Error: missing data file {data_path}. Skipping.")
        return None, None

    X = data[:, cfg.FEATURE_COL].reshape(-1, 1)
    y = data[:, cfg.TARGET_COL] * cfg.TARGET_SIGN
    print(f"Data loaded. Points: {len(y)}")

    if getattr(cfg, "ANGLE_FILTER_ENABLED", False):
        angle_mask = ((X >= cfg.ANGLE_MIN) & (X <= cfg.ANGLE_MAX)).flatten()
        X = X[angle_mask].reshape(-1, 1)
        y = y[angle_mask]
        print(f"Angle-filtered points: {len(y)}")

    X_filtered, y_filtered = remove_outliers(X, y, cfg, method="zscore")

    if getattr(cfg, "BINNING_ENABLED", False):
        X_original = X_filtered.flatten()
        y_original = y_filtered.ravel()
        start_point, end_point = X_original.min(), X_original.max()
        X_smoothed, y_smoothed = [], []
        current_center = start_point + cfg.WINDOW_WIDTH / 2.0
        while current_center <= end_point + cfg.WINDOW_WIDTH / 2.0 + 1e-6:
            lower_bound = current_center - cfg.WINDOW_WIDTH / 2.0
            upper_bound = current_center + cfg.WINDOW_WIDTH / 2.0
            y_in_window = y_original[(X_original >= lower_bound) & (X_original < upper_bound)]
            if len(y_in_window) > 0:
                X_smoothed.append(current_center)
                y_smoothed.append(np.mean(y_in_window))
            current_center += cfg.STEP_SIZE
        X_filtered = np.array(X_smoothed).reshape(-1, 1)
        y_filtered = np.array(y_smoothed).ravel()

    scaler_X = StandardScaler().fit(X_filtered)
    scaler_y = StandardScaler().fit(y_filtered.reshape(-1, 1))
    X_scaled = scaler_X.transform(X_filtered)
    y_scaled = scaler_y.transform(y_filtered.reshape(-1, 1)).ravel()

    from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

    if hasattr(cfg, "KERNELS") and "Matern_WK" in cfg.KERNELS:
        kernel_to_use = cfg.KERNELS["Matern_WK"]
        print(f"Using configured kernel 'Matern_WK' for {element_name}")
    else:
        print(f"Warning: kernel 'Matern_WK' not found for {element_name}, using default RBF.")
        kernel_to_use = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1)

    gpr = GaussianProcessRegressor(
        kernel=kernel_to_use,
        n_restarts_optimizer=cfg.FINAL_N_RESTARTS,
        alpha=cfg.FINAL_ALPHA,
        normalize_y=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X_scaled, y_scaled)

    X_pred_original = np.linspace(cfg.PRED_ANGLE_MIN, cfg.PRED_ANGLE_MAX, cfg.PRED_POINTS).reshape(-1, 1)
    X_pred_scaled = scaler_X.transform(X_pred_original)

    y_pred_scaled, _ = gpr.predict(X_pred_scaled, return_std=True)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    print(f"--- Finished: {element_name} ---\n")
    return X_pred_original, y_pred_original


def main():
    """Discover elements automatically and plot combined trend lines."""
    current_dir = "."
    try:
        all_dirs = [
            d
            for d in os.listdir(current_dir)
            if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith("__") and "." not in d
        ]
        elements = sorted([d for d in all_dirs if os.path.exists(os.path.join(d, "gpr_config.py"))])
    except Exception as e:
        print(f"Error discovering element folders: {e}")
        elements = []

    if not elements:
        print("Error: no valid element folders containing gpr_config.py were found.")
        return

    print(f"Found {len(elements)} elements: {', '.join(elements)}")

    colors = PALETTE

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(12, 9))
    fig_lines, ax_lines = plt.subplots(figsize=(12, 9))

    for i, element in enumerate(elements):
        X_pred, y_pred = get_trend_line_for_element(element)

        if X_pred is not None and y_pred is not None:
            color = colors[i % len(colors)]
            label = f"GPR Trend - {element.capitalize()}"
            ax.plot(
                X_pred.ravel(),
                y_pred,
                color=color,
                linewidth=2.2,
                label=label,
            )

            # secondary figure: trend lines + prediction points only
            ax_lines.plot(
                X_pred.ravel(),
                y_pred,
                color=color,
                linewidth=2.2,
                label=label,
            )
            ax_lines.scatter(
                X_pred.ravel(),
                y_pred,
                s=24,  # show prediction samples clearly
                facecolors="none",
                edgecolors=color,
                linewidths=0.8,
                alpha=0.85,
                marker="o",
            )

    ax.set_title("Angle - ICOHP Trend Lines for Rutile Type Metal Elements", fontsize=18, fontweight="bold")
    ax.set_xlabel("O-M-O Angle (deg)", fontsize=15)
    ax.set_ylabel("-ICOHP (eV)", fontsize=15)
    ax.set_xlim(130, 180.6)
    ax.set_ylim(0.4, 1.9)
    ax.legend(fontsize=12, loc="best", frameon=True, shadow=True)
    ax.tick_params(axis="both", which="major", labelsize=13)
    fig.tight_layout(pad=1.5)

    save_path = "Combined_GPR_Trends_Optimized.png"
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"\nCombined trend plot saved to {save_path}")

    ax_lines.set_title("Angle - ICOHP Trend Lines (Predictions Only)", fontsize=18, fontweight="bold")
    ax_lines.set_xlabel("O-M-O Angle (deg)", fontsize=15)
    ax_lines.set_ylabel("-ICOHP (eV)", fontsize=15)
    ax_lines.set_xlim(130, 180.6)
    ax_lines.set_ylim(0.4, 1.9)
    ax_lines.legend(fontsize=12, loc="best", frameon=True, shadow=True)
    ax_lines.tick_params(axis="both", which="major", labelsize=15)
    fig_lines.tight_layout(pad=1.5)
    save_path_lines = "Combined_GPR_Trends_LinesOnly.png"
    fig_lines.savefig(save_path_lines, dpi=600, bbox_inches="tight")
    print(f"Lines-only trend plot saved to {save_path_lines}")

    # no plt.show(): save only, no window


if __name__ == "__main__":
    main()
