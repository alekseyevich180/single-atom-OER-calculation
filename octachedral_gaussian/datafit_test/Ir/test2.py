"""
Cleaned-up GPR analysis script.
- Removes duplicated remove_outliers
- Adds small helper functions for clarity
- Keeps compatibility with gpr_config.py
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
import gpr_config as cfg

# ------------------------
# Outlier handling
# ------------------------
def remove_outliers(X, y, method="zscore"):
    if method == "zscore":
        y_zscores = zscore(y)
        mask = np.abs(y_zscores) < cfg.Z_THRESHOLD
    elif method == "iqr":
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - cfg.IQR_MULTIPLIER * iqr
        upper_bound = q3 + cfg.IQR_MULTIPLIER * iqr
        mask = (y >= lower_bound) & (y <= upper_bound)
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


def save_plot(filename):
    if not cfg.SAVE_PLOTS:
        return
    if not os.path.exists(cfg.SAVE_DIR):
        os.makedirs(cfg.SAVE_DIR)
    full_path = os.path.join(cfg.SAVE_DIR, f"{filename}.{cfg.FILE_FORMAT}")
    try:
        plt.savefig(full_path, dpi=cfg.DPI, bbox_inches="tight")
        print(f"Plot saved: {full_path}")
    except Exception as e:
        print(f"Saving plot {filename} failed: {e}")


# ------------------------
# Data loading and optional sliding-window smoothing
# ------------------------
def load_data():
    data = np.loadtxt(
        cfg.FILE_NAME,
        delimiter=",",
        skiprows=cfg.SKIP_ROWS,
        usecols=cfg.USE_COLS,
    )
    X = data[:, cfg.FEATURE_COL].reshape(-1, 1)
    y = data[:, cfg.TARGET_COL] * cfg.TARGET_SIGN
    return X, y


def apply_sliding_window(X, y):
    X_original = X.flatten()
    y_original = y.ravel()
    start_point = X_original.min()
    end_point = X_original.max()

    X_smoothed = []
    y_smoothed = []
    current_center = start_point + cfg.WINDOW_WIDTH / 2.0

    while current_center <= end_point + cfg.WINDOW_WIDTH / 2.0 + 1e-6:
        lower = current_center - cfg.WINDOW_WIDTH / 2.0
        upper = current_center + cfg.WINDOW_WIDTH / 2.0
        window_mask = (X_original >= lower) & (X_original < upper)
        y_in_window = y_original[window_mask]
        if len(y_in_window) > 0:
            X_smoothed.append(current_center)
            y_smoothed.append(np.mean(y_in_window))
        current_center += cfg.STEP_SIZE

    return np.array(X_smoothed).reshape(-1, 1), np.array(y_smoothed).ravel()


# ------------------------
# Evaluation utilities
# ------------------------
def evaluate_methods(X, y):
    results = {}
    for method in cfg.DETECTION_METHODS:
        X_f, y_f = remove_outliers(X, y, method=method)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_f)
        y_scaled = scaler_y.fit_transform(y_f.reshape(-1, 1)).ravel()

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
        )

        method_results = []
        for name, kernel in cfg.KERNELS.items():
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=cfg.INITIAL_N_RESTARTS,
                alpha=cfg.INITIAL_ALPHA,
                normalize_y=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                try:
                    gpr.fit(X_train, y_train)
                except ValueError as e:
                    print(f"Warning: {method}/{name} fitting failed: {e}")
                    method_results.append((name, np.inf, -np.inf))
                    continue

                try:
                    log_like = gpr.log_marginal_likelihood_value_
                except AttributeError:
                    log_like = gpr.log_marginal_likelihood(gpr.kernel_.theta)

                num_params = gpr.kernel_.theta.size
                aic = 2 * num_params - 2 * log_like
                scores = cross_val_score(gpr, X_scaled, y_scaled, cv=cfg.CV_FOLDS, scoring="r2")
                mean_r2 = scores.mean()
                method_results.append((name, aic, mean_r2))

        results[method] = {"method_results": method_results, "data_points": len(y_f)}
    return results


def pick_best(results):
    best_method = None
    best_kernel = None
    best_aic = np.inf
    best_r2 = -np.inf

    print("\n--- Outlier methods / kernels ---")
    for method, result in results.items():
        print(f"Method: {method} (kept {result['data_points']} points)")
        for kernel_name, aic_value, r2_value in result["method_results"]:
            print(f"  Kernel {kernel_name}, AIC: {aic_value:.3f}, Mean R2: {r2_value:.3f}")
            if r2_value > best_r2 or (np.isclose(r2_value, best_r2) and aic_value < best_aic):
                best_r2 = r2_value
                best_aic = aic_value
                best_method = method
                best_kernel = kernel_name

    print("----------------------------------------")
    print(f"Best: method={best_method}, kernel={best_kernel}, AIC={best_aic:.3f}, R2={best_r2:.3f}")
    return best_method, best_kernel


# ------------------------
# Training and plotting
# ------------------------
def train_final(X, y, method, kernel_name):
    X_f, y_f = remove_outliers(X, y, method=method)
    if cfg.BINNING_ENABLED:
        print(
            f"Sliding-window smoothing enabled. Window {cfg.WINDOW_WIDTH}, step {cfg.STEP_SIZE}."
        )
        X_f, y_f = apply_sliding_window(X_f, y_f)
        print(f"Smoothed points: {len(y_f)}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_f)
    y_scaled = scaler_y.fit_transform(y_f.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
    )

    gpr = GaussianProcessRegressor(
        kernel=cfg.KERNELS[kernel_name],
        n_restarts_optimizer=cfg.FINAL_N_RESTARTS,
        alpha=cfg.FINAL_ALPHA,
        normalize_y=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gpr.fit(X_train, y_train)

    train_r2 = r2_score(y_train, gpr.predict(X_train))
    test_r2 = r2_score(y_test, gpr.predict(X_test))
    print(f"\nFinal Train R2: {train_r2:.3f}, Test R2: {test_r2:.3f}")
    print("Optimized kernel:\n", gpr.kernel_)

    return gpr, scaler_X, scaler_y, (train_r2, test_r2)


def plot_trend(gpr, scaler_X, scaler_y, X_scaled, y_scaled, best_method, best_kernel, r2_train, r2_test):
    X_pred_original = np.linspace(cfg.PRED_ANGLE_MIN, cfg.PRED_ANGLE_MAX, cfg.PRED_POINTS).reshape(-1, 1)
    X_pred_scaled = scaler_X.transform(X_pred_original)
    y_pred_scaled = gpr.predict(X_pred_scaled)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    plt.figure(figsize=cfg.FIG_SIZE)
    plt.scatter(
        scaler_X.inverse_transform(X_scaled),
        scaler_y.inverse_transform(y_scaled.reshape(-1, 1)),
        color="black",
        alpha=0.6,
        label="Filtered Data",
        marker="x",
    )
    plt.plot(
        X_pred_original.ravel(),
        y_pred_original,
        color="red",
        linewidth=cfg.LINE_WIDTH_TREND,
        label="GPR Trend",
    )
    plt.title(
        f"Gaussian Process Regression (Method: {best_method}, Kernel: {best_kernel})\n"
        f"Train R2: {r2_train:.3f}, Test R2: {r2_test:.3f}",
        fontsize=cfg.FONT_SIZE_TITLE,
    )
    plt.xlabel(cfg.X_LABEL_GPR, fontsize=cfg.FONT_SIZE_LABEL)
    plt.ylabel(cfg.Y_LABEL_GPR, fontsize=cfg.FONT_SIZE_LABEL)
    if not cfg.AUTO_X_LIMITS_GPR and cfg.X_LIM_GPR is not None:
        plt.xlim(cfg.X_LIM_GPR[0], cfg.X_LIM_GPR[1])
    if not cfg.AUTO_Y_LIMITS_GPR and cfg.Y_LIM_GPR is not None:
        plt.ylim(cfg.Y_LIM_GPR[0], cfg.Y_LIM_GPR[1])
    plt.legend(fontsize=cfg.FONT_SIZE_LEGEND, loc="lower right")
    plt.tight_layout()
    save_plot("Figure_5_GPR_Trend")
    plt.show()


def plot_true_vs_pred(gpr, scaler_y, X_train, y_train, r2_train):
    y_train_pred = gpr.predict(X_train)
    y_train_true_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()

    plt.figure(figsize=cfg.FIG_SIZE_SCATTER)
    plt.scatter(
        y_train_true_original,
        y_train_pred_original,
        alpha=0.7,
        marker="x",
        label="Training Data",
    )
    if cfg.TRUE_VS_PRED_LIMITS is not None:
        limit_min, limit_max = cfg.TRUE_VS_PRED_LIMITS
        plt.xlim(limit_min, limit_max)
        plt.ylim(limit_min, limit_max)
        plt.plot([limit_min, limit_max], [limit_min, limit_max], ls="--", color="red", label="y = x")
    else:
        y_min = y_train_true_original.min()
        y_max = y_train_true_original.max()
        plt.plot([y_min, y_max], [y_min, y_max], ls="--", color="red", label="y = x")

    plt.xlabel("True Values (eV)", fontsize=cfg.FONT_SIZE_LABEL)
    plt.ylabel("Predicted Values (eV)", fontsize=cfg.FONT_SIZE_LABEL)
    plt.title(f"True vs Predicted Values (Training, R2: {r2_train:.3f})", fontsize=cfg.FONT_SIZE_TITLE)
    plt.legend(fontsize=cfg.FONT_SIZE_LEGEND)
    plt.tight_layout()
    save_plot("Figure_6_True_vs_Predicted")
    plt.show()


def plot_residuals(gpr, X_train, X_test, y_train, y_test):
    y_train_pred = gpr.predict(X_train)
    y_test_pred = gpr.predict(X_test)
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred

    plt.figure(figsize=cfg.FIG_SIZE_HIST)
    plt.hist(residuals_train, bins=cfg.HIST_BINS, alpha=0.6, label="Train Residuals")
    plt.hist(residuals_test, bins=cfg.HIST_BINS, alpha=0.6, label="Test Residuals")
    plt.axvline(0, color="red", linestyle="dashed", linewidth=2)
    if cfg.RESIDUAL_X_LIMITS is not None:
        plt.xlim(cfg.RESIDUAL_X_LIMITS[0], cfg.RESIDUAL_X_LIMITS[1])
    plt.title("Residuals Distribution (Scaled)", fontsize=cfg.FONT_SIZE_TITLE)
    plt.xlabel("Residuals (Scaled)", fontsize=cfg.FONT_SIZE_LABEL)
    plt.ylabel("Frequency", fontsize=cfg.FONT_SIZE_LABEL)
    plt.legend(fontsize=cfg.FONT_SIZE_LEGEND)
    plt.tight_layout()
    save_plot("Figure_7_Residuals_Distribution")
    plt.show()


# ------------------------
# Main
# ------------------------
def main():
    X, y = load_data()
    print(f"Data loaded. Points: {len(y)}")

    results = evaluate_methods(X, y)
    best_method, best_kernel = pick_best(results)

    gpr, scaler_X, scaler_y, (train_r2, test_r2) = train_final(X, y, best_method, best_kernel)

    X_f, y_f = remove_outliers(X, y, method=best_method)
    if cfg.BINNING_ENABLED:
        X_f, y_f = apply_sliding_window(X_f, y_f)
    X_scaled = scaler_X.transform(X_f)
    y_scaled = scaler_y.transform(y_f.reshape(-1, 1)).ravel()

    plot_trend(gpr, scaler_X, scaler_y, X_scaled, y_scaled, best_method, best_kernel, train_r2, test_r2)
    plot_true_vs_pred(gpr, scaler_y, scaler_X.transform(X_f), scaler_y.transform(y_f.reshape(-1, 1)).ravel(), train_r2)
    plot_residuals(gpr, scaler_X.transform(X_f), scaler_X.transform(X_f), scaler_y.transform(y_f.reshape(-1, 1)).ravel(), scaler_y.transform(y_f.reshape(-1, 1)).ravel())


if __name__ == "__main__":
    main()
