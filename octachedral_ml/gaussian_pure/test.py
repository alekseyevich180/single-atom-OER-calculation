# gpr_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import zscore
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import os # 导入 os 模块用于文件操作
# 导入参数配置文件
import gpr_config as cfg

# ----------------------------------------
# ⚙️ 核心函数定义
# ----------------------------------------

# ==========================
# 异常值检测函数 (使用 cfg 中的参数)
# ==========================
def remove_outliers(X, y, method='zscore'):
    """使用 Z-score, IQR, 或 MAD 方法移除异常值。"""
    
    if method == 'zscore':
        y_zscores = zscore(y)
        mask = np.abs(y_zscores) < cfg.Z_THRESHOLD

    elif method == 'iqr':
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - cfg.IQR_MULTIPLIER * iqr
        upper_bound = q3 + cfg.IQR_MULTIPLIER * iqr
        mask = (y >= lower_bound) & (y <= upper_bound)

    elif method == 'mad':
        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))
        if mad_y == 0:
            mask = np.ones_like(y, dtype=bool)
        else:
            mask = np.abs(y - median_y) / (1.4826 * mad_y) < cfg.MAD_THRESHOLD

    else:
        raise ValueError("Invalid method. Choose 'zscore', 'iqr', or 'mad'.")

    return X[mask], y[mask]

# ----------------------------------------
# 图像保存函数
# ----------------------------------------
def save_plot(filename):
    """根据配置保存当前 Matplotlib 图像。"""
    if cfg.SAVE_PLOTS:
        # 确保保存目录存在
        if not os.path.exists(cfg.SAVE_DIR):
            os.makedirs(cfg.SAVE_DIR)
            
        full_path = os.path.join(cfg.SAVE_DIR, f"{filename}.{cfg.FILE_FORMAT}")
        try:
            plt.savefig(full_path, dpi=cfg.DPI, bbox_inches='tight')
            print(f"图像已保存至: {full_path}")
        except Exception as e:
            print(f"保存图像 {filename} 失败: {e}")

# ==========================
# 1. 读取数据
# ==========================
try:
    data = np.loadtxt(cfg.FILE_NAME, delimiter=',', skiprows=cfg.SKIP_ROWS, usecols=cfg.USE_COLS)
except FileNotFoundError:
    print(f"错误：找不到文件 {cfg.FILE_NAME}。请确保文件存在于同一目录下。")
    exit()

X = data[:, cfg.FEATURE_COL].reshape(-1, 1)
y = data[:, cfg.TARGET_COL] * cfg.TARGET_SIGN
print(f"数据加载完成。总数据点: {len(y)}")

# ==========================
# 2. 比较异常值检测方法 + 不同 kernel
# ==========================
results = {}

for method in cfg.DETECTION_METHODS:
    X_filtered, y_filtered = remove_outliers(X, y, method=method)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_filtered)
    y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
    )

    method_results = []

    for name, kernel in cfg.KERNELS.items():
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=cfg.INITIAL_N_RESTARTS,
            alpha=cfg.INITIAL_ALPHA,
            normalize_y=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            try:
                gpr.fit(X_train, y_train)
            except ValueError as e:
                print(f"Warning: {method}/{name} fitting failed: {e}")
                mean_r2 = -np.inf
                aic = np.inf
                method_results.append((name, aic, mean_r2))
                continue

            try:
                log_likelihood = gpr.log_marginal_likelihood_value_
            except AttributeError:
                log_likelihood = gpr.log_marginal_likelihood(gpr.kernel_.theta)

            num_params = gpr.kernel_.theta.size
            aic = 2 * num_params - 2 * log_likelihood

            scores = cross_val_score(gpr, X_scaled, y_scaled, cv=cfg.CV_FOLDS, scoring='r2')
            mean_r2 = scores.mean()

            method_results.append((name, aic, mean_r2))

    results[method] = {
        'method_results': method_results,
        'data_points': len(y_filtered)
    }

# ==========================
# 3. 选择最佳方法和核函数
# ==========================
best_method = None
best_kernel = None
best_aic = np.inf
best_r2 = -np.inf

print("\n--- 异常值检测方法和核函数比较 ---")
for method, result in results.items():
    print(f"方法: {method} (剩余数据点: {result['data_points']})")
    for kernel_name, aic_value, r2_value in result['method_results']:
        print(f"  核函数: {kernel_name}, AIC: {aic_value:.3f}, Mean R²: {r2_value:.3f}")
        
        if r2_value > best_r2:
            best_r2 = r2_value
            best_aic = aic_value
            best_method = method
            best_kernel = kernel_name
        elif np.isclose(r2_value, best_r2) and aic_value < best_aic:
            best_aic = aic_value
            best_method = method
            best_kernel = kernel_name

print("----------------------------------------")
print(f"\n✨ 最佳配置:")
print(f"方法: {best_method}, 核函数: {best_kernel}")
print(f"最佳 AIC: {best_aic:.3f}, 最佳平均 R²: {best_r2:.3f}")

# ==========================
# 4. 用最佳方法和核函数重新训练 + 评估
# ==========================
X_filtered, y_filtered = remove_outliers(X, y, method=best_method)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_filtered)
y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
)

final_kernel = cfg.KERNELS[best_kernel]
gpr = GaussianProcessRegressor(
    kernel=final_kernel,
    n_restarts_optimizer=cfg.FINAL_N_RESTARTS,
    alpha=cfg.FINAL_ALPHA,
    normalize_y=True
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    gpr.fit(X_train, y_train)

train_r2 = r2_score(y_train, gpr.predict(X_train))
test_r2 = r2_score(y_test, gpr.predict(X_test))
print(f"\nFinal Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
print("Optimized kernel parameters:\n", gpr.kernel_)

# ==========================
# 5. 可视化：散点 + 趋势线
# ==========================
X_pred_original = np.linspace(cfg.PRED_ANGLE_MIN, cfg.PRED_ANGLE_MAX, cfg.PRED_POINTS).reshape(-1, 1)
X_pred_scaled = scaler_X.transform(X_pred_original)

y_pred_scaled = gpr.predict(X_pred_scaled)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

plt.figure(figsize=cfg.FIG_SIZE)

plt.scatter(
    scaler_X.inverse_transform(X_scaled),
    scaler_y.inverse_transform(y_scaled.reshape(-1, 1)),
    color='black',
    alpha=0.6,
    label='Filtered Data',
    marker='x'
)

plt.plot(
    X_pred_original.ravel(),
    y_pred_original,
    color='red',
    linewidth=cfg.LINE_WIDTH_TREND,
    label='GPR Trend'
)

plt.title(f'Gaussian Process Regression (Method: {best_method}, Kernel: {best_kernel})\nTrain R²: {train_r2:.3f}, Test R²: {test_r2:.3f}', fontsize=cfg.FONT_SIZE_TITLE)
plt.xlabel(cfg.X_LABEL_GPR, fontsize=cfg.FONT_SIZE_LABEL)
plt.ylabel(cfg.Y_LABEL_GPR, fontsize=cfg.FONT_SIZE_LABEL)


# --- 轴限制控制 (图 5) ---
if not cfg.AUTO_X_LIMITS_GPR and cfg.X_LIM_GPR is not None:
    plt.xlim(cfg.X_LIM_GPR[0], cfg.X_LIM_GPR[1])

if not cfg.AUTO_Y_LIMITS_GPR and cfg.Y_LIM_GPR is not None:
    plt.ylim(cfg.Y_LIM_GPR[0], cfg.Y_LIM_GPR[1])
# ------------------------

plt.legend(fontsize=cfg.FONT_SIZE_LEGEND, loc='lower right')
plt.tight_layout()

# --- 自动保存 ---
save_plot("Figure_5_GPR_Trend")
plt.show() # 显示图像

# ==========================
# 6. 真值 vs 预测值（训练集）
# ==========================
y_train_pred = gpr.predict(X_train)
y_train_true_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()

plt.figure(figsize=cfg.FIG_SIZE_SCATTER)
plt.scatter(
    y_train_true_original,
    y_train_pred_original,
    alpha=0.7,
    marker='x',
    label="Training Data"
)
y_min = y_train_true_original.min()
y_max = y_train_true_original.max()

# --- 轴限制控制 (图 6) ---
if cfg.TRUE_VS_PRED_LIMITS is not None:
    limit_min, limit_max = cfg.TRUE_VS_PRED_LIMITS
    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)
    # y=x 线也使用手动限制
    plt.plot([limit_min, limit_max], [limit_min, limit_max], ls="--", color="red", label="y = x")
else:
    # 自动适应
    plt.plot([y_min, y_max], [y_min, y_max], ls="--", color="red", label="y = x")
# ------------------------

plt.xlabel("True Values (eV)", fontsize=cfg.FONT_SIZE_LABEL)
plt.ylabel("Predicted Values (eV)", fontsize=cfg.FONT_SIZE_LABEL)
plt.title(f"True vs Predicted Values (Training, R²: {train_r2:.3f})", fontsize=cfg.FONT_SIZE_TITLE)
plt.legend(fontsize=cfg.FONT_SIZE_LEGEND)
plt.tight_layout()

save_plot("Figure_6_True_vs_Predicted")
plt.show()

# ==========================
# 7. 残差分布
# ==========================
y_train_pred = gpr.predict(X_train)
y_test_pred = gpr.predict(X_test)
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=cfg.FIG_SIZE_HIST)
plt.hist(residuals_train, bins=cfg.HIST_BINS, alpha=0.6, label='Train Residuals')
plt.hist(residuals_test, bins=cfg.HIST_BINS, alpha=0.6, label='Test Residuals')
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)

# --- X 轴限制控制 (图 7) ---
if cfg.RESIDUAL_X_LIMITS is not None:
    plt.xlim(cfg.RESIDUAL_X_LIMITS[0], cfg.RESIDUAL_X_LIMITS[1])
# ------------------------

plt.title('Residuals Distribution (Scaled)', fontsize=cfg.FONT_SIZE_TITLE)
plt.xlabel('Residuals (Scaled)', fontsize=cfg.FONT_SIZE_LABEL)
plt.ylabel('Frequency', fontsize=cfg.FONT_SIZE_LABEL)
plt.legend(fontsize=cfg.FONT_SIZE_LEGEND)
plt.tight_layout()
save_plot("Figure_7_Residuals_Distribution")
plt.show()

