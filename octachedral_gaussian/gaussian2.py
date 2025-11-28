import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import zscore
from sklearn.metrics import r2_score
import warnings
from sklearn.exceptions import ConvergenceWarning


# ==========================
# 异常值检测函数
# method: 'zscore', 'iqr', or 'mad'
# ==========================
def remove_outliers(X, y, method='zscore', z_threshold=2, iqr_multiplier=1, mad_threshold=2):
    if method == 'zscore':
        # Z-score 方法
        y_zscores = zscore(y)
        mask = np.abs(y_zscores) < z_threshold

    elif method == 'iqr':
        # IQR 方法
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        mask = (y >= lower_bound) & (y <= upper_bound)

    elif method == 'mad':
        # MAD 方法
        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))
        # 避免 mad 为 0 的除零问题
        if mad_y == 0:
            mask = np.ones_like(y, dtype=bool)
        else:
            mask = np.abs(y - median_y) / mad_y < mad_threshold

    else:
        raise ValueError("Invalid method. Choose 'zscore', 'iqr', or 'mad'.")

    return X[mask], y[mask]


# ==========================
# 1. 读取数据
# 假设 Mn.dat 格式类似：
# -1.23502,180,89.92
# 用逗号分隔
# ==========================
data = np.loadtxt('Ir.dat', delimiter=',')

# 假设：
# 第 0 列：IpCOHP
# 第 1 列：O–Ir–O 角度
X = data[:, 1].reshape(-1, 1)  # O–Ir–O angle
y = -data[:, 0]                # -IpCOHP (使其越大越稳定之类)

# ==========================
# 2. 比较三种异常值检测方法 + 不同 kernel
# ==========================
detection_methods = ['zscore', 'iqr', 'mad']
results = {}

# 定义一组候选核函数
kernels = {
    "RBF": C(1.0) * RBF(length_scale=5.0),
    "RBF2": C(1.5, (1e-1, 1e3)) * RBF(length_scale=2.0, length_scale_bounds=(1e-6, 1e6)),
    "Matern": C(1.0) * Matern(length_scale=2.0, nu=1.5),
    "kernel_matern": C(1.2, (1e-3, 1e4)) * Matern(length_scale=20.0, nu=1.5),
    "kernel_matern2": C(1.0, (1e-1, 1e2)) * Matern(length_scale=10.0, length_scale_bounds=(1e-6, 1e3), nu=2.5),
    "Matern + WhiteKernel": C(1.0, (1e-3, 1e5)) * Matern(length_scale=12.0, nu=1.5) +
                             WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e1)),
    "Matern + WhiteKernel2": C(1.0) * Matern(length_scale=10.0, nu=1.5) +
                              WhiteKernel(noise_level=1e-6),
    "kernel": C(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e2)) +
              WhiteKernel(noise_level=1e-6),
    "Matern + WhiteKernel + RBF": C(1.0) * Matern(length_scale=2.0, nu=1.5) +
                                  WhiteKernel(noise_level=1e-6) +
                                  RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e3)),
    "Matern + WhiteKernel + RBF2": C(1.5, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5,
                                                                 length_scale_bounds=(1e-3, 1e3)) +
                                   WhiteKernel(noise_level=1e-7, noise_level_bounds=(1e-8, 1e-1)) +
                                   C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-6, 1e4)),
}

for method in detection_methods:
    # 2.1 异常值检测
    X_filtered, y_filtered = remove_outliers(X, y, method=method)

    # 2.2 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_filtered)
    y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

    # 2.3 划分训练 / 测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    method_results = []

    for name, kernel in kernels.items():
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,   # 初步比较时不用太大
            alpha=1e-2,               # 稍大的噪声，可以防过拟合
            normalize_y=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gpr.fit(X_train, y_train)

            # 对数边际似然
            try:
                log_likelihood = gpr.log_marginal_likelihood_value_
            except AttributeError:
                log_likelihood = gpr.log_marginal_likelihood(gpr.kernel_.theta)

            num_params = gpr.kernel_.theta.size
            aic = 2 * num_params - 2 * log_likelihood

            # 交叉验证 R²
            scores = cross_val_score(gpr, X_scaled, y_scaled, cv=5, scoring='r2')
            mean_r2 = scores.mean()

            method_results.append((name, aic, mean_r2))

    results[method] = {
        'method_results': method_results,
        'data_points': len(y_filtered)
    }

# ==========================
# 3. 选择最佳方法和核函数
#    优先：R² 最大，其次：AIC 最小
# ==========================
best_method = None
best_kernel = None
best_aic = None
best_r2 = -np.inf

for method, result in results.items():
    print(f"Method: {method}")
    for kernel_name, aic_value, r2_value in result['method_results']:
        print(f"  Kernel: {kernel_name}, AIC: {aic_value:.3f}, Mean R²: {r2_value:.3f}")
        if (r2_value > best_r2) or (np.isclose(r2_value, best_r2) and (best_aic is None or aic_value < best_aic)):
            best_aic = aic_value
            best_r2 = r2_value
            best_method = method
            best_kernel = kernel_name
    print(f"  Remaining Data Points: {result['data_points']}")
    print("----------------------------------------")

print(f"\nBest Method: {best_method}, Best Kernel: {best_kernel}, "
      f"Best AIC: {best_aic:.3f}, Best Mean R²: {best_r2:.3f}")

# ==========================
# 4. 用最佳方法和核函数重新训练 + 可视化
# ==========================

# 使用最佳异常值处理
X_filtered, y_filtered = remove_outliers(X, y, method=best_method)

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_filtered)
y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

# 训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# 最终 GPR 模型（参数稍微优化）
final_kernel = kernels[best_kernel]
gpr = GaussianProcessRegressor(
    kernel=final_kernel,
    n_restarts_optimizer=20,   # 适中，避免太慢
    alpha=1e-5,                # 更小噪声，更贴数据
    normalize_y=True
)
gpr.fit(X_train, y_train)

train_r2 = r2_score(y_train, gpr.predict(X_train))
test_r2 = r2_score(y_test, gpr.predict(X_test))
print(f"\nFinal Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
print("Optimized kernel:\n", gpr.kernel_)

# ==========================
# 5. 可视化：散点 + 趋势线（无置信区间）
# ==========================

# 在「原始角度」空间生成趋势线点
X_pred_original = np.linspace(140, 180, 1000).reshape(-1, 1)  # 你关心的角度范围
X_pred_scaled = scaler_X.transform(X_pred_original)

y_pred_scaled = gpr.predict(X_pred_scaled)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

plt.figure(figsize=(8, 6))

# 原始（去异常值后）数据点：散点
plt.scatter(
    scaler_X.inverse_transform(X_scaled),
    scaler_y.inverse_transform(y_scaled.reshape(-1, 1)),
    color='black',
    alpha=0.6,
    label='Filtered Data',
    marker='x'
)

# GPR 趋势线
plt.plot(
    X_pred_original.ravel(),
    y_pred_original,
    color='red',
    linewidth=2,
    label='GPR Trend'
)

plt.title(f'Gaussian Process Regression\nTrain R²: {train_r2:.3f}, Test R²: {test_r2:.3f}', fontsize=16)
plt.xlabel('O-Cr-O angle (°)', fontsize=12)
plt.ylabel('-IpCOHP (eV)', fontsize=12)
plt.ylim(1, 2.0)
plt.xlim(140, 180)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.show()

# ==========================
# 6. 真值 vs 预测值（训练集）
# ==========================
y_train_pred = gpr.predict(X_train)
y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()

plt.figure(figsize=(6, 6))
plt.scatter(
    scaler_y.inverse_transform(y_train.reshape(-1, 1)),
    y_train_pred_original,
    alpha=0.7,
    marker='x',
    label="Training Data"
)
y_min = scaler_y.inverse_transform(y_train.reshape(-1, 1)).min()
y_max = scaler_y.inverse_transform(y_train.reshape(-1, 1)).max()
plt.plot([y_min, y_max], [y_min, y_max], ls="--", color="red", label="y = x")

plt.xlabel("True Values (eV)")
plt.ylabel("Predicted Values (eV)")
plt.title("True vs Predicted Values (Training)")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================
# 7. 残差分布
# ==========================
y_train_pred = gpr.predict(X_train)
y_test_pred = gpr.predict(X_test)
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=(12, 6))
plt.hist(residuals_train, bins=30, alpha=0.6, label='Train Residuals')
plt.hist(residuals_test, bins=30, alpha=0.6, label='Test Residuals')
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)

plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
