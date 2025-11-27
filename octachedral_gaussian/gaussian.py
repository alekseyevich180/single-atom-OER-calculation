import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import zscore
from sklearn.metrics import r2_score
from scipy.optimize import fmin_l_bfgs_b
import warnings
from sklearn.exceptions import ConvergenceWarning


# 异常值检测函数
# method: 'zscore', 'iqr', or 'mad'
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
        mask = np.abs(y - median_y) / mad_y < mad_threshold

    else:
        raise ValueError("Invalid method. Choose 'zscore', 'iqr', or 'mad'.")

    return X[mask], y[mask]

# 1. データを准备する
data = np.loadtxt('Mn.dat', delimiter=',')
X = data[:, 1].reshape(-1, 1)
y = -data[:, 0]

# 比较三种异常值检测方法
detection_methods = ['zscore', 'iqr', 'mad']
results = {}

for method in detection_methods:
    # 异常值检测
    X_filtered, y_filtered = remove_outliers(X, y, method=method)

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_filtered)
    y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # 核函数优化
    kernels = {
        "RBF": C(1.0) * RBF(length_scale=5.0),
        "RBF2" : C(1.5, (1e-1, 1e3)) * RBF(length_scale=2.0, length_scale_bounds=(1e-6, 1e6)),
        "Matern": C(1.0) * Matern(length_scale=2.0, nu=1.5),
        "kernel_matern" : C(1.2, (1e-3, 1e4)) * Matern(length_scale=20.0, nu=1.5), 
        "kernel_matern2" : C(1.0, (1e-1, 1e2)) * Matern(length_scale=10.0, length_scale_bounds=(1e-6, 1e3), nu=2.5),
        "Matern + WhiteKernel" : C(1.0, (1e-3, 1e5)) * Matern(length_scale=12.0, nu=1.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e1)),
        "Matern + WhiteKernel2": C(1.0) * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=1e-6),
        "kernel": C(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e2)) + WhiteKernel(noise_level=1e-6),
        "Matern + WhiteKernel + RBF": C(1.0) * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=1e-6) + RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e3)),
        #"kernel" : gp_kern.RBF(num) * gp_kern.Bias(num) + gp_kern.Linear(num) * gp_kern.Bias(num)
        #model = GPy.models.GPRegression(X.values, y.values, kernel=kernel, normalizer=True)
        #model.optimize()
        "Matern + WhiteKernel + RBF2": C(1.5, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-7, noise_level_bounds=(1e-8, 1e-1)) + C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-6, 1e4)),

    }

    method_results = []

    for name, kernel in kernels.items():
        gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                alpha=1e-2,
                
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gpr.fit(X_train, y_train)  
       
       
        gpr.fit(X_train, y_train)

        log_likelihood = gpr.log_marginal_likelihood_value_
        num_params = gpr.kernel_.n_dims
        aic = 2 * num_params - 2 * log_likelihood

        # 使用交叉验证计算 R² 分数
        scores = cross_val_score(gpr, X_scaled, y_scaled, cv=5, scoring='r2')
        mean_r2 = scores.mean()

        method_results.append((name, aic, mean_r2))

    # 保存每种方法的所有核函数的 AIC 和 R² 值
    results[method] = {
        'method_results': method_results,
        'data_points': len(y_filtered)
    }

# 输出比较结果并选择最佳方法和核函数
best_method = None
best_kernel = None
best_aic = float('inf')
best_r2 = -float('inf')

for method, result in results.items():
    print(f"Method: {method}")
    for kernel_name, aic_value, r2_value in result['method_results']:
        print(f"  Kernel: {kernel_name}, AIC: {aic_value:.3f}, Mean R²: {r2_value:.3f}")
        if aic_value < best_aic and r2_value > best_r2:
            best_aic = aic_value
            best_r2 = r2_value
            best_method = method
            best_kernel = kernel_name
    print(f"  Remaining Data Points: {result['data_points']}")
    print("----------------------------------------")

# 选择最佳方法重新训练模型并可视化预测结果
print(f"Best Method: {best_method}, Best Kernel: {best_kernel}, Best AIC: {best_aic:.3f}, Best Mean R²: {best_r2:.3f}")
X_filtered, y_filtered = remove_outliers(X, y, method=best_method)

def remove_outliers(X, y, method='zscore', z_threshold=2, iqr_multiplier=1, mad_threshold=2):
    if method == 'zscore':
        y_zscores = zscore(y)
        mask = np.abs(y_zscores) < z_threshold


# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_filtered)
y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 使用最佳核函数重新训练
kernel = kernels[best_kernel]
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=1e-4, optimizer="fmin_l_bfgs_b")
#'fmin_l_bfgs_b'	默认优化器，适合大多数场景，使用 L-BFGS-B 算法。处理边界约束问题表现良好。
#'fmin_tnc'	基于牛顿共轭梯度法的优化器。适合较大的数据集，但对初值敏感。
#'fmin_cg'	共轭梯度法优化器。适合大型优化问题，但对精度要求较高时可能表现不佳。
#'fmin_powell'	基于 Powell 的方法，适合无梯度的优化问题，但速度较慢。
#'fmin_bfgs'	传统的 BFGS 优化器，无边界限制。适合小型数据集的精确优化。
#'fmin_ncg'	使用牛顿-共轭梯度法进行优化。对小型优化问题效果好，但需要计算二阶导数。
#'trust-constr'	基于信任区域的优化器。适用于带有约束条件的优化问题。
gpr.fit(X_train, y_train)

# 可视化预测结果
#X_pred = np.linspace(X_scaled.min(), X_scaled.max(), 1000).reshape(-1, 1)
X_pred = np.concatenate([
    np.linspace(X_scaled.min(), X_scaled.max(), 800),
    np.linspace(175, 180, 200)  # 增加边界点的密度
]).reshape(-1, 1)
y_pred, sigma = gpr.predict(X_pred, return_std=True)

X_pred_original = scaler_X.inverse_transform(X_pred).ravel()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
sigma_original = scaler_y.scale_ * sigma

train_r2 = r2_score(y_train, gpr.predict(X_train))
test_r2 = r2_score(y_test, gpr.predict(X_test))
#90%： 1.645 95%：1.960 99%：2.58 99.9%：3.291

plt.figure(figsize=(8, 6))
plt.scatter(scaler_X.inverse_transform(X_scaled), scaler_y.inverse_transform(y_scaled.reshape(-1, 1)), color='black', alpha=0.6, label='Filtered Data',marker = 'x')
plt.plot(X_pred_original, y_pred_original, color='red', label='Mean Prediction', linewidth=2)
plt.fill_between(X_pred_original, y_pred_original - 1.64 * sigma_original, y_pred_original + 1.64 * sigma_original, color='lightblue', alpha=0.5, label='Confidence Interval')
plt.title(f'Gaussian Process Regression \nTrain R²: {train_r2:.3f}, Test R²: {test_r2:.3f}',fontsize=16)
plt.xlabel('O-Ir-O angle (°)',fontsize=12)
plt.ylabel('-IpCOHP (eV)',fontsize=12)
plt.ylim(0.8,1.5)
plt.xlim(140,180)
plt.legend(fontsize=12,loc='lower right')
plt.tight_layout()
plt.show()

# 7. 真值 vs 预测值
# 真值 vs 预测值 (训练集)
y_train_pred = gpr.predict(X_train)
y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()  # 修复：reshape(-1, 1)

# 可视化
plt.figure(figsize=(6, 6))
plt.scatter(
    scaler_y.inverse_transform(y_train.reshape(-1, 1)),
    y_train_pred_original,
    alpha=0.7,
    marker='x',
    label="Training Data"
)
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    ls="--",
    color="red",
    label="y = x"
)
plt.xlabel("True Values (eV)")
plt.ylabel("Predicted Values (eV)")
plt.title("True vs Predicted Values (Training)")
plt.legend()
plt.tight_layout()
plt.show()

y_train_pred = gpr.predict(X_train)
y_test_pred = gpr.predict(X_test)
# Calculate residuals for train and test sets
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# Plot residuals distribution (histogram and density plot)
plt.figure(figsize=(12, 6))

# Histogram for residuals
plt.hist(residuals_train, bins=30, alpha=0.6, color='blue', label='Train Residuals')
plt.hist(residuals_test, bins=30, alpha=0.6, color='orange', label='Test Residuals')

# Vertical line for zero residuals
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)

# Title and labels
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
