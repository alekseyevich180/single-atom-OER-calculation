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

def remove_outliers(X, y, method='zscore'):
    """使用 Z-score, IQR, 或 MAD 方法移除异常值。"""
    
    if method == 'zscore':
        y_zscores = zscore(y)
        mask = np.abs(y_zscores) < cfg.Z_THRESHOLD # 使用更新后的 Z_THRESHOLD

    elif method == 'iqr':
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        # 使用更新后的 IQR_MULTIPLIER (例如 2.0)
        lower_bound = q1 - cfg.IQR_MULTIPLIER * iqr 
        upper_bound = q3 + cfg.IQR_MULTIPLIER * iqr
        mask = (y >= lower_bound) & (y <= upper_bound)

    elif method == 'mad':
        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))
        if mad_y == 0:
            mask = np.ones_like(y, dtype=bool)
        else:
            # 使用更新后的 MAD_THRESHOLD (例如 3.0)
            mask = np.abs(y - median_y) / (1.4826 * mad_y) < cfg.MAD_THRESHOLD 

    else:
        raise ValueError("Invalid method. Choose 'zscore', 'iqr', or 'mad'.")

    return X[mask], y[mask]

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
# ==========================
# 2. NEW: Process data in chunks
# ==========================
chunk_size = 120
num_chunks = int(np.ceil(len(y) / chunk_size))
print(f"Data will be processed in {num_chunks} chunks of size {chunk_size}.")

# --- Use the first configured method and find the best kernel using the first chunk ---
best_method = cfg.DETECTION_METHODS[0]
print(f"\n--- Using outlier detection method: '{best_method}' ---")
print("--- Finding best kernel using the first chunk ---")

X_chunk_1 = X[:chunk_size]
y_chunk_1 = y[:chunk_size]

# Remove outliers from the first chunk
X_filtered_1, y_filtered_1 = remove_outliers(X_chunk_1, y_chunk_1, method=best_method)

best_kernel_name = None
best_test_r2 = -np.inf

# Only try to find best kernel if there's enough data after filtering
if len(y_filtered_1) > 10: 
    scaler_X_temp = StandardScaler()
    scaler_y_temp = StandardScaler()
    X_scaled = scaler_X_temp.fit_transform(X_filtered_1)
    y_scaled = scaler_y_temp.fit_transform(y_filtered_1.reshape(-1, 1)).ravel()

    # Simple train-test split for quick evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=cfg.RANDOM_STATE
    )

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
                test_r2 = gpr.score(X_test, y_test)
                print(f"  Kernel: {name}, Test R²: {test_r2:.3f}")
                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_kernel_name = name
            except ValueError as e:
                print(f"  Warning: Kernel {name} fitting failed on first chunk: {e}")

# Fallback to the first kernel if no best kernel was found
if best_kernel_name is None:
    best_kernel_name = list(cfg.KERNELS.keys())[0]
    print(f"Warning: Could not determine a best kernel. Falling back to default: {best_kernel_name}")

print("----------------------------------------")
print(f"✨ Best kernel from first chunk: {best_kernel_name} (Test R²: {best_test_r2:.3f})")
print("This configuration will be used for all chunks.")


# ==========================
# 3. Process each chunk and plot
# ==========================
plt.figure(figsize=cfg.FIG_SIZE)
colors = plt.cm.jet(np.linspace(0, 1, num_chunks))

final_kernel = cfg.KERNELS[best_kernel_name]

for i in range(num_chunks):
    print(f"\n--- Processing Chunk {i+1}/{num_chunks} ---")
    start_index = i * chunk_size
    end_index = start_index + chunk_size
    X_chunk = X[start_index:end_index]
    y_chunk = y[start_index:end_index]

    if len(y_chunk) == 0:
        continue

    # 1. Remove outliers within the chunk
    X_filtered, y_filtered = remove_outliers(X_chunk, y_chunk, method=best_method)
    print(f"Removed {len(y_chunk) - len(y_filtered)} outliers using '{best_method}' method.")
    
    if len(y_filtered) < 2:
        print("Not enough data points to train GPR for this chunk.")
        continue

    # 2. Standardize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_filtered)
    y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

    # 3. Train GPR model
    gpr = GaussianProcessRegressor(
        kernel=final_kernel,
        n_restarts_optimizer=cfg.FINAL_N_RESTARTS,
        alpha=cfg.FINAL_ALPHA,
        normalize_y=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gpr.fit(X_scaled, y_scaled)
    
    train_r2 = gpr.score(X_scaled, y_scaled)
    print(f"Train R² for chunk {i+1}: {train_r2:.3f}")

    # 4. Generate predictions for trend line
    chunk_angle_min = X_filtered.min()
    chunk_angle_max = X_filtered.max()
    X_pred_original = np.linspace(chunk_angle_min, chunk_angle_max, 200).reshape(-1, 1) # Use 200 points for trend
    X_pred_scaled = scaler_X.transform(X_pred_original)
    
    y_pred_scaled = gpr.predict(X_pred_scaled)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 5. Plotting
    # Plot filtered data for the chunk
    plt.scatter(
        X_filtered,
        y_filtered,
        color='grey', # Keep scatter points neutral
        alpha=0.4,
        marker='x'
    )
    # Plot GPR trend for the chunk
    plt.plot(
        X_pred_original.ravel(),
        y_pred_original,
        color=colors[i],
        linewidth=cfg.LINE_WIDTH_TREND,
        label=f'Chunk {i+1} Trend'
    )

# Add a placeholder scatter for the legend
plt.scatter([], [], color='grey', alpha=0.4, marker='x', label='Filtered Data')

# --- Final plot configuration ---
plt.title(f'GPR Analysis by Chunks (Method: {best_method}, Kernel: {best_kernel_name})', fontsize=cfg.FONT_SIZE_TITLE)
plt.xlabel(cfg.X_LABEL_GPR, fontsize=cfg.FONT_SIZE_LABEL)
plt.ylabel(cfg.Y_LABEL_GPR, fontsize=cfg.FONT_SIZE_LABEL)

# --- Axis limits control ---
if not cfg.AUTO_X_LIMITS_GPR and cfg.X_LIM_GPR is not None:
    plt.xlim(cfg.X_LIM_GPR[0], cfg.X_LIM_GPR[1])
else:
    # Auto-adjust x-axis to fit all data if not manually set
    plt.xlim(X.min(), X.max())

if not cfg.AUTO_Y_LIMITS_GPR and cfg.Y_LIM_GPR is not None:
    plt.ylim(cfg.Y_LIM_GPR[0], cfg.Y_LIM_GPR[1])

plt.legend(fontsize=cfg.FONT_SIZE_LEGEND, loc='best')
plt.tight_layout()

# --- Auto save ---
save_plot("Figure_5_GPR_Trend_Chunked")
plt.show()

# NOTE: The True vs. Predicted and Residual plots are omitted as they are less meaningful
# in a chunked analysis where models are retrained for each chunk.

