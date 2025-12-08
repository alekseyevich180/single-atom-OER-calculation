# plot_all_trends.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import warnings
import os
import importlib.util

# ----------------------------------------
# ⚙️ 核心函数定义 (Core Function Definitions)
# ----------------------------------------

def remove_outliers(X, y, cfg, method='zscore'):
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

def get_trend_line_for_element(element_name):
    """
    为单个元素计算GPR趋势线。
    返回 (X_pred, y_pred)
    """
    print(f"--- 开始处理元素: {element_name} ---")

    # 1. 动态加载配置文件
    config_path = os.path.join(element_name, 'gpr_config.py')
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}。跳过此元素。")
        return None, None
        
    spec = importlib.util.spec_from_file_location(f"gpr_config_{element_name}", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    
    # 2. 读取数据
    data_path = os.path.join(element_name, cfg.FILE_NAME)
    try:
        data = np.loadtxt(data_path, delimiter=',', skiprows=cfg.SKIP_ROWS, usecols=cfg.USE_COLS)
    except FileNotFoundError:
        print(f"错误：找不到文件 {data_path}。跳过此元素。")
        return None, None

    X = data[:, cfg.FEATURE_COL].reshape(-1, 1)
    y = data[:, cfg.TARGET_COL] * cfg.TARGET_SIGN
    print(f"数据加载完成。总数据点: {len(y)}")

    # 3. 数据筛选和预处理
    if hasattr(cfg, 'ANGLE_FILTER_ENABLED') and cfg.ANGLE_FILTER_ENABLED:
        angle_mask = ((X >= cfg.ANGLE_MIN) & (X <= cfg.ANGLE_MAX)).flatten()
        X = X[angle_mask].reshape(-1, 1)
        y = y[angle_mask]
        print(f"数据筛选完成: 剩余 {len(y)} 个数据点。")

    # 使用一个固定的、鲁棒的策略：z-score 去异常值
    X_filtered, y_filtered = remove_outliers(X, y, cfg, method='zscore')

    # 滑动窗口平滑
    if hasattr(cfg, 'BINNING_ENABLED') and cfg.BINNING_ENABLED:
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

    # 4. 标准化
    scaler_X = StandardScaler().fit(X_filtered)
    scaler_y = StandardScaler().fit(y_filtered.reshape(-1, 1))
    X_scaled = scaler_X.transform(X_filtered)
    y_scaled = scaler_y.transform(y_filtered.reshape(-1, 1)).ravel()

    # 5. GPR模型训练 - 从每个元素的配置中动态选择核函数
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    
    if hasattr(cfg, 'KERNELS') and 'Matern_WK' in cfg.KERNELS:
        kernel_to_use = cfg.KERNELS['Matern_WK']
        print(f"信息: 使用 {element_name} 配置中的 'Matern_WK' 核函数。")
    else:
        print(f"警告: 在 {element_name} 的配置中未找到 'Matern_WK'。将使用默认的RBF核函数。")
        kernel_to_use = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1)

    gpr = GaussianProcessRegressor(
        kernel=kernel_to_use,
        n_restarts_optimizer=cfg.FINAL_N_RESTARTS,
        alpha=cfg.FINAL_ALPHA,
        normalize_y=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X_scaled, y_scaled)
        
    # 6. 生成预测趋势线
    X_pred_original = np.linspace(cfg.PRED_ANGLE_MIN, cfg.PRED_ANGLE_MAX, cfg.PRED_POINTS).reshape(-1, 1)
    X_pred_scaled = scaler_X.transform(X_pred_original)
    
    y_pred_scaled, y_std_scaled = gpr.predict(X_pred_scaled, return_std=True)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    print(f"--- 完成处理: {element_name} ---\n")
    return X_pred_original, y_pred_original

def main():
    """
    主函数：自动发现元素，循环处理并绘制组合图。
    """
    # 自动发现包含 'gpr_config.py' 的元素目录
    current_dir = '.'
    try:
        # 筛选出是目录且不以 '__' 开头或不含 '.' 的文件夹
        all_dirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith('__') and '.' not in d]
        elements = sorted([d for d in all_dirs if os.path.exists(os.path.join(d, 'gpr_config.py'))])
    except Exception as e:
        print(f"错误: 发现元素目录时出错: {e}")
        elements = []

    if not elements:
        print("错误: 未找到任何有效的元素目录 (需要包含 'gpr_config.py')。")
        return

    print(f"成功发现 {len(elements)} 个元素: {', '.join(elements)}")
    
    # 使用 'tab20' colormap 提供更多区分度
    colors = plt.cm.get_cmap('tab20', len(elements))

    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(8, 6))

    for i, element in enumerate(elements):
        X_pred, y_pred = get_trend_line_for_element(element)
        
        if X_pred is not None and y_pred is not None:
            plt.plot(
                X_pred.ravel(),
                y_pred,
                color=colors(i),
                linewidth=2.5,
                label=f'GPR Trend - {element.capitalize()}'
            )

    plt.title('Angle - ICOHP Trend Lines for Rutile Type Metal Elements', fontsize=18, fontweight='bold')
    plt.xlabel('O-M-O Angle (°)', fontsize=14)
    plt.ylabel('-ICOHP (eV)', fontsize=14)
    plt.xlim(130, 180.6)
    plt.ylim(0.6, 1.9)
    plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout(pad=1.5)

    # 保存图像
    save_path = "Combined_GPR_Trends_Optimized.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"\n✅ 优化后的组合趋势图已保存至: {save_path}")

    plt.show()

if __name__ == "__main__":
    main()
