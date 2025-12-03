# gpr_config.py

# ========================================
# ⚙️ 高斯过程回归 (GPR) 参数配置
# ========================================

# --- 1. 数据和特征 ---
FILE_NAME = 'Ru.dat'
FEATURE_COL = 2        # O–Ir–O 角度
TARGET_COL = 0         # IpCOHP
TARGET_SIGN = -1.0     # 目标值符号调整：-IpCOHP
SKIP_ROWS = 1
USE_COLS = range(4)

# --- 2. 异常值检测参数 ---
DETECTION_METHODS = ['zscore', 'iqr', 'mad']
Z_THRESHOLD = 2.0
IQR_MULTIPLIER = 1.5
MAD_THRESHOLD = 2.0

# --- 3. GPR 模型和训练参数 ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
INITIAL_ALPHA = 1e-2
INITIAL_N_RESTARTS = 5
FINAL_ALPHA = 1e-5
FINAL_N_RESTARTS = 20

# --- 4. 候选核函数定义 ---
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel

KERNELS = {
    "RBF": C(1.0) * RBF(length_scale=5.0),
    "RBF2": C(1.5, (1e-1, 1e3)) * RBF(length_scale=2.0, length_scale_bounds=(1e-6, 1e6)),
    "Matern_1.5": C(1.0) * Matern(length_scale=2.0, nu=1.5),
    "Matern_WK": C(1.0, (1e-3, 1e5)) * Matern(length_scale=12.0, nu=1.5) +
                 WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e1)),
    "Matern_WK_RBF": C(1.5, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5,
                                                   length_scale_bounds=(1e-3, 1e3)) +
                     WhiteKernel(noise_level=1e-7, noise_level_bounds=(1e-8, 1e-1)) +
                     C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-6, 1e4)),
}

# --- 5. 可视化参数 ---
# 通用绘图
FIG_SIZE = (8, 6)
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 12
LINE_WIDTH_TREND = 2

# GPR 趋势线图 (图 5)
PRED_ANGLE_MIN = 120.0   # 趋势线预测的最小角度 (用于生成趋势线数据)
PRED_ANGLE_MAX = 180.0   # 趋势线预测的最大角度 (用于生成趋势线数据)
PRED_POINTS = 1000       # 趋势线上的点数
X_LABEL_GPR = 'O-Ru-O angle (°)'
Y_LABEL_GPR = '-IpCOHP (eV)'

# --- 图 5 轴限制配置 ---
# AUTO_... = True: 自动适应数据范围 (忽略 X/Y_LIM_GPR)
# AUTO_... = False: 使用手动设置 X/Y_LIM_GPR
AUTO_X_LIMITS_GPR = False 
AUTO_Y_LIMITS_GPR = True 

# 手动设置的轴限制 (仅在 AUTO_... 为 False 时有效，格式为 (min, max))
X_LIM_GPR = (120.0, 181.0)
Y_LIM_GPR = (1.0, 2.0)


# 真值 vs 预测图 (图 6)
FIG_SIZE_SCATTER = (6, 6)
# 设置 X 轴和 Y 轴的统一限制，None 表示自动适应。
# 示例: TRUE_VS_PRED_LIMITS = (1.2, 1.8)
TRUE_VS_PRED_LIMITS = None 

# 残差图 (图 7)
FIG_SIZE_HIST = (12, 6)
HIST_BINS = 30
# 设置残差图的 X 轴限制，None 表示自动适应。
# 示例: RESIDUAL_X_LIMITS = (-0.2, 0.2)
RESIDUAL_X_LIMITS = None

# --- 图像保存配置 ---
SAVE_PLOTS = True                  # 是否自动保存图像 (True/False)
SAVE_DIR = 'Ru_Results'           # 图像保存的文件夹名称
DPI = 300                          # 保存图像的分辨率 (DPI)
FILE_FORMAT = 'png'                # 保存图像的文件格式 ('png', 'pdf', 'svg' 等)