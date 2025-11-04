import os
import GPy
import numpy as np
import pandas as pd
from GPy import kern as gp_kern
from scipy.stats import norm


def _load_par(path: str) -> dict:
    cfg = {}
    if not path:
        return cfg
    if not os.path.isfile(path):
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def _get_par_path(all_env: dict) -> str:
    # 1) all_env may provide explicit PAR_FILE
    if all_env and isinstance(all_env, dict):
        path = all_env.get("PAR_FILE")
        if path and os.path.isfile(path):
            return path
    # 2) parent folder of this file: carbon_chain_rot/par
    base = os.path.dirname(os.path.dirname(__file__))
    cand = os.path.join(base, "par")
    if os.path.isfile(cand):
        return cand
    # 3) fallback: carbon_chain_rot/par_all
    cand2 = os.path.join(base, "par_all")
    if os.path.isfile(cand2):
        return cand2
    # 4) not found
    return ""


def _as_bool(val, default=False):
    if val is None:
        return default
    low = str(val).strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    return default


def _as_list(val: str) -> list:
    if val is None:
        return []
    return [x.strip() for x in str(val).split(',') if x.strip()]


class Share:
    def __init__(self, all_env, df_with_energy, df_without_energy):
        self.all_env = all_env
        self.par_path = _get_par_path(all_env)
        self.cfg = _load_par(self.par_path)

        # log path: prefer LOCAL_DIR+WORK_DIR if available; else local log
        try:
            self.log_path = (
                f"{all_env['LOCAL_DIR']}{all_env['WORK_DIR']}/calc/results/bo-py.log"
            )
        except Exception:
            self.log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         "calc", "results", "bo-py.log")

        # Human-controlled knobs via par (with defaults)
        self.num_candidates = int(self.cfg.get("NUM_CANDIDATES", 4))
        # Allow par to override INIT_NUM / BO_NUM; else fall back to all_env
        self.INIT_NUM = int(self.cfg.get("INIT_NUM", all_env.get("INIT_NUM", 0)))
        self.BO_NUM = int(self.cfg.get("BO_NUM", all_env.get("BO_NUM", 0)))

        # Acquisition function controls
        self.ACQ_FUNCTION = self.cfg.get("ACQ_FUNCTION", "EI").upper()  # LCB or EI
        self.KAPPA = float(self.cfg.get("KAPPA", 7))
        self.XI = float(self.cfg.get("XI", 0.001))

        # Data selection controls
        self.TARGET_COLUMN = self.cfg.get("TARGET_COLUMN", "result")
        self.FEATURE_COLUMNS = _as_list(self.cfg.get("FEATURE_COLUMNS"))
        self.FILTER_DONE = self.cfg.get("FILTER_DONE")
        self.FILTER_TODO = self.cfg.get("FILTER_TODO")

        # Copy inputs; apply filters/column selection per par
        self.df_with_energy = df_with_energy.copy()
        self.df_without_energy = df_without_energy.copy()

        # Apply row filters if provided
        if self.FILTER_DONE:
            try:
                self.df_with_energy = self.df_with_energy.query(self.FILTER_DONE)
            except Exception:
                pass
        if self.FILTER_TODO:
            try:
                self.df_without_energy = self.df_without_energy.query(self.FILTER_TODO)
            except Exception:
                pass

        # Decide feature columns
        if not self.FEATURE_COLUMNS:
            # default: all columns minus target in df_with_energy
            self.FEATURE_COLUMNS = [c for c in self.df_with_energy.columns if c != self.TARGET_COLUMN]

        # Ensure df have these columns in order
        self.df_with_energy = self.df_with_energy[self.FEATURE_COLUMNS + [self.TARGET_COLUMN]]
        self.df_without_energy = self.df_without_energy[self.FEATURE_COLUMNS]

    def log_message(self, message):
        with open(f"{self.log_path}", "a") as log_file:
            log_file.write(message + "\n")

    def acquisition_lcb(self, mean, std, kappa):
        """
        计算下置信界（Lower Confidence Bound, LCB）。

        参数：
            - mean: float，预测的均值。
            - std: float，预测的不确定性（标准差）。
            - kappa: float，控制探索与利用平衡的权重参数。

        返回：
            - float，LCB值。
        """
        return mean - kappa * std

    def acquisition_EI(self, mean, std, min_value, xi=0.001):
        """
        计算期望改进（Expected Improvement, EI）。

        参数：
            - mean: float，预测的均值。
            - std: float，预测的不确定性（标准差）。
            - min_value: float，当前最优值。
            - xi: float，探索因子（默认值为0.001）。

        返回：
            - float，EI值。
        """
        imp = min_value - mean - xi
        # guard against zero std
        safe_std = np.where(std <= 1e-12, 1.0, std)
        Z = imp / safe_std
        ei = imp * norm.cdf(Z) + safe_std * norm.pdf(Z)
        # where std is ~0, EI should be ~0 if imp<=0 else small; clamp negatives
        return np.maximum(ei, 0.0)

    def main(self):
        """
        主逻辑：根据当前数据生成新的候选任务。

        如果已完成的任务数未达到初始化任务数 INIT_NUM，则从未完成任务中直接挑选候选。
        如果已完成任务数满足初始化要求，则使用高斯过程回归模型预测，并通过采集函数挑选候选。
        """
        if len(self.df_with_energy) < self.INIT_NUM:
            # 初始化任务不足，从未完成任务中选择
            calc_dir = "/init"

            if len(self.df_with_energy) > self.INIT_NUM - self.num_candidates:
                # 选择缺少的任务数量
                shortage_num = self.INIT_NUM - len(self.df_with_energy)
                head_ids = self.df_without_energy.head(shortage_num).index
                self.log_message("Python 脚本运行完成")
            else:
                head_ids = self.df_without_energy.head(self.num_candidates).index
                self.log_message("Python 脚本运行完成")

            return list(head_ids), calc_dir

        else:
            # 特征数据（X）与目标变量（y）的分离
            X = self.df_with_energy.iloc[:, :-1]
            y = self.df_with_energy.iloc[:, -1:]

            # 配置并训练高斯过程回归模型
            num = X.shape[1]
            kernel = gp_kern.RBF(num) * gp_kern.Bias(num) + gp_kern.Linear(
                num
            ) * gp_kern.Bias(num)
            model = GPy.models.GPRegression(
                X.values, y.values, kernel=kernel, normalizer=True
            )
            model.optimize()

            # 计算采集函数值（目标列可配置）
            min_value = float(self.df_with_energy[self.TARGET_COLUMN].min())
            means = []
            stds = []
            acs_ei = []
            acs_lcb = []

            itera = self.df_without_energy.values
            for item in itera:
                mean, val = model.predict(np.array(item).reshape(1, -1))
                std = np.sqrt(val)

                ac_ei = self.acquisition_EI(mean, std, min_value, self.XI)
                ac_lcb = self.acquisition_lcb(mean, std, self.KAPPA)

                means.append(mean.flatten()[0])
                stds.append(std.flatten()[0])
                acs_ei.append(ac_ei.flatten()[0])
                acs_lcb.append(ac_lcb.flatten()[0])

            # 将结果输出为 Pandas 数据框
            result_df = pd.DataFrame(self.df_without_energy)
            result_df = result_df.assign(mean=means, std=stds, EI=acs_ei, LCB=acs_lcb)

            if (
                len(self.df_with_energy)
                > self.INIT_NUM + self.BO_NUM - self.num_candidates
            ):
                # 计算还需要的任务数量
                shortage_num = self.INIT_NUM + self.BO_NUM - len(self.df_with_energy)
            else:
                shortage_num = self.num_candidates

            # 根据采集函数排序并选择候选
            if self.ACQ_FUNCTION == "EI":
                # EI 越大越好
                result_df_sort = result_df.sort_values("EI", ascending=False)
            else:
                # LCB 越小越好
                result_df_sort = result_df.sort_values("LCB", ascending=True)
            self.log_message(f"排序结果: {result_df_sort.head(shortage_num*2)}")

            candidates = result_df_sort.head(shortage_num).index
            self.log_message("Python 脚本运行完成")
            calc_dir = "/BO"
            return list(candidates), calc_dir


def make_candidates(all_env, df_with_energy, df_without_energy):
    """
    调用 Share 类的实例生成候选任务。

    参数：
        - all_env: dict，包含环境变量（目录、任务数量等）。
        - df_with_energy: DataFrame，已完成任务的数据。
        - df_without_energy: DataFrame，未完成任务的数据。

    返回：
        - candidates: list，候选任务的 ID 列表。
        - calc_dir: str，计算任务的目录路径。
    """
    share_instance = Share(all_env, df_with_energy, df_without_energy)
    return share_instance.main()
