import GPy
import numpy as np
import pandas as pd
from GPy import kern as gp_kern
from scipy.stats import norm


class Share:
    def __init__(self, all_env, df_with_energy, df_without_energy):
        self.all_env = all_env
        self.log_path = (
            f"{all_env['LOCAL_DIR']}{all_env['WORK_DIR']}/calc/results/bo-py.log"
        )
        self.num_candidates = 4
        self.INIT_NUM = int(all_env["INIT_NUM"])
        self.BO_NUM = int(all_env["BO_NUM"])
        self.df_with_energy = df_with_energy
        self.df_without_energy = df_without_energy

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
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        return ei

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

            # 计算采集函数值
            min_value = min(self.df_with_energy["result"])
            means = []
            stds = []
            acs_ei = []
            acs_lcb = []

            itera = self.df_without_energy.values
            for item in itera:
                mean, val = model.predict(np.array(item).reshape(1, -1))
                std = np.sqrt(val)

                ac_ei = self.acquisition_EI(mean, std, min_value)
                ac_lcb = self.acquisition_lcb(mean, std, 7)

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

            # 根据 LCB 值排序并选择候选
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
