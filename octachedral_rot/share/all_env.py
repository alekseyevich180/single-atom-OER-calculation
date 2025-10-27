import pandas as pd

# 环境变量
all_env = {
    "LOCAL_DIR": "local-study-directory",
    "REMOTE_DIR": "remote-study-directory",
    "WORK_DIR": "heo-work-directory",
    "JOB_USER": "job-name-prefix",
    "INIT_NUM": 100,
    "BO_NUM": 200,
}

# 已完成任务的数据
df_with_energy = pd.DataFrame(
    [[1.0, 2.0, -0.5], [2.0, 3.0, -1.0], [3.0, 4.0, -1.5]],
    columns=["feature1", "feature2", "result"]
)

# 未完成任务的数据
df_without_energy = pd.DataFrame(
    [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]],
    columns=["feature1", "feature2"]
)

# 调用生成候选任务
from make_candidates import make_candidates

candidates, calc_dir = make_candidates(all_env, df_with_energy, df_without_energy)
print(f"候选任务 ID: {candidates}")
print(f"计算目录: {calc_dir}")
