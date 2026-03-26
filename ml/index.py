#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import GPy
from GPy import kern as gp_kern
import numpy as np
from scipy.stats import norm
from sklearn.metrics import r2_score
import time


OUTPUT_LOG = True
NUM_CANDIDATES = 20


def debug_print(*args):
    if OUTPUT_LOG:
        print(*args)
df = pd.read_csv(
    "data.csv",
    engine="python",
    encoding="utf-8",
)

df = df.set_index("ID")
debug_print(df)

duplicates = df[df.duplicated(keep=False)]
debug_print(duplicates)

df_with_energy = df.dropna(subset=["energy"])
debug_print(df_with_energy)

df_without_energy = df[df["energy"].isnull()].drop(columns=["energy"])
debug_print(df_without_energy)

min_index = df["energy"].idxmin()
min_value = min(df["energy"])
debug_print(min_index, min_value)
X = df_with_energy.iloc[:, :-1]
y = df_with_energy.iloc[:, -1:]

num = X.shape[1]
kernel = gp_kern.RBF(num) * gp_kern.Bias(num) + gp_kern.Linear(num) * gp_kern.Bias(num)
model = GPy.models.GPRegression(X.values, y.values, kernel=kernel, normalizer=True)
model.optimize()


pred_y, _ = model.predict(X.values)
if OUTPUT_LOG:
    fig_vs = plt.figure(figsize=(4, 4))
    plt.scatter(y.values.flatten(), pred_y.flatten())
    plt.plot([min(y.values), max(y.values)], [min(y.values), max(y.values)], ls="--")
    plt.xlabel("calc /eV")
    plt.ylabel("predict /eV")
    plt.show()
correlation_coefficient = np.corrcoef(y.values.flatten(), pred_y.flatten())[0, 1]
r2 = r2_score(y.values, pred_y)

debug_print(f"相関係数: {correlation_coefficient:.3f}")
debug_print(f"決定係数 (R²): {r2:.3f}")
if OUTPUT_LOG:
    for i in range(len(X.columns)):
        vis_dim = [(vis1, vis2) for vis1, vis2 in enumerate(X.loc[min_index, :])]

        debug_print(vis_dim)

        view_variable = i
        vis_dim.pop(view_variable)
        debug_print(X.columns[view_variable])
        fig = plt.figure(figsize=(6, 4))

        ax = fig.add_subplot(111)
        ax.set_xlabel(X.columns[i])
        model.plot(fixed_inputs=vis_dim, plot_density=False, ax=ax)
        plt.show()
        plt.clf()
def acquisition_lcb(mean, std, kappa):
    a = mean - kappa * std
    return a


def acquisition_EI(mean, std, min_value, xi=0.001):
    imp = min_value - mean - xi
    Z = imp / std
    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    return ei


progress_count = 0
progress_level = 0
total_iterations = df.shape[0] - df_with_energy.shape[0]
progress_interval = total_iterations // 50 + 1

debug_print(total_iterations)

start_time = time.time()


def display_progress():
    global progress_count
    global progress_level
    progress_count += 1

    current_level = progress_count // progress_interval

    if not (current_level > progress_level or progress_count == total_iterations):
        return

    progress_level = current_level
    elapsed_time = time.time() - start_time
    debug_print(
        f"Progress: {progress_count}/{total_iterations} ({progress_count/total_iterations*100:.2f}%), Elapsed time: {elapsed_time:.2f}s"
    )

    debug_print(f"progress_count: {progress_count}, progress_level: {progress_level}")


means = []
stds = []
acs_ei = []
acs_lcb = []

itera = df_without_energy.values

debug_print(itera)

for item in itera:
    display_progress()

    mean, val = model.predict(np.array(item).reshape(1, -1))
    std = np.sqrt(val)

    ac_ei = acquisition_EI(mean, std, min_value)
    ac_lcb = acquisition_lcb(mean, std, 7)

    means.append(mean.flatten()[0])
    stds.append(std.flatten()[0])
    acs_ei.append(ac_ei.flatten()[0])
    acs_lcb.append(ac_lcb.flatten()[0])


result_df = pd.DataFrame(df_without_energy)
result_df = result_df.assign(mean=means, std=stds, EI=acs_ei, LCB=acs_lcb)
result_df_sort = result_df.sort_values("LCB", ascending=True)
min_lcb_id = result_df_sort.index[0]

debug_print("ID with the minimum LCB value:", min_lcb_id)


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

result_df_sort = result_df.sort_values("EI", ascending=False)
result_df_sort[:20]


result_df_sort = result_df.sort_values("LCB", ascending=True)
result_df_sort[:20]


result_df_sort = result_df.sort_values("mean", ascending=True)
result_df_sort[:20]


result_df_sort = result_df.sort_values("LCB", ascending=True)
debug_print(result_df_sort.head(NUM_CANDIDATES).index.tolist())

