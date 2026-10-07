"""Reproduce legacy adsorption-energy fits; --all-data disables legacy selection."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import CONFIG
from plot_potential import filter_allowed


def load_correlation_data(source):
    """Preserve unnamed trailing columns and normalize Unicode minus in headers."""
    lines = [line.strip() for line in Path(source).read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if not lines:
        raise ValueError(f"Empty data file: {source}")
    header = re.split(r"\s+|,", lines[0])
    width = max(len(re.split(r"\s+|,", line)) for line in lines)
    names = header + [f"extra_{i}" for i in range(len(header), width)]
    df = pd.read_csv(source, sep=r"\s+|,", engine="python", header=0,
                     names=names, encoding="utf-8-sig", index_col=False)
    df.columns = [name.strip().replace("\u2212", "-").replace("\u0394", "delta_")
                  for name in df.columns]
    required = {"element", "delta_E1(eV)", "delta_E2(eV)",
                "delta_E3_HOO(eV)", "delta_E_HOO-HO(eV)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source}: missing columns: {sorted(missing)}")
    return df


@plt.rc_context(CONFIG.get("style", {}))
def plot_three_lines(file_path, output_dir=None, all_data=False):
    source = Path(file_path)
    target = Path(output_dir) if output_dir else source.parent / source.stem
    target.mkdir(parents=True, exist_ok=True)
    df = load_correlation_data(source)
    original_count = len(df)
    if not all_data:
        energy_difference = pd.to_numeric(df["delta_E_HOO-HO(eV)"], errors="raise")
        in_range = energy_difference.between(2.0, 4.0)
        print("Excluded by 2-4 eV range:", ", ".join(df.loc[~in_range, "element"]))
        ranged = df.loc[in_range].copy()
        allowed_bases = {name.split("_")[0] for name in CONFIG["allowed_elements"]}
        allowed = ranged["element"].astype(str).str.split("_").str[0].isin(allowed_bases)
        print("Excluded by element list:", ", ".join(ranged.loc[~allowed, "element"]))
        df = filter_allowed(ranged).reset_index(drop=True)
    suffix = "legacy_all_data" if all_data else "legacy_filtered"
    columns = ["delta_E1(eV)", "delta_E2(eV)", "delta_E3_HOO(eV)"]
    energies = df[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(energies.to_numpy()).all():
        raise ValueError("All rows must contain finite energies; no rows will be silently dropped.")
    if len(df) < 2 or energies.iloc[:, 0].nunique() < 2:
        raise ValueError("Linear fitting requires at least two distinct x values.")
    x = energies.iloc[:, 0].to_numpy()
    # The legacy figure used cumulative adsorption energies, not differences to HO.
    ys = [energies.iloc[:, 1].to_numpy(), energies.iloc[:, 2].to_numpy()]
    cfg = CONFIG["plot32"]
    colors = [cfg["colors"]["y1"], cfg["colors"]["y2"]]
    markers = [cfg["markers"]["y1"], cfg["markers"]["y2"]]
    legend_labels = cfg["legend_labels"]
    labels = [legend_labels["y1_data"], legend_labels["y2_data"]]
    fig, ax = plt.subplots(figsize=cfg.get("figsize", (5, 4)))
    scatter_handles, fit_handles, fits = [], [], []
    line_x = np.linspace(x.min(), x.max(), 200)
    for index, (y, color, marker, label) in enumerate(zip(ys, colors, markers, labels), 1):
        scatter_handles.append(ax.scatter(
            x, y, s=cfg.get("scatter_size", 30), color=color, marker=marker,
            alpha=cfg.get("scatter_alpha", 0.7), label=label, zorder=3))
        m, b = np.polyfit(x, y, 1)
        ss_total = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - (m * x + b)) ** 2) / ss_total if ss_total else float("nan")
        fit_label = legend_labels[f"y{index}_fit"].format(y=index, m=m, b=b, r2=r2).replace("+-", "-")
        fit_handles.append(ax.plot(
            line_x, m * line_x + b, color=color,
            linestyle=cfg.get("line_style", "--"), linewidth=cfg.get("line_width", 1.3),
            label=fit_label)[0])
        fits.append({"series": label, "n": len(x), "slope": m, "intercept": b, "r_squared": r2})

    ax.set_xlabel(cfg["xlabel_override"], fontsize=cfg.get("axes_label_fontsize", 11))
    ax.set_ylabel(cfg["ylabel"], fontsize=cfg.get("axes_label_fontsize", 11))
    ax.set_title(cfg["title"], fontsize=cfg.get("title_fontsize", 12))
    ax.tick_params(labelsize=cfg.get("tick_label_fontsize", 10))
    ax.grid(True, **cfg.get("grid", {"linestyle": "--", "linewidth": 0.5}))
    ax.set_axisbelow(True)
    ax.margins(*cfg.get("margins", (0.05, 0.05)))
    if cfg.get("ylim") is not None:
        ax.set_ylim(*cfg["ylim"])
    ax.legend(handles=scatter_handles + fit_handles,
              loc=cfg.get("legend_loc", "lower right"), fontsize=cfg.get("legend_fontsize", 9),
              framealpha=cfg.get("legend_framealpha", 0.8), frameon=cfg.get("legend_frameon", True),
              handlelength=cfg.get("legend_handlelength", 2.0))
    # Reference layout: labels at points, with only the configured manual offsets.
    for y, color in zip(ys, colors):
        for name, xx, yy in zip(df["element"].astype(str), x, y):
            if not ax.get_ylim()[0] <= yy <= ax.get_ylim()[1]:
                continue
            ax.annotate(name, (xx, yy),
                        xytext=cfg.get("label_offsets", {}).get(name, (0, 0)),
                        textcoords="offset points",
                        fontsize=cfg.get("annotation_fontsize", 10), color=color)
    fig.tight_layout()

    stem = target / ("plot_32_all_data" if all_data else "plot_32")
    fig.savefig(stem.with_suffix(".png"), dpi=cfg.get("dpi", 600), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame({"element": df["element"], "delta_E_HO(eV)": x,
                  "delta_E_O(eV)": ys[0], "delta_E_HOO(eV)": ys[1]}).to_csv(
                      target / f"correlation_{suffix}.csv", index=False)
    pd.DataFrame(fits).to_csv(target / f"correlation_fits_{suffix}.csv", index=False)
    print(f"Used {len(df)} of {original_count} rows; {len(df)} points per series. Saved: {stem}.png")
    for fit in fits:
        print(f"n={fit['n']}, slope={fit['slope']:.6f}, intercept={fit['intercept']:.6f}, R2={fit['r_squared']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "oer_deltaE_results.dat")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--all-data", action="store_true", help="Disable range, element, and duplicate filtering")
    args = parser.parse_args()
    plot_three_lines(args.data, args.output_dir, all_data=args.all_data)
