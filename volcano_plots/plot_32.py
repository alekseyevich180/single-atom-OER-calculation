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
    labels = [
        r"$\Delta E_{\mathrm{O*}}$ (eV)",
        r"$\Delta E_{\mathrm{HOO*}}$ (eV)",
    ]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    scatter_handles, fit_handles, fits = [], [], []
    line_x = np.linspace(x.min(), x.max(), 200)
    for y, color, marker, label in zip(ys, colors, markers, labels):
        scatter_handles.append(ax.scatter(x, y, s=55, color=color, marker=marker,
                                          alpha=0.7, label=label, zorder=3))
        m, b = np.polyfit(x, y, 1)
        ss_total = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - (m * x + b)) ** 2) / ss_total if ss_total else float("nan")
        fit_label = rf"$y={m:.3f}x{b:+.3f},\ R^2={r2:.3f}$"
        fit_handles.append(ax.plot(line_x, m * line_x + b, color=color,
                                   linestyle="--", linewidth=2, label=fit_label)[0])
        fits.append({"series": label, "n": len(x), "slope": m, "intercept": b, "r_squared": r2})

    ax.set_xlabel(cfg["xlabel_override"], fontsize=17)
    ax.set_ylabel(cfg["ylabel"], fontsize=17)
    ax.set_title(cfg["title"], fontsize=16, pad=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", linewidth=0.8, color="0.7")
    ax.set_axisbelow(True)
    ax.margins(x=0.09, y=0.12)
    if cfg.get("ylim") is not None:
        ax.set_ylim(*cfg["ylim"])
    legend = ax.legend(handles=scatter_handles + fit_handles, loc="lower right",
                       fontsize=12, framealpha=0.92)
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [legend.get_window_extent(renderer).expanded(1.02, 1.04)]
    points = [ax.transData.transform((xx, yy)) for y in ys for xx, yy in zip(x, y)]
    from matplotlib.transforms import Bbox
    point_boxes = [Bbox.from_bounds(px - 5, py - 5, 10, 10) for px, py in points]
    axes_box = ax.get_window_extent(renderer)
    candidates = [(4, 5), (4, -12), (-5, 5), (-5, -12), (0, 13), (0, -20)]
    candidates += [(dx, dy) for dy in (22, -28, 32, -38, 42, -48)
                   for dx in (0, 15, -15, 30, -30)]
    for y, color in zip(ys, colors):
        for name, xx, yy in zip(df["element"].astype(str), x, y):
            if not ax.get_ylim()[0] <= yy <= ax.get_ylim()[1]:
                continue
            annotation = ax.annotate(name, (xx, yy), xytext=(4, 5),
                                     textcoords="offset points", fontsize=10, color=color,
                                     annotation_clip=False)
            best = None
            for dx, dy in candidates:
                annotation.set_position((dx, dy))
                annotation.set_ha("right" if dx < 0 else "left" if dx > 0 else "center")
                box = annotation.get_window_extent(renderer).expanded(1.1, 1.12)
                overlaps = sum(box.overlaps(other) for other in occupied)
                hits = sum(box.overlaps(other) for other in point_boxes)
                outside = not (axes_box.contains(box.x0, box.y0) and axes_box.contains(box.x1, box.y1))
                score = 10000 * outside + 1000 * overlaps + 100 * hits + abs(dx) + abs(dy)
                if best is None or score < best[0]:
                    best = (score, dx, dy, annotation.get_ha(), box)
            _, dx, dy, ha, box = best
            annotation.set_position((dx, dy))
            annotation.set_ha(ha)
            occupied.append(box)
            if abs(dx) + abs(dy) > 25:
                ax.annotate("", (xx, yy), xytext=(dx, dy + 3), textcoords="offset points",
                            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5, "alpha": 0.6})

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
