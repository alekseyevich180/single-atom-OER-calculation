import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).with_name("Sn.dat")
CHUNK = 121  # size of each block in the data


def smooth_block(block):
    """Apply Savitzky-Golay and spline smoothing to one block."""
    x = block["c2"].to_numpy()
    y = block["c4"].to_numpy()

    # Pick an odd window <= block length, capped at 21
    max_window = min(len(block), 21)
    window = max_window if max_window % 2 == 1 else max_window - 1
    if window < 5:
        window = len(block) if len(block) % 2 == 1 else len(block) - 1
        window = max(window, 3)

    polyorder = 3 if window > 5 else 2
    y_sg = savgol_filter(y, window_length=window, polyorder=polyorder)

    # Smoothing factor scales with block size; increase for stronger smoothing
    spline = UnivariateSpline(x, y, s=len(block) * 0.1)
    y_sp = spline(x)

    block = block.copy()
    block["c4_savgol"] = y_sg
    block["c4_spline"] = y_sp
    return block


def main():
    df = pd.read_csv(DATA_PATH, header=None, names=["c1", "c2", "c3", "c4"])

    blocks = []
    for start in range(0, len(df), CHUNK):
        blocks.append(smooth_block(df.iloc[start:start + CHUNK]))
    out = pd.concat(blocks, ignore_index=True)

    out_path = Path(__file__).with_name("Sn_smoothed.csv")
    plot_path = Path(__file__).with_name("Sn_smoothed.png")

    out.to_csv(out_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df["c2"], df["c4"], ".", alpha=0.3, label="raw")
    plt.plot(out["c2"], out["c4_savgol"], "-", label="savgol")
    plt.plot(out["c2"], out["c4_spline"], "-", label="spline")
    plt.xlabel("c2")
    plt.ylabel("c4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {out_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
