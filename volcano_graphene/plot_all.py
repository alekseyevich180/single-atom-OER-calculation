from pathlib import Path

from plot_volcano import filter_and_plot_data
from plot_potential import plot_oer_potential
from combine_potential import combine_potential
from plot_config import CONFIG

try:
    from plot_32 import plot_three_lines
except ImportError:
    plot_three_lines = None


def main():
    data_dir = Path(__file__).resolve().parent
    data_files = sorted(data_dir.glob("*.dat"))
    if not data_files:
        data_files = sorted(data_dir.glob("*.csv"))
    if not data_files:
        print("No .dat or .csv files found.")
        return

    for data_path in data_files:
        out_dir = data_dir / data_path.stem
        print(f"\nProcessing {data_path.name} -> {out_dir}")
        filter_and_plot_data(str(data_path), output_dir=out_dir)
        if plot_three_lines is not None:
            plot_three_lines(str(data_path), output_dir=out_dir)
        else:
            print("plot_32.py not found; skipping plot_three_lines.")
        plot_oer_potential(str(data_path), output_dir=out_dir)
        # Combine per-element potential images into a grid
        potential_subdir = out_dir / data_path.stem
        combine_cols = CONFIG.get("potential", {}).get("combine_columns", 4)
        combine_potential(str(potential_subdir), cols=combine_cols)


if __name__ == "__main__":
    main()
