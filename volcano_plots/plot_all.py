from pathlib import Path

from plot_volcano import filter_and_plot_data
from plot_32 import plot_three_lines
from plot_potential import plot_oer_potential
from combine_potential import combine_potential
from plot_config import CONFIG


def main():
    data_dir = Path(__file__).resolve().parent
    dat_files = sorted(data_dir.glob("*.dat"))
    if not dat_files:
        print("No .dat files found.")
        return

    for dat_path in dat_files:
        out_dir = data_dir / dat_path.stem
        print(f"\nProcessing {dat_path.name} -> {out_dir}")
        filter_and_plot_data(str(dat_path), output_dir=out_dir)
        plot_three_lines(str(dat_path), output_dir=out_dir)
        plot_oer_potential(str(dat_path), output_dir=out_dir)
        # Combine per-element potential images into a grid
        potential_subdir = out_dir / dat_path.stem
        combine_cols = CONFIG.get("potential", {}).get("combine_columns", 4)
        combine_potential(str(potential_subdir), cols=combine_cols)


if __name__ == "__main__":
    main()
