from pathlib import Path

from plot_volcano import filter_and_plot_data


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


if __name__ == "__main__":
    main()
