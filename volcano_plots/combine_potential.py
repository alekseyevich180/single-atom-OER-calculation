import argparse
from math import ceil
from pathlib import Path

from PIL import Image
from plot_config import CONFIG


def combine_potential(folder: str, cols: int | None = None, output_name: str | None = None) -> Path | None:
    """
    Combine per-element potential images in `folder` into a grid with `cols` columns.
    The merged image is saved to the parent directory of `folder`.
    """
    src_dir = Path(folder)
    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}")
        return None

    # Collect images sorted by name (Ag first, etc.)
    img_paths = sorted(
        [p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not img_paths:
        print(f"No images found in {src_dir}")
        return None

    cfg = CONFIG.get("potential", {})
    cols = cols or cfg.get("combine_columns", 4)

    images = [Image.open(p) for p in img_paths]
    widths, heights = zip(*(im.size for im in images))
    max_w, max_h = max(widths), max(heights)

    rows = ceil(len(images) / cols)
    grid_w = cols * max_w
    grid_h = rows * max_h
    canvas = Image.new("RGB", (grid_w, grid_h), color="white")

    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        x = c * max_w
        y = r * max_h
        canvas.paste(im, (x, y))

    parent = src_dir.parent
    output_name = output_name or f"{src_dir.name}_potential_grid.png"
    out_path = parent / output_name
    canvas.save(out_path, dpi=im.info.get("dpi", (cfg.get("dpi", 300), cfg.get("dpi", 300))))
    print(f"Saved grid: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Combine potential images into a grid.")
    parser.add_argument("folder", help="Folder containing per-element images (e.g., volcano/pbe-d2/pbe-d2)")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid")
    parser.add_argument("--output", type=str, default=None, help="Output filename (saved in parent folder)")
    args = parser.parse_args()

    combine_potential(args.folder, cols=args.cols, output_name=args.output)


if __name__ == "__main__":
    main()
