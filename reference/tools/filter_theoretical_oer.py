#!/usr/bin/env python3
"""Extract pure-computational OER single-atom catalyst papers."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "single_atom_catalyst_references.csv"
OUT_MD = ROOT / "Theoretical_OER_single_atom_papers.md"
OUT_CSV = ROOT / "Theoretical_OER_single_atom_papers.csv"

OER_TERMS = (
    "oxygen evolution",
    "oer",
    "water oxidation",
)


def is_oer(row: dict[str, str]) -> bool:
    text = " ".join(str(row.get(k, "")) for k in row).lower()
    return any(term in text for term in OER_TERMS)


def main() -> int:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    selected = [
        row for row in rows
        if row.get("Category", "").strip().lower() == "theoretical" and is_oer(row)
    ]

    fields = [
        "ID", "Category", "Elements", "Support_or_material",
        "Reaction_or_application", "Title", "Journal", "Year", "Publisher",
        "DOI", "Accessible_URL", "Open_PDF_URL", "PDF_Status",
        "Local_PDF_Path", "Chinese_Introduction", "Classification_Reason",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("# Pure Computational OER Single-Atom Catalyst Papers\n\n")
        fh.write(f"- Source index: `single_atom_catalyst_references.csv`\n")
        fh.write(f"- Selection rule: `Category = Theoretical` and OER/oxygen evolution related keywords\n")
        fh.write(f"- Count: {len(selected)}\n\n")
        for row in selected:
            fh.write(f"## {row['ID']}. {row['Title']}\n\n")
            fh.write(f"- 元素：{row['Elements']}\n")
            fh.write(f"- 载体：{row['Support_or_material']}\n")
            fh.write(f"- 反应：{row['Reaction_or_application']}\n")
            fh.write(f"- 期刊/年份：{row['Journal']}, {row['Year']}\n")
            fh.write(f"- 出版方：{row['Publisher']}\n")
            fh.write(f"- DOI：{row['DOI']}\n")
            fh.write(f"- 可访问位置：{row['Accessible_URL']}\n")
            fh.write(f"- PDF：{row['Local_PDF_Path'] or row['PDF_Status']}\n")
            fh.write(f"- 中文简介：{row['Chinese_Introduction']}\n")
            fh.write(f"- 纯计算依据：{row['Classification_Reason']}\n\n")

    print(f"theoretical_oer={len(selected)}")
    print(OUT_MD)
    print(OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
