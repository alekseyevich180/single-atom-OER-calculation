#!/usr/bin/env python3
"""Filter OER papers from direct graphene/graphdiyne SAC lists."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUTS = [
    (
        ROOT / "Direct_graphene_graphdiyne_single_atom_papers.csv",
        ROOT / "Direct_graphene_graphdiyne_OER_papers.csv",
        ROOT / "Direct_graphene_graphdiyne_OER_papers.md",
        "Direct graphene/graphdiyne OER papers excluding explicit N4/O4 motifs",
    ),
    (
        ROOT / "Direct_pristine_graphene_graphdiyne_single_atom_papers.csv",
        ROOT / "Direct_pristine_graphene_graphdiyne_OER_papers.csv",
        ROOT / "Direct_pristine_graphene_graphdiyne_OER_papers.md",
        "Direct pristine graphene/graphdiyne OER papers excluding N-doped/N4/O4 motifs",
    ),
]

OER_TERMS = (
    "oxygen evolution",
    "oer",
    "water oxidation",
)


def is_oer(row: dict[str, str]) -> bool:
    text = " ".join(str(row.get(k, "")) for k in row).lower()
    return any(term in text for term in OER_TERMS)


def write_outputs(input_csv: Path, output_csv: Path, output_md: Path, title: str) -> int:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    selected = [row for row in rows if is_oer(row)]
    fields = rows[0].keys() if rows else []

    with output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    with output_md.open("w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(f"- Source: `{input_csv.name}`\n")
        fh.write("- Rule: contains oxygen evolution / OER / water oxidation\n")
        fh.write(f"- Count: {len(selected)}\n\n")
        for row in selected:
            fh.write(f"## {row['ID']}. {row['Title']}\n\n")
            fh.write(f"- 类型：{row['Category']}\n")
            fh.write(f"- 元素：{row['Elements']}\n")
            fh.write(f"- 载体：{row['Support_or_material']}\n")
            fh.write(f"- 反应：{row['Reaction_or_application']}\n")
            fh.write(f"- 期刊/年份：{row['Journal']}, {row['Year']}\n")
            fh.write(f"- DOI：{row['DOI']}\n")
            fh.write(f"- 可访问位置：{row['Accessible_URL']}\n")
            fh.write(f"- PDF：{row['Local_PDF_Path'] or row['PDF_Status']}\n")
            fh.write(f"- 中文简介：{row['Chinese_Introduction']}\n\n")
    return len(selected)


def main() -> int:
    for input_csv, output_csv, output_md, title in INPUTS:
        count = write_outputs(input_csv, output_csv, output_md, title)
        print(f"{output_md.name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
