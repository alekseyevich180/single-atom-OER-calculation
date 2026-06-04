#!/usr/bin/env python3
"""Filter direct graphene/graphdiyne-supported SAC papers excluding N4/O4 motifs."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "single_atom_catalyst_references.csv"
OUT_MD = ROOT / "Direct_graphene_graphdiyne_single_atom_papers.md"
OUT_CSV = ROOT / "Direct_graphene_graphdiyne_single_atom_papers.csv"
STRICT_OUT_MD = ROOT / "Direct_pristine_graphene_graphdiyne_single_atom_papers.md"
STRICT_OUT_CSV = ROOT / "Direct_pristine_graphene_graphdiyne_single_atom_papers.csv"

SUPPORT_TERMS = (
    "graphene",
    "graphdiyne",
    "graphyne",
    "graphene nanoribbon",
    "graphene-supported",
    "graphene supported",
)

EXCLUDE_PATTERNS = (
    r"\bN\s*4\b",
    r"\bO\s*4\b",
    r"\bM\s*-\s*N\s*4\b",
    r"\bM\s*-\s*O\s*4\b",
    r"\bTM\s*-\s*N\s*4\b",
    r"\bTM\s*-\s*O\s*4\b",
    r"\bMN\s*4\b",
    r"\bMO\s*4\b",
    r"\bNi\s*-\s*O\s*4\b",
    r"\bFe\s*-\s*N\s*4\b",
    r"\bCo\s*-\s*N\s*4\b",
    r"\bNi\s*-\s*N\s*4\b",
    r"\bN4\b",
    r"\bO4\b",
)
STRICT_EXCLUDE_TERMS = (
    "n-doped",
    "nitrogen-doped",
    "n doped",
    "nitrogen doped",
    "carbon nitride",
    "c3n4",
    "g-c3n4",
    "c 3 n 4",
    "n-rich",
    "nitrogen-rich",
    "m-n-c",
    "fe-n-c",
    "co-n-c",
    "ni-n-c",
)


def has_support(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in SUPPORT_TERMS)


def has_excluded_motif(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in EXCLUDE_PATTERNS)


def has_strict_excluded_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in STRICT_EXCLUDE_TERMS)


def main() -> int:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    selected = []
    strict_selected = []
    excluded_support_hits = []
    for row in rows:
        text = " ".join(str(row.get(k, "")) for k in row)
        if not has_support(text):
            continue
        if has_excluded_motif(text):
            excluded_support_hits.append(row)
            continue
        selected.append(row)
        if not has_strict_excluded_term(text):
            strict_selected.append(row)

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
        fh.write("# Direct Graphene/Graphdiyne-Supported Single-Atom Papers Excluding N4/O4 Motifs\n\n")
        fh.write(f"- Source index: `single_atom_catalyst_references.csv`\n")
        fh.write("- Rule: graphene/graphdiyne/graphyne support terms; exclude explicit N4/O4/M-N4/M-O4 motifs\n")
        fh.write(f"- Selected count: {len(selected)}\n")
        fh.write(f"- Graphene/graphdiyne hits excluded by N4/O4 rule: {len(excluded_support_hits)}\n\n")

        fh.write("## Selected Papers\n\n")
        for row in selected:
            fh.write(f"### {row['ID']}. {row['Title']}\n\n")
            fh.write(f"- 类型：{row['Category']}\n")
            fh.write(f"- 元素：{row['Elements']}\n")
            fh.write(f"- 载体：{row['Support_or_material']}\n")
            fh.write(f"- 反应：{row['Reaction_or_application']}\n")
            fh.write(f"- 期刊/年份：{row['Journal']}, {row['Year']}\n")
            fh.write(f"- DOI：{row['DOI']}\n")
            fh.write(f"- 可访问位置：{row['Accessible_URL']}\n")
            fh.write(f"- PDF：{row['Local_PDF_Path'] or row['PDF_Status']}\n")
            fh.write(f"- 中文简介：{row['Chinese_Introduction']}\n\n")

    with STRICT_OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(strict_selected)

    with STRICT_OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("# Direct Pristine Graphene/Graphdiyne-Supported Single-Atom Papers Excluding N-Doped/N4/O4 Motifs\n\n")
        fh.write(f"- Source index: `single_atom_catalyst_references.csv`\n")
        fh.write("- Rule: graphene/graphdiyne/graphyne support terms; exclude explicit N4/O4 motifs and N-doped/carbon-nitride supports\n")
        fh.write(f"- Selected count: {len(strict_selected)}\n\n")

        fh.write("## Selected Papers\n\n")
        for row in strict_selected:
            fh.write(f"### {row['ID']}. {row['Title']}\n\n")
            fh.write(f"- 类型：{row['Category']}\n")
            fh.write(f"- 元素：{row['Elements']}\n")
            fh.write(f"- 载体：{row['Support_or_material']}\n")
            fh.write(f"- 反应：{row['Reaction_or_application']}\n")
            fh.write(f"- 期刊/年份：{row['Journal']}, {row['Year']}\n")
            fh.write(f"- DOI：{row['DOI']}\n")
            fh.write(f"- 可访问位置：{row['Accessible_URL']}\n")
            fh.write(f"- PDF：{row['Local_PDF_Path'] or row['PDF_Status']}\n")
            fh.write(f"- 中文简介：{row['Chinese_Introduction']}\n\n")

    print(f"selected={len(selected)} strict_selected={len(strict_selected)} excluded_n4_o4={len(excluded_support_hits)}")
    print(OUT_MD)
    print(OUT_CSV)
    print(STRICT_OUT_MD)
    print(STRICT_OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
