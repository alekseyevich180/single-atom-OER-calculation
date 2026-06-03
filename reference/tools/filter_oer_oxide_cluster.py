#!/usr/bin/env python3
"""Filter OER papers and OER + oxide-cluster-like papers from the reference CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "single_atom_catalyst_references.csv"
OUT_MD = ROOT / "OER_oxide_cluster_candidates.md"
OUT_CSV = ROOT / "OER_oxide_cluster_candidates.csv"

OER_TERMS = ("oxygen evolution", "oer")
CLUSTER_TERMS = (
    "oxide cluster",
    "oxo cluster",
    "metal oxide cluster",
    "polyoxometalate",
    "pom",
    "cluster-based",
    "cluster based",
    "oxide nanocluster",
    "hydroxide cluster",
)
LOOSE_CLUSTER_TERMS = CLUSTER_TERMS + (
    "oxide",
    "oxides",
    "hydroxide",
    "hydroxides",
    "cluster",
    "clusters",
    "nanocluster",
    "nanoclusters",
)


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def main() -> int:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    oer_rows = []
    cluster_rows = []
    loose_rows = []
    for row in rows:
        text = " ".join(str(row.get(k, "")) for k in row)
        if has_any(text, OER_TERMS):
            oer_rows.append(row)
            if has_any(text, CLUSTER_TERMS):
                cluster_rows.append(row)
            if has_any(text, LOOSE_CLUSTER_TERMS):
                loose_rows.append(row)

    fields = [
        "ID", "Category", "Elements", "Support_or_material",
        "Reaction_or_application", "Title", "Journal", "Year", "DOI",
        "Accessible_URL", "Open_PDF_URL", "PDF_Status", "Local_PDF_Path",
        "Chinese_Introduction",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cluster_rows)

    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("# OER + Oxide Cluster Candidate Papers\n\n")
        fh.write(f"- OER papers in current index: {len(oer_rows)}\n")
        fh.write(f"- OER papers with oxide/oxo/POM/cluster keywords: {len(cluster_rows)}\n\n")
        fh.write(f"- OER papers with broad oxide or cluster keywords: {len(loose_rows)}\n\n")

        if cluster_rows:
            fh.write("## Strong Candidates\n\n")
            for row in cluster_rows:
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
        else:
            fh.write("## Strong Candidates\n\n")
            fh.write("当前 1000 篇索引中没有同时满足 OER 与 oxide/oxo/POM/cluster 关键词的强匹配条目。\n\n")

        fh.write("## All OER Papers\n\n")
        for row in oer_rows:
            fh.write(f"- {row['ID']} | {row['Category']} | {row['Elements']} | {row['Title']} | {row['Journal']} {row['Year']} | DOI: {row['DOI']}\n")

        fh.write("\n## Broad OER Oxide/Cluster Keyword Matches\n\n")
        for row in loose_rows:
            fh.write(f"- {row['ID']} | {row['Category']} | {row['Elements']} | {row['Title']} | {row['Journal']} {row['Year']} | DOI: {row['DOI']}\n")

    print(f"oer={len(oer_rows)} cluster_candidates={len(cluster_rows)} loose_candidates={len(loose_rows)}")
    print(OUT_MD)
    print(OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
