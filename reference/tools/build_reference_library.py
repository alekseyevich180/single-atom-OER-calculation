#!/usr/bin/env python3
"""Build a curated single-atom catalyst reference library.

The script uses OpenAlex metadata because it exposes DOI, abstracts, publisher
metadata, and legal open-access PDF links when they are available.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


OPENALEX = "https://api.openalex.org/works"

QUALITY_PUBLISHERS = (
    "american chemical society",
    "royal society of chemistry",
    "springer nature",
    "nature portfolio",
    "wiley",
    "wiley-vch",
    "national academy of sciences",
    "american association for the advancement of science",
    "science partner journal",
)

QUALITY_SOURCES = (
    "nature",
    "nature catalysis",
    "nature communications",
    "nature chemistry",
    "nature energy",
    "science",
    "science advances",
    "research",
    "proceedings of the national academy of sciences",
    "pnas",
    "acs catalysis",
    "journal of the american chemical society",
    "acs central science",
    "acs nano",
    "nano letters",
    "energy & environmental science",
    "chemical science",
    "journal of materials chemistry a",
    "physical chemistry chemical physics",
    "chemical communications",
    "angewandte chemie",
    "advanced materials",
    "advanced energy materials",
    "small",
    "chemistry - a european journal",
    "chemcatchem",
    "advanced science",
    "chem",
    "joule",
)

ELEMENTS = (
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Mo", "Ru", "Rh", "Pd", "Ag", "W", "Re", "Ir", "Pt", "Au",
    "Ce", "La", "Sn", "Bi", "In", "Ga",
)

QUERY_TERMS = (
    '"single atom catalyst" graphene',
    '"single-atom catalyst" graphene',
    '"single atom catalysts" "nitrogen-doped graphene"',
    '"single-atom catalysts" "N-doped carbon"',
    '"single atom catalyst" "graphene oxide"',
    '"single atom catalyst" "reduced graphene oxide"',
    '"single atom catalyst" "carbon nitride"',
    '"single atom catalyst" "MXene"',
    '"single atom catalyst" "MoS2"',
    '"M-N-C" "single atom catalyst"',
    '"Fe-N-C" "single atom catalyst"',
    '"Ni-N-C" "single atom catalyst"',
    '"Co-N-C" "single atom catalyst"',
    '"single atom catalyst" "CO2 reduction"',
    '"single atom catalyst" "oxygen reduction"',
    '"single atom catalyst" "hydrogen evolution"',
    '"single atom catalyst" "nitrogen reduction"',
)

THEORY_WORDS = (
    "density functional", "dft", "first-principles", "first principles",
    "computational", "screening", "theoretical", "mechanism and kinetics",
    "microkinetic", "quantum mechanics", "ab initio", "free energy",
    "descriptor", "reaction pathway", "reaction mechanism",
)

EXPERIMENT_WORDS = (
    "synthesis", "synthesized", "fabrication", "prepared", "x-ray",
    "xas", "stem", "haadf", "electrochemical", "catalyst exhibits",
    "we report", "operando", "in situ", "activity", "faradaic",
)

SUPPORT_PATTERNS = (
    ("graphene oxide", "graphene oxide"),
    ("reduced graphene oxide", "reduced graphene oxide"),
    ("nitrogen-doped graphene", "N-doped graphene"),
    ("n-doped graphene", "N-doped graphene"),
    ("graphene", "graphene"),
    ("n-doped carbon", "N-doped carbon"),
    ("nitrogen-doped carbon", "N-doped carbon"),
    ("carbon nitride", "carbon nitride"),
    ("g-c3n4", "g-C3N4"),
    ("mxene", "MXene"),
    ("mos2", "MoS2"),
    ("sno2", "SnO2"),
    ("fe2o3", "Fe2O3"),
)


@dataclass
class Paper:
    uid: str = ""
    category: str = ""
    elements: list[str] = field(default_factory=list)
    support: str = ""
    reaction: str = ""
    title: str = ""
    journal: str = ""
    year: str = ""
    publisher: str = ""
    doi: str = ""
    url: str = ""
    oa_pdf_url: str = ""
    pdf_status: str = ""
    pdf_path: str = ""
    chinese_intro: str = ""
    classify_reason: str = ""


def request_json(url: str, retries: int = 3) -> dict:
    headers = {"User-Agent": "single-atom-reference-curator/1.0 (mailto:example@example.com)"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(2 + attempt)
    return {}


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def abstract_from_inverted(index: dict | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, places in index.items():
        for place in places:
            positions.append((int(place), word))
    positions.sort()
    return clean_text(" ".join(word for _, word in positions))


def source_name(work: dict) -> str:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return clean_text(source.get("display_name") or "")


def publisher_name(work: dict) -> str:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return clean_text(source.get("host_organization_name") or "")


def is_quality_source(work: dict) -> bool:
    publisher = publisher_name(work).lower()
    journal = re.sub(r"\s+", " ", source_name(work).lower()).strip()
    quality_journals = {re.sub(r"\s+", " ", s.lower()).strip() for s in QUALITY_SOURCES}
    return any(p in publisher for p in QUALITY_PUBLISHERS) or journal in quality_journals


def is_relevant(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    has_sac = (
        "single atom" in text
        or "single-atom" in text
        or "atomically dispersed" in text
        or "single atomic" in text
        or "m-n-c" in text
    )
    has_catalysis = "catalyst" in text or "catalysis" in text or "electrocatal" in text or "photocatal" in text
    has_support = (
        "graphene" in text
        or "carbon" in text
        or "mxene" in text
        or "mos2" in text
        or "oxide" in text
        or "nitride" in text
    )
    return has_sac and has_catalysis and has_support


def detect_elements_in_text(text: str) -> list[str]:
    text = f" {text} "
    lowered = text.lower()
    found: list[str] = []
    for el in ELEMENTS:
        if el == "In":
            symbol_pattern = r"(?<![A-Za-z])In(?=\d|[-_/])"
        else:
            symbol_pattern = rf"(?<![A-Za-z]){re.escape(el)}(?![a-z])"
        name_pattern = rf"\b{element_name(el)}\b"
        symbol_hit = bool(re.search(symbol_pattern, text))
        name_hit = bool(re.search(name_pattern, lowered))
        if symbol_hit or name_hit:
            found.append(el)
    return found


def detect_elements(title: str, abstract: str) -> list[str]:
    title_found = detect_elements_in_text(title)
    if title_found:
        return title_found
    abstract_found = detect_elements_in_text(abstract)
    return abstract_found or ["Mixed_or_unclear"]


def element_name(symbol: str) -> str:
    names = {
        "Sc": "scandium", "Ti": "titanium", "V": "vanadium", "Cr": "chromium",
        "Mn": "manganese", "Fe": "iron", "Co": "cobalt", "Ni": "nickel",
        "Cu": "copper", "Zn": "zinc", "Mo": "molybdenum", "Ru": "ruthenium",
        "Rh": "rhodium", "Pd": "palladium", "Ag": "silver", "W": "tungsten",
        "Re": "rhenium", "Ir": "iridium", "Pt": "platinum", "Au": "gold",
        "Ce": "cerium", "La": "lanthanum", "Sn": "tin", "Bi": "bismuth",
        "In": "indium", "Ga": "gallium",
    }
    return names.get(symbol, symbol)


def detect_support(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    for needle, label in SUPPORT_PATTERNS:
        if needle in text:
            return label
    return "carbon/other support"


def detect_reaction(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    compact = re.sub(r"\s+", "", text)
    checks = (
        ("co2" in compact or "carbon dioxide" in text, "CO2 reduction/conversion"),
        ("oxygen reduction" in text or "orr" in text, "oxygen reduction reaction"),
        ("oxygen evolution" in text or "oer" in text, "oxygen evolution reaction"),
        ("hydrogen evolution" in text or "her" in text, "hydrogen evolution reaction"),
        ("nitrogen reduction" in text or "nrr" in text or "ammonia" in text, "nitrogen reduction/ammonia synthesis"),
        ("co oxidation" in text, "CO oxidation"),
        ("hydrogenation" in text, "hydrogenation"),
        ("formic acid oxidation" in text, "formic acid oxidation"),
    )
    for ok, label in checks:
        if ok:
            return label
    return "catalysis"


def classify_category(title: str, abstract: str) -> tuple[str, str]:
    text = f"{title} {abstract}".lower()
    title_lower = title.lower()
    theory = sum(1 for word in THEORY_WORDS if word in text)
    exp = sum(1 for word in EXPERIMENT_WORDS if word in text)
    title_theory = any(word in title_lower for word in ("screening", "theoretical", "computational", "dft", "first-principles", "first principles"))
    strong_experiment = (
        "we report" in text
        or "synthesized" in text
        or "synthesis of" in text
        or "prepared" in text
        or "fabricated" in text
        or "x-ray absorption" in text
        or "haadf" in text
        or "stem" in text
        or "experimentally and theoretically" in text
        or "electroreduction" in title_lower
        or "electrocatalyst" in title_lower
    )
    strong_theory = (
        "density functional theory" in text
        or "first-principles" in text
        or "first principles" in text
        or "computational screening" in text
        or "theoretical insights" in title_lower
        or "screening" in title_lower
    )
    misleading_exp_context = "experimental synthesis of" in text or "experimentally synthesized" in text
    if strong_experiment and not title_theory and not misleading_exp_context:
        return "Experimental", "题名/摘要包含合成、制备、表征、测试或实验-理论联合研究信息，按实验论文归类；其中计算部分视作机理支持。"
    if title_theory:
        return "Theoretical", "题名明确包含 computational、DFT、screening、theoretical 或 first-principles 等词，按纯理论计算归类。"
    if strong_theory and (exp <= 2 or misleading_exp_context):
        return "Theoretical", "题名/摘要明确显示 DFT、第一性原理、理论洞察或计算筛选为主，按纯理论计算归类。"
    if theory and exp == 0:
        return "Theoretical", "题名/摘要显示以 DFT、第一性原理或计算筛选为主，未识别到实验合成/表征关键词。"
    if theory >= 2 and exp <= 1 and "we report" not in text:
        return "Theoretical", "计算关键词占主导，按纯理论计算归类。"
    return "Experimental", "识别到合成、表征、电化学/催化测试或 operando/in situ 等实验关键词；若含 DFT，则视作实验论文的机理支持。"


def chinese_intro(title: str, abstract: str, elements: list[str], support: str, reaction: str, category: str) -> str:
    base = f"本文关注 {', '.join(elements)} 单原子位点在 {support} 等载体上的催化行为，主要应用于{reaction}。"
    if category == "Theoretical":
        base += " 文章以理论计算、稳定性分析或反应路径筛选为主，可用于建立结构-活性关系和筛选候选催化剂。"
    else:
        base += " 文章以材料制备、结构表征和催化性能测试为核心，通常结合谱学/显微表征确认单原子分散。"
    abs_short = abstract[:260].strip()
    if abs_short:
        base += " 摘要要点：" + abs_short
    else:
        base += " 摘要未在元数据源中提供，建议通过 DOI 页面查看全文摘要。"
    return base


def safe_name(value: str, max_len: int = 90) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value[:max_len] or "paper"


def fetch_works(target: int, mailto: str = "") -> list[Paper]:
    papers: dict[str, Paper] = {}
    per_page = 200
    for query in QUERY_TERMS:
        cursor = "*"
        while len(papers) < target:
            params = {
                "search": query,
                "filter": "type:article",
                "per-page": str(per_page),
                "cursor": cursor,
                "select": ",".join([
                    "id", "doi", "title", "display_name", "publication_year",
                    "primary_location", "locations", "open_access", "authorships",
                    "abstract_inverted_index",
                ]),
            }
            if mailto:
                params["mailto"] = mailto
            url = OPENALEX + "?" + urllib.parse.urlencode(params)
            data = request_json(url)
            results = data.get("results") or []
            if not results:
                break
            for work in results:
                title = clean_text(work.get("title") or work.get("display_name") or "")
                abstract = abstract_from_inverted(work.get("abstract_inverted_index"))
                doi_url = clean_text(work.get("doi") or "")
                doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "")
                key = doi.lower() or clean_text(work.get("id") or "").lower()
                if not key or key in papers:
                    continue
                if not is_quality_source(work):
                    continue
                if not is_relevant(title, abstract):
                    continue
                primary = work.get("primary_location") or {}
                open_access = work.get("open_access") or {}
                best_oa = open_access.get("oa_url") or ""
                pdf_url = (primary.get("pdf_url") or open_access.get("pdf_url") or "")
                if not pdf_url:
                    best = open_access.get("best_oa_location") or {}
                    pdf_url = best.get("pdf_url") or ""
                if not pdf_url:
                    for loc in work.get("locations") or []:
                        pdf_url = loc.get("pdf_url") or ""
                        if pdf_url:
                            break
                category, reason = classify_category(title, abstract)
                elements = detect_elements(title, abstract)
                support = detect_support(title, abstract)
                reaction = detect_reaction(title, abstract)
                papers[key] = Paper(
                    category=category,
                    elements=elements,
                    support=support,
                    reaction=reaction,
                    title=title,
                    journal=source_name(work),
                    year=str(work.get("publication_year") or ""),
                    publisher=publisher_name(work),
                    doi=doi,
                    url=doi_url or best_oa or clean_text(work.get("id") or ""),
                    oa_pdf_url=pdf_url,
                    pdf_status="pending" if pdf_url else "no_open_pdf_in_metadata",
                    chinese_intro=chinese_intro(title, abstract, elements, support, reaction, category),
                    classify_reason=reason,
                )
                if len(papers) >= target:
                    break
            cursor = ((data.get("meta") or {}).get("next_cursor") or "")
            if not cursor:
                break
            time.sleep(0.15)
        if len(papers) >= target:
            break
    sorted_papers = sorted(papers.values(), key=lambda p: (p.category, p.year, p.title))
    exp_i = 1
    th_i = 1
    for paper in sorted_papers:
        if paper.category == "Experimental":
            paper.uid = f"E{exp_i:04d}"
            exp_i += 1
        else:
            paper.uid = f"T{th_i:04d}"
            th_i += 1
    return sorted_papers


def download_pdf(paper: Paper, root: Path, max_bytes: int) -> None:
    if not paper.oa_pdf_url:
        paper.pdf_status = "no_open_pdf_in_metadata"
        return
    element = paper.elements[0] if paper.elements else "Mixed_or_unclear"
    pdf_dir = root / paper.category / element / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{paper.uid}_{safe_name(paper.title)}.pdf"
    target = pdf_dir / filename
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        "Accept": "application/pdf,text/html;q=0.8,*/*;q=0.5",
    }
    try:
        req = urllib.request.Request(paper.oa_pdf_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            paper.pdf_status = "skipped_too_large"
            return
        if not (data.startswith(b"%PDF") or "pdf" in ctype):
            paper.pdf_status = "open_url_not_pdf"
            return
        target.write_bytes(data)
        paper.pdf_path = str(target.relative_to(root)).replace("\\", "/")
        paper.pdf_status = "downloaded_open_access_pdf"
    except Exception as exc:  # noqa: BLE001
        paper.pdf_status = f"download_failed: {type(exc).__name__}"


def write_outputs(papers: list[Paper], root: Path) -> None:
    fields = [
        "ID", "Category", "Elements", "Support_or_material", "Reaction_or_application",
        "Title", "Journal", "Year", "Publisher", "DOI", "Accessible_URL",
        "Open_PDF_URL", "PDF_Status", "Local_PDF_Path", "Chinese_Introduction",
        "Classification_Reason",
    ]
    csv_path = root / "single_atom_catalyst_references.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for p in papers:
            writer.writerow([
                p.uid, p.category, "; ".join(p.elements), p.support, p.reaction,
                p.title, p.journal, p.year, p.publisher, p.doi, p.url,
                p.oa_pdf_url, p.pdf_status, p.pdf_path, p.chinese_intro,
                p.classify_reason,
            ])

    xls_path = root / "single_atom_catalyst_references.xls"
    with xls_path.open("w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset=\"utf-8\"><style>")
        fh.write("table{border-collapse:collapse;font-family:Arial,'Microsoft YaHei',sans-serif;font-size:10pt}")
        fh.write("th,td{border:1px solid #999;padding:4px 6px;vertical-align:top}")
        fh.write("th{background:#d9eaf7;font-weight:bold}")
        fh.write("</style></head><body><table>\n<tr>")
        for field in fields:
            fh.write(f"<th>{html.escape(field)}</th>")
        fh.write("</tr>\n")
        for p in papers:
            row = [
                p.uid, p.category, "; ".join(p.elements), p.support, p.reaction,
                p.title, p.journal, p.year, p.publisher, p.doi, p.url,
                p.oa_pdf_url, p.pdf_status, p.pdf_path, p.chinese_intro,
                p.classify_reason,
            ]
            fh.write("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>\n")
        fh.write("</table></body></html>\n")

    for old_md in list((root / "Experimental").glob("*/references.md")) + list((root / "Theoretical").glob("*/references.md")):
        old_md.unlink()

    by_folder: dict[tuple[str, str], list[Paper]] = {}
    for p in papers:
        for el in p.elements:
            by_folder.setdefault((p.category, el), []).append(p)
    for (category, element), items in by_folder.items():
        folder = root / category / element
        folder.mkdir(parents=True, exist_ok=True)
        md = folder / "references.md"
        with md.open("w", encoding="utf-8") as fh:
            fh.write(f"# {category} {element} Single-Atom Catalyst References\n\n")
            for p in items:
                fh.write(f"## {p.uid}. {p.title}\n\n")
                fh.write(f"- 期刊/年份：{p.journal}, {p.year}\n")
                fh.write(f"- DOI：{p.doi or 'N/A'}\n")
                fh.write(f"- 可访问位置：{p.url}\n")
                fh.write(f"- 开放 PDF：{p.oa_pdf_url or '元数据中未发现开放 PDF'}\n")
                fh.write(f"- 本地 PDF：{p.pdf_path or p.pdf_status}\n")
                fh.write(f"- 中文简介：{p.chinese_intro}\n")
                fh.write(f"- 分类依据：{p.classify_reason}\n\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reference", help="reference directory")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--clean-pdfs", action="store_true", help="delete previously generated PDFs before downloading")
    parser.add_argument("--max-pdf-mb", type=int, default=40)
    parser.add_argument("--mailto", default="", help="optional email for polite OpenAlex requests")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    papers = fetch_works(args.target, args.mailto)
    if args.download_pdfs:
        if args.clean_pdfs:
            for pdf in list((root / "Experimental").glob("*/pdfs/*.pdf")) + list((root / "Theoretical").glob("*/pdfs/*.pdf")):
                pdf.unlink()
        max_bytes = args.max_pdf_mb * 1024 * 1024
        for idx, paper in enumerate(papers, 1):
            download_pdf(paper, root, max_bytes)
            if idx % 25 == 0:
                time.sleep(1)
    write_outputs(papers, root)
    downloaded = sum(1 for p in papers if p.pdf_status == "downloaded_open_access_pdf")
    print(f"papers={len(papers)} downloaded_open_access_pdfs={downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
