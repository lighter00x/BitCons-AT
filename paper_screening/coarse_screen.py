#!/usr/bin/env python3
"""Coarse, evidence-preserving screening for BitCons/BitPlane literature."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GROUPS = {
    "adversarial_core": {
        "weight": 5,
        "patterns": [
            r"\badversarial (?:training|robustness|attack(?:s)?|defen[cs]e(?:s)?|example(?:s)?|perturbation(?:s)?)\b",
            r"\brobust(?:ness)? (?:against|to) adversarial\b",
            r"\brobust accuracy\b",
            r"\bPGD[- ]?AT\b", r"\bTRADES\b", r"\bMART\b", r"\bAutoAttack\b",
        ],
    },
    "input_bit_quantization": {
        "weight": 8,
        "patterns": [
            r"\bbit[- ]?plane(?:s)?\b", r"\bbit[- ]?depth\b", r"\blow[- ]?order bits?\b",
            r"\bpixel bits?\b", r"\binput quanti[sz]ation\b", r"\bimage quanti[sz]ation\b",
            r"\binput discreti[sz]ation\b", r"\bdiscrete input(?:s)?\b",
            r"\bfeature squeezing\b", r"\bquantized input(?:s)?\b",
        ],
    },
    "input_transform_compression": {
        "weight": 6,
        "patterns": [
            r"\bimage preprocessing\b", r"\bpreprocessing images\b",
            r"\binput transformation(?:s)?\b", r"\brandomized transformation(?:s)?\b",
            r"\blossy compression\b", r"\bcompression[- ]based adversarial\b",
            r"\badversarial purification\b", r"\bfeature compression\b",
            r"\bcompressibility and adversarial robustness\b",
            r"\bcompressing image space\b",
            r"\bneural image compression and robustness\b",
            r"\bpreprocessing defen[cs]e(?:s)?\b",
        ],
    },
    "adaptive_evaluation": {
        "weight": 8,
        "patterns": [
            r"\bBPDA\b", r"backward pass differentiable approximation",
            r"\badaptive attack(?:s)?\b", r"\bobfuscated gradients?\b",
            r"\bgradient masking\b", r"\bgradient obfuscation\b",
            r"\bexpectation over transformation(?:s)?\b", r"\bEOT\b",
            r"\bnon[- ]?differentiable defen[cs]e(?:s)?\b",
        ],
    },
    "discrete_worst_case": {
        "weight": 6,
        "patterns": [
            r"\bdiscrete inner maximization\b", r"\bmixed discrete[- ]continuous\b",
            r"\bworst[- ]case (?:input )?transformation(?:s)?\b",
            r"\bdiscrete adversarial training\b", r"\badversarial bit(?:s| flip(?:s)?)?\b",
            r"\bmaximum[- ]loss candidate(?:s)?\b", r"\bhard example mining\b",
        ],
    },
    "consistency_alignment": {
        "weight": 3,
        "patterns": [
            r"\bprediction consistency\b", r"\blogit alignment\b",
            r"\brepresentation alignment\b", r"\binvariance regulari[sz]ation\b",
            r"\badversarial contrastive (?:learning|training)\b",
            r"\bcontrastive adversarial training\b", r"\brobust representation learning\b",
        ],
    },
    "evaluation_protocol": {
        "weight": 3,
        "patterns": [
            r"\bAutoAttack\b", r"\bSquare Attack\b", r"\bRobustBench\b",
            r"\bmultiple restarts?\b", r"\bthreat model\b",
            r"\brobustness evaluation\b", r"\bevaluat(?:e|ing|ion) adversarial robustness\b",
        ],
    },
    "vision_context": {
        "weight": 1,
        "patterns": [
            r"\bimage(?:s)?\b", r"\bvision\b", r"\bvisual\b", r"\bpixel(?:s)?\b",
            r"\bCIFAR[- ]?(?:10|100)\b", r"\bImageNet\b", r"\bSVHN\b",
            r"\bimage classification\b",
        ],
    },
    "classification_context": {
        "weight": 1,
        "patterns": [
            r"\bimage classifi(?:er|ers|cation)\b",
            r"\bCNNs?\b", r"\bResNet(?:s)?\b", r"\bWideResNet(?:s)?\b",
            r"\bCIFAR[- ]?(?:10|100)\b", r"\bImageNet\b", r"\brobust accuracy\b",
        ],
    },
}

PENALTIES = {
    "weight_quantization_only": {
        "weight": -5,
        "patterns": [
            r"\bweight quanti[sz]ation\b", r"\bquantized weights?\b",
            r"\bmodel quanti[sz]ation\b", r"\bpost[- ]training quanti[sz]ation\b",
            r"\bquantization[- ]aware training\b",
        ],
    },
    "non_vision_domain": {
        "weight": -4,
        "patterns": [
            r"\blarge language models?\b", r"\bLLMs?\b", r"\bgraph neural networks?\b",
            r"\breinforcement learning\b", r"\bspeech recognition\b", r"\btime series\b",
        ],
    },
}

COMPILED_GROUPS = {
    name: [re.compile(pattern, re.IGNORECASE) for pattern in spec["patterns"]]
    for name, spec in GROUPS.items()
}
COMPILED_PENALTIES = {
    name: [re.compile(pattern, re.IGNORECASE) for pattern in spec["patterns"]]
    for name, spec in PENALTIES.items()
}


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def infer_venue_year(path: Path) -> tuple[str, str]:
    match = re.match(r"([a-z]+)(\d{4})_papers$", path.stem.lower())
    return (match.group(1).upper(), match.group(2)) if match else (path.stem, "")


def evidence_for(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    evidence = []
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 55)
            end = min(len(text), match.end() + 55)
            snippet = text[start:end].strip()
            if snippet not in evidence:
                evidence.append(snippet)
    return evidence[:4]


def score_paper(record: dict[str, Any], source: Path, index: int) -> dict[str, Any]:
    title = clean_text(record.get("Title") or record.get("title"))
    abstract = clean_text(record.get("Abstract") or record.get("abstract"))
    keywords_value = record.get("Keywords") or record.get("keywords") or []
    keywords = [clean_text(item) for item in keywords_value] if isinstance(keywords_value, list) else [clean_text(keywords_value)]
    text = " ".join([title, abstract, " ".join(keywords)])
    venue, year = infer_venue_year(source)

    matches: dict[str, list[str]] = {}
    score = 0
    for name, patterns in COMPILED_GROUPS.items():
        evidence = evidence_for(patterns, text)
        if evidence:
            matches[name] = evidence
            score += int(GROUPS[name]["weight"]) + min(2, len(evidence) - 1)

    penalties: dict[str, list[str]] = {}
    for name, patterns in COMPILED_PENALTIES.items():
        evidence = evidence_for(patterns, text)
        if not evidence:
            continue
        if name == "weight_quantization_only" and "input_bit_quantization" in matches:
            continue
        if name == "non_vision_domain" and "vision_context" in matches:
            continue
        penalties[name] = evidence
        score += int(PENALTIES[name]["weight"])

    has_adv = "adversarial_core" in matches
    target_groups = {
        "input_bit_quantization", "input_transform_compression",
        "adaptive_evaluation", "discrete_worst_case",
    }
    has_direct_target = bool(target_groups.intersection(matches))
    has_auxiliary_target = "consistency_alignment" in matches
    has_core_scope = bool(
        {"classification_context", "input_bit_quantization", "input_transform_compression"}
        .intersection(matches)
    )

    if has_adv and has_direct_target and has_core_scope and score >= 12:
        tier = "A-Core"
    elif has_adv and (has_direct_target or has_auxiliary_target or "evaluation_protocol" in matches) and score >= 8:
        tier = "B-Method"
    elif has_adv and score >= 5:
        tier = "C-Foundation"
    else:
        tier = "D-Exclude"

    selected = tier != "D-Exclude"
    return {
        "paper_id": f"{source.stem}:{index}",
        "venue": venue,
        "year": year,
        "source_file": source.name,
        "title": title,
        "abstract": abstract,
        "authors": record.get("Authors") or record.get("authors") or [],
        "keywords": keywords,
        "pdf": clean_text(record.get("PDF") or record.get("pdf") or record.get("pdf_url")),
        "url": clean_text(record.get("URL") or record.get("url")),
        "score": score,
        "tier": tier,
        "selected": selected,
        "matched_groups": sorted(matches),
        "evidence": matches,
        "penalties": penalties,
        "has_abstract": bool(abstract),
        "normalized_title": normalized_title(title),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "paper_id", "venue", "year", "title", "score", "tier", "matched_groups",
        "abstract", "evidence", "penalties", "pdf", "url", "has_abstract", "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field, "") for field in fields}
            output["matched_groups"] = ";".join(row["matched_groups"])
            output["evidence"] = json.dumps(row["evidence"], ensure_ascii=False)
            output["penalties"] = json.dumps(row["penalties"], ensure_ascii=False)
            writer.writerow(output)


def write_report(path: Path, stats: dict[str, Any], combined: list[dict[str, Any]]) -> None:
    lines = [
        "# Coarse Screening Report",
        "",
        f"Rules version: `{stats['rules_version']}`",
        "",
        "## Source Summary",
        "",
        "| Source | Total | Selected | A-Core | B-Method | C-Foundation | Missing abstract |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, source_stats in stats["sources"].items():
        tiers = source_stats["tiers"]
        lines.append(
            f"| {source} | {source_stats['total']} | {source_stats['selected']} | "
            f"{tiers.get('A-Core', 0)} | {tiers.get('B-Method', 0)} | "
            f"{tiers.get('C-Foundation', 0)} | {source_stats['missing_abstract']} |"
        )
    dedup = stats["deduplication"]
    lines.extend([
        "",
        "## Combined Results",
        "",
        f"- Selected before title deduplication: {dedup['selected_before_deduplication']}",
        f"- Selected after title deduplication: {dedup['selected_after_deduplication']}",
        f"- Duplicate records removed: {dedup['duplicate_records_removed']}",
        f"- Tier counts: {json.dumps(dedup['tiers'], ensure_ascii=False)}",
        "",
        "## A-Core Candidates",
        "",
        "| Score | Venue | Year | Title | Matched groups |",
        "|---:|---|---:|---|---|",
    ])
    for row in combined:
        if row["tier"] != "A-Core":
            continue
        safe_title = row["title"].replace("|", "\\|")
        lines.append(
            f"| {row['score']} | {row['venue']} | {row['year']} | {safe_title} | "
            f"{', '.join(row['matched_groups'])} |"
        )
    lines.extend([
        "",
        "## Interpretation Limits",
        "",
        "- This is a high-recall lexical screen, not a final relevance judgment.",
        "- Venue JSON files do not contain acceptance decisions; counts must not be treated as accepted-paper counts.",
        "- Missing CVPR abstracts can cause false negatives.",
        "- A-Core papers still require abstract/PDF review to separate image classification from adjacent generative, 3D, medical, and multimodal tasks.",
        "- The 2024-2026 corpus does not replace seminal-paper and citation-chain retrieval.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "rules_version": 2,
        "sources": {},
        "deduplication": {},
    }

    for source in sorted(args.input_dir.glob("*_papers.json")):
        with source.open(encoding="utf-8") as handle:
            records = json.load(handle)
        rows = [score_paper(record, source, index) for index, record in enumerate(records)]
        selected = sorted(
            (row for row in rows if row["selected"]),
            key=lambda row: (-row["score"], row["title"].lower()),
        )
        stem = source.stem.removesuffix("_papers")
        write_jsonl(args.output_dir / f"{stem}_coarse_selected.jsonl", selected)
        write_csv(args.output_dir / f"{stem}_coarse_selected.csv", selected)
        stats["sources"][source.name] = {
            "total": len(rows),
            "selected": len(selected),
            "missing_abstract": sum(not row["has_abstract"] for row in rows),
            "tiers": dict(Counter(row["tier"] for row in rows)),
            "selected_groups": dict(Counter(group for row in selected for group in row["matched_groups"])),
        }
        all_rows.extend(selected)

    deduplicated: dict[str, dict[str, Any]] = {}
    duplicate_sources: defaultdict[str, list[str]] = defaultdict(list)
    for row in sorted(all_rows, key=lambda item: -item["score"]):
        key = row["normalized_title"] or row["paper_id"]
        duplicate_sources[key].append(row["source_file"])
        deduplicated.setdefault(key, row)
    combined = list(deduplicated.values())
    combined.sort(key=lambda row: (-row["score"], row["title"].lower()))
    for row in combined:
        row["also_in_sources"] = sorted(set(duplicate_sources[row["normalized_title"]]))

    write_jsonl(args.output_dir / "all_coarse_selected_deduplicated.jsonl", combined)
    write_json(args.output_dir / "candidate_papers.json", combined)
    write_csv(args.output_dir / "all_coarse_selected_deduplicated.csv", combined)
    for tier, filename in (
        ("A-Core", "a_core_candidates"),
        ("B-Method", "b_method_candidates"),
        ("C-Foundation", "c_foundation_candidates"),
    ):
        tier_rows = [row for row in combined if row["tier"] == tier]
        write_jsonl(args.output_dir / f"{filename}.jsonl", tier_rows)
        write_csv(args.output_dir / f"{filename}.csv", tier_rows)
    stats["deduplication"] = {
        "selected_before_deduplication": len(all_rows),
        "selected_after_deduplication": len(combined),
        "duplicate_records_removed": len(all_rows) - len(combined),
        "tiers": dict(Counter(row["tier"] for row in combined)),
    }
    with (args.output_dir / "screening_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    write_report(args.output_dir / "screening_report.md", stats, combined)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
