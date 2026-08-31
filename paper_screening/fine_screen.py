#!/usr/bin/env python3
"""Curated abstract-level fine screening over coarse BitCons candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# These papers were retained after title/abstract review. The reason is specific
# to the BitPlane/BPDA/BitMax research question rather than generic robustness.
INCLUDED: dict[str, tuple[str, str, int, str]] = {
    "DRIFT: Divergent Response in Filtered Transformations for Robust Adversarial Defense": (
        "adaptive_input_defense", "P0", 98,
        "Directly studies transformed-input defenses, adaptive attacks, gradient masking, and prediction/logit behavior.",
    ),
    "Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification": (
        "adaptive_evaluation", "P0", 97,
        "Direct adaptive evaluation of lossy-compression preprocessing and explicit analysis of gradient masking.",
    ),
    "Randomized Feature Squeezing against Unseen $ {l_p} $ Attacks without Adversarial Training": (
        "input_discretization", "P0", 96,
        "Direct input-layer feature squeezing defense and the closest recent work to bit-depth reduction.",
    ),
    "DiffBreak: Is Diffusion-Based Purification Robust?": (
        "adaptive_evaluation", "P0", 95,
        "Shows how inaccurate gradients and improper stochastic evaluation inflate preprocessing-defense robustness.",
    ),
    "A Combination of Noise and Bilateral Filters Achieve Supralinear and Scalable Adversarial Robustness in CNNs": (
        "adaptive_input_defense", "P0", 94,
        "Evaluates a simple input preprocessor combined with adversarial training and AutoAttack.",
    ),
    "On the Adversarial Robustness of Discrete Image Tokenizers": (
        "discrete_representation", "P0", 93,
        "Directly analyzes adversarial vulnerability and adversarial training of discrete visual representations.",
    ),
    "NIC-RobustBench: A Comprehensive Open-Source Toolkit for Neural Image Compression and Robustness Analysis": (
        "compression_evaluation", "P0", 92,
        "Provides attacks, defenses, and evaluation infrastructure for neural image compression robustness.",
    ),
    "Diffusion Models Improve Adversarial Robustness by Compressing Image Space": (
        "compression_mechanism", "P0", 90,
        "Mechanistic analysis connecting image-space compression to adversarial robustness gains.",
    ),
    "Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks": (
        "adaptive_evaluation", "P0", 89,
        "Surrogate-gradient evaluation of a discontinuous model is methodologically analogous to BPDA validation.",
    ),
    "TriQDef: Disrupting Semantic and Gradient Alignment to Prevent Adversarial Patch Transferability in Quantized Neural Networks": (
        "quantization_adjacent", "P1", 86,
        "Studies cross-bit adversarial behavior and semantic/gradient alignment in quantized networks; weights rather than inputs.",
    ),
    "Treating Neural Image Compression via Modular Adversarial Optimization: From Global Distortion to Local Artifacts": (
        "compression_adjacent", "P1", 82,
        "Adversarial optimization over neural image compression is useful for understanding discrete/compressed image spaces.",
    ),
    "Feature compression is the root cause of adversarial fragility in neural networks": (
        "compression_mechanism", "P1", 80,
        "Provides a theoretical feature-compression explanation for classifier adversarial fragility.",
    ),
    "On the Interaction of Compressibility and Adversarial Robustness": (
        "compression_mechanism", "P1", 78,
        "Analyzes how structured model/representation compressibility creates adversarially sensitive directions.",
    ),
    "Robust Classification via a Single Diffusion Model": (
        "adaptive_input_defense", "P1", 82,
        "Image classification defense evaluated against adaptive L-infinity attacks at the same 8/255 scale.",
    ),
    "Adversary Aware Optimization for Robust Defense": (
        "input_purification", "P1", 78,
        "Optimization-based test-time input purification inside an explicit adversarial perturbation space.",
    ),
    "Diffusion Models Demand Contrastive Guidance for Adversarial Purification to Advance": (
        "input_purification", "P1", 77,
        "Uses contrastive guidance in input purification and is relevant to the failed BitCons contrastive branch.",
    ),
    "Progressive Residual Tensor Networks for Adversarial Purification": (
        "input_purification", "P1", 74,
        "Frequency-aware input reconstruction illustrates semantic-preservation versus perturbation-removal trade-offs.",
    ),
    "Model-Free Adversarial Purification via Coarse-To-Fine Tensor Network Representation": (
        "input_purification", "P1", 74,
        "Model-free input purification with explicit adversarial optimization and multiple norm threats.",
    ),
    "Pro-Trans: Progressive Tensor Ring with Attention Guided Local Smoothing Regularization": (
        "input_purification", "P1", 72,
        "Local smoothing purification is an adjacent deterministic input-transformation defense.",
    ),
    "Diffusion-based Adversarial Purification from the Perspective of the Frequency Domain": (
        "input_purification", "P1", 76,
        "Analyzes which image-frequency components are damaged by adversarial perturbations and purification.",
    ),
    "MAE-Pure: Semantic-Preserving Adversarial Purification": (
        "input_purification", "P1", 74,
        "Input optimization preserving patch semantics is relevant to reliable-bit semantic preservation.",
    ),
    "Multimodal Information is All You Need for Adversarial Purification via Diffusion Models": (
        "input_purification", "P2", 65,
        "A more distant purification method, retained for comparison of label/semantic preservation mechanisms.",
    ),
    "One Stone, Two Birds: Enhancing Adversarial Defense Through the Lens of Distributional Discrepancy": (
        "alignment_defense", "P1", 76,
        "Uses clean/adversarial distribution alignment to train a denoiser and evaluates adaptive white-box attacks.",
    ),
    "SINAI: Strategic Injection of Noise for Adversarial Defense with Improved Accuracy–Robustness Tradeoffs": (
        "randomized_defense", "P2", 68,
        "Noise-based defense provides a useful randomized alternative to deterministic low-bit masking.",
    ),
    "Boosting Adversarial Robustness and Generalization with Dictionary Structure": (
        "evaluation_support", "P1", 78,
        "Explicitly identifies false security and validates robustness under strong adaptive attacks and RobustBench.",
    ),
    "Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies": (
        "evaluation_support", "P1", 77,
        "Provides strong AutoAttack baselines and analyzes invalid images under the standard L-infinity threat model.",
    ),
    "Be Your Own Neighborhood: Detecting Adversarial Examples by the Neighborhood Relations Built on Self-Supervised Learning": (
        "consistency_support", "P1", 72,
        "Studies representation and label consistency under adaptive attacks, directly informing consistency-based defenses.",
    ),
    "Two Heads are Actually Better than One: Towards Better Adversarial Robustness via Transduction and Rejection": (
        "evaluation_support", "P2", 66,
        "Demonstrates how stronger adaptive evaluation changes defense conclusions and reports AutoAttack/GMSA.",
    ),
    "Average Certified Radius is a Poor Metric for Randomized Smoothing": (
        "evaluation_support", "P1", 74,
        "A metric-failure study useful for the paper's broader argument about misleading robustness evaluation.",
    ),
    "OODRobustBench: a Benchmark and Large-Scale Analysis of Adversarial Robustness under Distribution Shift": (
        "evaluation_support", "P2", 68,
        "Large-scale robustness benchmark covering unseen threat-model shifts.",
    ),
    "Position: Certified Robustness Does Not (Yet) Imply Model Security": (
        "evaluation_support", "P2", 63,
        "Supports careful separation of formal robustness claims from practical security conclusions.",
    ),
    "Probabilistic Robustness for Free? Revisiting Training via a Benchmark": (
        "evaluation_support", "P2", 65,
        "Highlights non-comparable protocols and establishes strong adversarial-training baseline comparisons.",
    ),
    "Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation": (
        "consistency_support", "P1", 72,
        "Analyzes robustness transfer and consistency weighting on CIFAR/Tiny-ImageNet with AutoAttack.",
    ),
    "Improving Accuracy-robustness Trade-off via Pixel Reweighted Adversarial Training": (
        "bitmax_support", "P1", 79,
        "Uses spatially nonuniform pixel perturbation budgets, relevant to structured pixel/bit inner maximization.",
    ),
    "Vulnerable Data-Aware Adversarial Training": (
        "adversarial_training_support", "P2", 64,
        "Sample-wise worst-case selection and training efficiency are relevant to BitMax candidate selection.",
    ),
    "Robust Fine-Tuning from Non-Robust Pretrained Models: Mitigating Suboptimal Transfer With Epsilon-Scheduling": (
        "adversarial_training_support", "P2", 62,
        "Perturbation-strength scheduling helps interpret optimization collapse under overly strong auxiliary objectives.",
    ),
    "Boosting Adversarial Robustness with CLAT: Criticality Leveraged Adversarial Training": (
        "adversarial_training_support", "P2", 62,
        "Selective robust fine-tuning provides an optimization comparison for regularization-induced trade-offs.",
    ),
    "Class-Wise Disparity in Adversarial Training: Implicit Bias Perspective": (
        "adversarial_training_support", "P2", 60,
        "Relevant for class-wise robustness diagnostics beyond aggregate accuracy.",
    ),
    "Layer-Aware Analysis of Catastrophic Overfitting: Revealing the Pseudo-Robust Shortcut Dependency": (
        "failure_mechanism", "P1", 72,
        "Pseudo-robust shortcuts and layer-wise distortion provide a close analogue to false masked robustness.",
    ),
    "Mitigating Error Amplification in Fast Adversarial Training": (
        "failure_mechanism", "P2", 65,
        "Analyzes catastrophic overfitting and spurious correlations under perturbation-strength changes.",
    ),
    "Nasty Adversarial Training: A Probability Sparsity Perspective for Robustness Enhancement": (
        "adversarial_training_support", "P2", 60,
        "A recent adversarial-training regularizer useful as an auxiliary-objective comparison.",
    ),
    "Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations": (
        "bitmax_support", "P1", 84,
        "Directly addresses sparse/discrete pixel perturbations, reliable attacks, and adversarial training.",
    ),
    "Understanding and Improving Fast Adversarial Training against $l_0$ Bounded Perturbations": (
        "bitmax_support", "P1", 83,
        "Explains optimization difficulty and perturbation-location selection for discrete sparse attacks.",
    ),
    "Uniformly Stable Algorithms for Adversarial Training and Beyond": (
        "failure_mechanism", "P2", 64,
        "Connects training instability to robust overfitting and proposes a stability-oriented optimizer.",
    ),
    "On the Duality Between Sharpness-Aware Minimization and Adversarial Training": (
        "adversarial_training_support", "P2", 61,
        "Useful comparison between input-space and weight-space perturbation objectives.",
    ),
    "DataFreeShield: Defending Adversarial Attacks without Training Data": (
        "adversarial_training_support", "P2", 58,
        "Retained as a modern adversarial-training baseline, but not directly tied to bit-plane mechanisms.",
    ),
    "Benign Overfitting in Adversarial Training of Neural Networks": (
        "failure_mechanism", "P2", 58,
        "Theoretical background on robust generalization and overfitting in adversarial training.",
    ),
    "On the Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics": (
        "failure_mechanism", "P2", 61,
        "Directly analyzes robust memorization, robust generalization gaps, and training dynamics.",
    ),
}


PENDING_TITLES = {
    "Robust Alignment: Harmonizing Clean Accuracy and Adversarial Robustness in Adversarial Training":
        "The title is relevant to BitCons alignment, but the crawled record has no abstract; inspect the PDF before inclusion.",
}

EVIDENCE_TERMS = re.compile(
    r"bit|quanti[sz]|discret|compress|preprocess|purif|adaptive attack|gradient|"
    r"masking|AutoAttack|adversarial train|consisten|align|contrast|robust overfit|"
    r"catastrophic overfit|CIFAR|ImageNet|threat model|restart|surrogate",
    re.IGNORECASE,
)


def evidence_sentences(abstract: str, limit: int = 4) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip()) if abstract else []
    evidence = [sentence for sentence in sentences if EVIDENCE_TERMS.search(sentence)]
    return evidence[:limit]


def exclusion_reason(record: dict[str, Any]) -> tuple[str, str]:
    title = record["title"]
    abstract = record.get("abstract", "")
    text = f"{title} {abstract}".lower()
    if not abstract:
        return "missing_abstract", "No abstract is available and the title alone does not justify inclusion."
    if re.search(r"language model|\bllm|lvlm|vision-language|multimodal|jailbreak|prompt injection", text):
        return "different_domain", "Focuses on language/multimodal model safety rather than image-classifier pixel robustness."
    if re.search(r"watermark|unlearn|copyright|mimicry|personalized generation|concept erasure|backdoor", text):
        return "different_security_problem", "Focuses on watermarking, unlearning, ownership protection, or backdoors rather than classifier robustness."
    if re.search(r"bandit|reinforcement learning|graph neural|time series|speech|voice|weather|control system", text):
        return "different_domain", "The adversarial problem is outside image-classifier robustness."
    if re.search(r"attack|adversarial example", text) and not re.search(r"defen[cs]e|robust training|adversarial training|robustness evaluation|benchmark", text):
        return "attack_only", "Primarily proposes an attack and does not inform the target defense, training, or evaluation mechanism."
    if re.search(r"certif|verification|randomized smoothing", text):
        return "certification_only", "Primarily concerns certification/verification without a direct link to the target empirical defense."
    if re.search(r"adversarial training|adversarial robustness|robust accuracy", text):
        return "general_robustness", "Relevant to adversarial robustness generally, but not sufficiently close to bit/input transforms, adaptive evaluation, discrete inner maximization, or BitCons failure mechanisms."
    return "lexical_false_positive", "The abstract does not address the target adversarial image-classification research question."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_reviews = []
    selected = []
    pending = []
    found_included = set()
    for record in records:
        title = record["title"]
        review = dict(record)
        review["fine_screen_version"] = 1
        review["abstract_reviewed"] = bool(record.get("abstract"))
        review["fine_evidence"] = evidence_sentences(record.get("abstract", ""))
        if title in INCLUDED:
            category, priority, relevance, reason = INCLUDED[title]
            review.update({
                "fine_decision": "include",
                "fine_category": category,
                "priority": priority,
                "relevance_score_100": relevance,
                "fine_reason": reason,
            })
            selected.append(review)
            found_included.add(title)
        elif title in PENDING_TITLES:
            review.update({
                "fine_decision": "manual_fulltext_needed",
                "fine_category": "missing_abstract",
                "priority": "P1-pending",
                "relevance_score_100": None,
                "fine_reason": PENDING_TITLES[title],
            })
            pending.append(review)
        else:
            category, reason = exclusion_reason(record)
            review.update({
                "fine_decision": "exclude",
                "fine_category": category,
                "priority": None,
                "relevance_score_100": None,
                "fine_reason": reason,
            })
        all_reviews.append(review)

    missing_curated = sorted(set(INCLUDED) - found_included)
    if missing_curated:
        raise RuntimeError(f"Curated titles missing from input: {missing_curated}")

    selected.sort(key=lambda row: (-row["relevance_score_100"], row["title"].lower()))
    all_reviews.sort(key=lambda row: (
        {"include": 0, "manual_fulltext_needed": 1, "exclude": 2}[row["fine_decision"]],
        -(row["relevance_score_100"] or 0),
        row["title"].lower(),
    ))

    (args.output_dir / "fine_selected_papers.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "fine_review_all_candidates.json").write_text(
        json.dumps(all_reviews, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "manual_fulltext_needed.json").write_text(
        json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "input_candidates": len(records),
        "included": len(selected),
        "manual_fulltext_needed": len(pending),
        "excluded": len(records) - len(selected) - len(pending),
        "priorities": dict(Counter(row["priority"] for row in selected)),
        "included_categories": dict(Counter(row["fine_category"] for row in selected)),
        "exclusion_categories": dict(Counter(
            row["fine_category"] for row in all_reviews if row["fine_decision"] == "exclude"
        )),
    }
    (args.output_dir / "fine_screening_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# Fine Screening Report",
        "",
        "The coarse candidates were reviewed using title and abstract evidence against the BitPlane/BPDA/BitMax research scope.",
        "",
        f"- Input candidates: {stats['input_candidates']}",
        f"- Included: {stats['included']}",
        f"- Manual full-text review needed: {stats['manual_fulltext_needed']}",
        f"- Excluded: {stats['excluded']}",
        f"- Priorities: `{json.dumps(stats['priorities'], ensure_ascii=False)}`",
        "",
        "## Reading Order",
        "",
        "| Priority | Score | Category | Venue | Year | Title |",
        "|---|---:|---|---|---:|---|",
    ]
    for row in selected:
        title = row["title"].replace("|", "\\|")
        report_lines.append(
            f"| {row['priority']} | {row['relevance_score_100']} | {row['fine_category']} | "
            f"{row['venue']} | {row['year']} | {title} |"
        )
    report_lines.extend([
        "",
        "## Decision Semantics",
        "",
        "- `P0`: directly supports input discretization/transformation or adaptive evaluation; read first.",
        "- `P1`: strong method, mechanism, or evaluation support.",
        "- `P2`: background material useful for experiments, failure analysis, or baselines.",
        "- Relevance scores are normalized to 100 and are curated reading priorities, not acceptance probabilities.",
        "- `fine_review_all_candidates.json` preserves an include/exclude decision and reason for every coarse candidate.",
        "",
    ])
    (args.output_dir / "fine_screening_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
