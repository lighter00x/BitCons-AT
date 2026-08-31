# Coarse Screening Report

Rules version: `2`

## Source Summary

| Source | Total | Selected | A-Core | B-Method | C-Foundation | Missing abstract |
|---|---:|---:|---:|---:|---:|---:|
| cvpr2026_papers.json | 5024 | 72 | 1 | 6 | 65 | 955 |
| iclr2026_papers.json | 10000 | 174 | 11 | 13 | 150 | 0 |
| icml2024_papers.json | 2610 | 62 | 2 | 5 | 55 | 0 |
| icml2025_papers.json | 3330 | 54 | 1 | 2 | 51 | 0 |
| neurips2025_papers.json | 5540 | 97 | 3 | 7 | 87 | 0 |

## Combined Results

- Selected before title deduplication: 459
- Selected after title deduplication: 458
- Duplicate records removed: 1
- Tier counts: {"A-Core": 18, "B-Method": 33, "C-Foundation": 407}

## A-Core Candidates

| Score | Venue | Year | Title | Matched groups |
|---:|---|---:|---|---|
| 26 | ICLR | 2026 | DRIFT: Divergent Response in Filtered Transformations for Robust Adversarial Defense | adaptive_evaluation, adversarial_core, classification_context, consistency_alignment, input_transform_compression, vision_context |
| 23 | ICLR | 2026 | Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification | adaptive_evaluation, adversarial_core, input_transform_compression, vision_context |
| 19 | ICLR | 2026 | Randomized Feature Squeezing against Unseen $ {l_p} $ Attacks without Adversarial Training | adversarial_core, classification_context, input_bit_quantization, vision_context |
| 19 | ICML | 2024 | Robust Classification via a Single Diffusion Model | adaptive_evaluation, adversarial_core, classification_context, vision_context |
| 18 | ICLR | 2026 | Boosting Adversarial Robustness and Generalization with Dictionary Structure | adaptive_evaluation, adversarial_core, classification_context, evaluation_protocol |
| 17 | ICLR | 2026 | NIC-RobustBench: A Comprehensive Open-Source Toolkit for Neural Image Compression and Robustness Analysis | adversarial_core, evaluation_protocol, input_transform_compression, vision_context |
| 16 | NEURIPS | 2025 | Adversary Aware Optimization for Robust Defense | adversarial_core, classification_context, input_transform_compression, vision_context |
| 16 | ICML | 2024 | Diffusion Models Demand Contrastive Guidance for Adversarial Purification to Advance | adversarial_core, classification_context, input_transform_compression, vision_context |
| 16 | ICLR | 2026 | Progressive Residual Tensor Networks for Adversarial Purification | adversarial_core, classification_context, input_transform_compression, vision_context |
| 15 | NEURIPS | 2025 | Model-Free Adversarial Purification via Coarse-To-Fine Tensor Network Representation | adversarial_core, classification_context, input_transform_compression, vision_context |
| 15 | ICLR | 2026 | Pro-Trans: Progressive Tensor Ring with Attention Guided Local Smoothing Regularization | adversarial_core, classification_context, input_transform_compression, vision_context |
| 14 | ICLR | 2026 | Feature compression is the root cause of adversarial fragility in neural networks | adversarial_core, classification_context, input_transform_compression, vision_context |
| 13 | ICLR | 2026 | Adversarial Attacks Already Tell the Answer: Directional Bias-Guided Test-time Defense for Vision-Language Models | adversarial_core, input_transform_compression, vision_context |
| 13 | ICLR | 2026 | Diffusion Models Improve Adversarial Robustness by Compressing Image Space | adversarial_core, input_transform_compression, vision_context |
| 13 | ICML | 2025 | Diffusion-based Adversarial Purification from the Perspective of the Frequency Domain | adversarial_core, input_transform_compression, vision_context |
| 13 | CVPR | 2026 | Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction | adversarial_core, input_transform_compression, vision_context |
| 12 | NEURIPS | 2025 | MAE-Pure: Semantic-Preserving Adversarial Purification | adversarial_core, input_transform_compression, vision_context |
| 12 | ICLR | 2026 | Multimodal Information is All You Need for Adversarial Purification via Diffusion Models | adversarial_core, input_transform_compression, vision_context |

## Interpretation Limits

- This is a high-recall lexical screen, not a final relevance judgment.
- Venue JSON files do not contain acceptance decisions; counts must not be treated as accepted-paper counts.
- Missing CVPR abstracts can cause false negatives.
- A-Core papers still require abstract/PDF review to separate image classification from adjacent generative, 3D, medical, and multimodal tasks.
- The 2024-2026 corpus does not replace seminal-paper and citation-chain retrieval.
