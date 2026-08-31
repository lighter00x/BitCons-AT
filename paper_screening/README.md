# Paper Coarse Screening

This directory contains a reproducible, high-recall coarse screen for the
BitCons/BitPlane/BPDA/BitMax literature review.

Run from the repository root:

```bash
python paper_screening/coarse_screen.py \
  --input-dir /mnt/data/share/xq/source/papers \
  --output-dir paper_screening/results
```

The source JSON files are read only. Results contain the matched topic groups,
score, tier, and exact title/abstract snippets that caused each match.

Tiers:

- `A-Core`: adversarial robustness plus a target mechanism, with additional
  evidence that the paper concerns image classification or an input
  transformation/compression defense.
- `B-Method`: adversarial robustness plus consistency/alignment or evaluation
  protocol work.
- `C-Foundation`: other explicit adversarial training/attack/defense papers,
  retained to preserve recall.
- `D-Exclude`: not emitted to selected result files.

The rules deliberately down-rank weight-only quantization and non-vision
domains when there is no countervailing input/vision evidence. They do not
hard-delete them during the coarse stage.
