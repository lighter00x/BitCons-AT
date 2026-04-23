# BitCons-AT

**Focus on what you see: High-information Robust Bit-plane Alignment Adversarial Training**

中文代码说明见 [项目代码说明](项目代码说明.md)，下面保留原始英文概述和快速上手。

## Core Idea

Adversarial perturbations are constrained to be imperceptible, so they preferentially exploit the **low-order (fragile) bit-planes** of pixel values where small changes have the least visual impact but can still mislead classifiers. Meanwhile, semantic image content is concentrated in the **high-order bit-planes**.

BitCons-AT introduces a lightweight third training stream alongside the standard adversarial stream:

1. **Fragile plane masking** — Given a clean image, zero out the specified bit-planes (e.g., bits 0–2, the three LSBs) to produce a "reliable-content-only" view.
2. **BitCons CE loss** — Train the model to correctly classify this masked view.
3. **Alignment loss** — Align the masked-view logits with the adversarial-stream logits, so the model learns that high-information clean content and adversarially-perturbed content produce consistent predictions.

This third stream is **plug-and-play**: enabled by a single flag in any existing config, without modifying the base method's logic.

The alignment loss weight **α warms up** from 0 to its final value over a configurable number of epochs, stabilising early training.

```
Total loss = L_base  +  α(t) · (L_bc_ce + L_bc_align)
```

where `α(t)` follows a linear or cosine schedule from 0 → `bitcons_alpha` over `bitcons_warmup` epochs.

---

## Project Structure

```
BitCons-AT/
├── configs/
│   ├── datasets/        # cifar10.yaml, cifar100.yaml, ...
│   └── training/        # each config has a BitCons section (bitcons: false by default)
│       ├── pgd_at.yaml
│       ├── trades.yaml
│       ├── mart.yaml
│       ├── cons_at.yaml
│       └── rpat.yaml
└── src/
    ├── train.py
    ├── eval.py
    ├── losses/
    │   └── bitcons.py   # apply_bitplane_mask / bitcons_align_loss / get_bitcons_weight
    └── training/methods/  # each method checks config.bitcons
```

---

## Quick Start

### 1. Dependencies

```bash
pip install torch torchvision
pip install autoattack   # for AutoAttack evaluation
```

### 2. Training

#### Without BitCons (default)

```bash
python src/train.py --dataset cifar10 --model preactresnet18 --config pgd_at
python src/train.py --dataset cifar10 --model preactresnet18 --config trades
python src/train.py --dataset cifar10 --model preactresnet18 --config mart
python src/train.py --dataset cifar10 --model preactresnet18 --config cons_at
python src/train.py --dataset cifar10 --model preactresnet18 --config rpat
```

#### With BitCons — set `bitcons: true` in the config YAML

Edit `configs/training/pgd_at.yaml` (or any other method config):

```yaml
bitcons: true            # ← flip this to enable
bitcons_planes: [0, 1, 2]
bitcons_align: js
bitcons_alpha: 1.0
bitcons_warmup: 20
bitcons_warmup_schedule: linear
```

Then run as usual:

```bash
python src/train.py --dataset cifar10 --model preactresnet18 --config pgd_at
```

#### Override BitCons hyperparameters via CLI

```bash
python src/train.py --dataset cifar10 --model preactresnet18 --config pgd_at \
    --bitcons_planes 0 1 2 3      \
    --bitcons_align  kl_zscore    \
    --bitcons_alpha  0.5          \
    --bitcons_warmup 30           \
    --bitcons_warmup_schedule cosine
```

#### Combine BitCons with weight perturbation (AWP)

```bash
python src/train.py --dataset cifar10 --model preactresnet18 --config pgd_at \
    --perturbation awp
# (bitcons: true already set in pgd_at.yaml)
```

### 3. Evaluation

```bash
# Standard evaluation (PGD-10/20/50, C&W)
python src/eval.py --exp <experiment_folder>

# Full suite including AutoAttack
python src/eval.py --exp <experiment_folder> --all-attacks

# With test-time bit-plane masking
python src/eval.py --exp <experiment_folder> --bitcons-test --bitcons-planes 0 1 2
```

---

## BitCons Hyperparameters

| Config key | CLI override | Default | Description |
|---|---|---|---|
| `bitcons` | — | `false` | Set to `true` in the YAML to enable BitCons |
| `bitcons_planes` | `--bitcons_planes` | `[0, 1, 2]` | Bit-planes to zero out (0=LSB, 7=MSB) |
| `bitcons_align` | `--bitcons_align` | `js` | Alignment loss: `js` / `kl` / `mse` / `kl_zscore` |
| `bitcons_alpha` | `--bitcons_alpha` | `1.0` | Final alignment loss weight α |
| `bitcons_warmup` | `--bitcons_warmup` | `20` | Epochs to ramp α: 0 → `bitcons_alpha` (0 = no warmup) |
| `bitcons_warmup_schedule` | `--bitcons_warmup_schedule` | `linear` | Warmup shape: `linear` or `cosine` |
| `temperature` | `--temperature` | `1.0` | Softmax temperature (used by `js` alignment) |

### Alignment loss options

| Type | Notes |
|---|---|
| `js` | JS divergence — **default**; symmetric, numerically stable (prob clamping) |
| `kl` | KL(BC ∥ adv) — asymmetric, slightly faster |
| `mse` | MSE on raw logits — simplest, scale-sensitive |
| `kl_zscore` | KL after Z-score normalising logits — robust to logit scale differences |

### α warmup schedule

```
linear:  α(t) = α_final × (t / warmup)
cosine:  α(t) = α_final × 0.5 × (1 − cos(π × t / warmup))
```

Both reach `α_final` at epoch `warmup` and stay constant. Set `bitcons_warmup: 0` to skip warmup.

---

## Adding BitCons to a new method

```python
from losses import apply_bitplane_mask, bitcons_align_loss, get_bitcons_weight

# After computing logits_adv in the training loop:
if getattr(config, 'bitcons', False):
    bc_alpha = get_bitcons_weight(config, epoch)   # respects warmup schedule
    if bc_alpha > 0:
        images_bc     = apply_bitplane_mask(clean_images, config.bitcons_planes)
        logits_bc     = model(images_bc)
        loss_bc_ce    = criterion(logits_bc, labels)
        loss_bc_align = bitcons_align_loss(
            logits_bc, logits_adv.detach(),
            config.bitcons_align, config.temperature or 1.0,
        )
        loss = loss + bc_alpha * (loss_bc_ce + loss_bc_align)
```

Add the BitCons fields to `configs/training/<your_method>.yaml` with `bitcons: false` as default.
