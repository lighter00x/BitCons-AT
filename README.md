# BitCons-AT

**Focus on what you see: High-information Robust Bit-plane Alignment Adversarial Training**

中文代码说明见 [项目代码说明](项目代码说明.md)，下面保留原始英文概述和快速上手。

## Core Idea

Adversarial perturbations are constrained to be imperceptible, so they preferentially exploit the **low-order (fragile) bit-planes** of pixel values where small changes have the least visual impact but can still mislead classifiers. Meanwhile, semantic image content is concentrated in the **high-order bit-planes**.

BitCons-AT introduces a lightweight third training stream alongside the standard adversarial stream:

1. **Fragile plane masking** — Given the final adversarial image, zero out bits 0–2 (the three LSBs) to produce a nearby reliable-content view.
2. **BitCons CE loss** — Train the model to correctly classify this masked view.
3. **Alignment loss** — Align the masked-view logits with the detached final-adversarial logits, so the model learns invariance to low-bit changes inside the threat radius.

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

### Two-GPU complete experiment suite

The complete single-seed CIFAR-10 suite can be started once and left running:

```bash
chmod +x run_experiment_suite.sh
nohup ./run_experiment_suite.sh > suite_launcher.log 2>&1 &
```

By default it uses Conda environment `bit`, GPUs `0 1`, ResNet18, seed `4243`,
and 110 epochs. It runs the Base/Core/Full comparison for PGD-AT, TRADES,
MART, and RPAT, completes all eight PGD-AT component ablations, then evaluates
the best checkpoint with Clean, PGD-10/20/50, C&W, and AutoAttack. Final
checkpoints are still saved, but their complete evaluation is disabled by
default. Set `RUN_FINAL_EVAL=1` with the same `RUN_ID` to evaluate them later.
Each GPU runs at most one task at a time. Logs and the task manifest are saved
under `logs/suite_<RUN_ID>/`. Metrics from every completed evaluation are
consolidated into `results.tsv` in the same directory.

Preview without starting GPU work:

```bash
DRY_RUN=1 ./run_experiment_suite.sh
```

To continue evaluation for an existing suite, reuse its run ID:

```bash
RUN_ID=<existing_run_id> STAGE=eval ./run_experiment_suite.sh
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

#### With BitCons

Enable the alignment stream explicitly from the command line:

```bash
python src/train.py --dataset cifar10 --model preactresnet18 --config pgd_at \
    --bitcons
```

Enable both alignment and feature contrastive loss:

```bash
python src/train.py --dataset cifar10 --model preactresnet18 --config pgd_at \
    --bitcons --bitcons_contrast
```

The current low-bit candidate defaults can also be changed in a training YAML:

```yaml
bitcons: true
bitcons_contrast: true
bitcons_planes: [0, 1, 2]
bitcons_align: kl
bitcons_alpha: 0.25
bitcons_ce_weight: 1.0
bitcons_align_weight: 1.0
bitcons_warmup: 60
bitcons_warmup_schedule: linear
bitcons_contrast_lam: 0.001
```

#### Override BitCons hyperparameters via CLI

```bash
python src/train.py --dataset cifar10 --model preactresnet18 --config pgd_at \
    --bitcons_planes 0 1 2        \
    --bitcons_align  kl_zscore    \
    --bitcons_alpha  0.5          \
    --bitcons_ce_weight 1.0       \
    --bitcons_align_weight 1.0    \
    --bitcons_warmup 30           \
    --bitcons_warmup_schedule cosine
```

#### Combine BitCons with weight perturbation (AWP)

```bash
python src/train.py --dataset cifar10 --model preactresnet18 --config pgd_at \
    --perturbation awp --bitcons
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
| `bitcons` | `--bitcons` / `--no-bitcons` | `false` | Enable the BitCons stream |
| `bitcons_contrast` | `--bitcons_contrast` / `--no-bitcons_contrast` | `false` | Enable feature contrastive loss; requires BitCons |
| `bitcons_planes` | `--bitcons_planes` | `[0, 1, 2]` | Bit-planes to zero out (0=LSB, 7=MSB) |
| `bitcons_align` | `--bitcons_align` | `kl` | Alignment loss: `js` / `kl` / `mse` / `kl_zscore` |
| `bitcons_alpha` | `--bitcons_alpha` | `0.25` | Final weight of the complete auxiliary branch |
| `bitcons_ce_weight` | `--bitcons_ce_weight` | `1.0` | Masked-view CE weight inside the BitCons branch |
| `bitcons_align_weight` | `--bitcons_align_weight` | `1.0` | Logit alignment weight inside the BitCons branch |
| `bitcons_warmup` | `--bitcons_warmup` | `60` | Epochs to ramp α: 0 → `bitcons_alpha` (0 = no warmup) |
| `bitcons_warmup_schedule` | `--bitcons_warmup_schedule` | `linear` | Warmup shape: `linear` or `cosine` |
| `temperature` | `--temperature` | `1.0` | Softmax temperature (used by `js` alignment) |
| `bitcons_contrast_lam` | `--bitcons_contrast_lam` | `0.001` | Feature contrastive loss weight relative to α |
| `bitcons_contrast_temp` | `--bitcons_contrast_temp` | `0.5` | Feature contrastive temperature |

### Alignment loss options

| Type | Notes |
|---|---|
| `js` | JS divergence — **default**; symmetric, numerically stable (prob clamping) |
| `kl` | KL(reference ∥ BC) — teacher-to-masked-view alignment |
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
from training.methods.utils import freeze_batchnorm_stats

# After computing logits_adv in the training loop:
if getattr(config, 'bitcons', False):
    bc_alpha = get_bitcons_weight(config, epoch)   # respects warmup schedule
    if bc_alpha > 0:
        images_bc     = apply_bitplane_mask(adv_images, config.bitcons_planes)
        with freeze_batchnorm_stats(model):
            logits_bc = model(images_bc)
        loss_bc_ce    = criterion(logits_bc, labels)
        loss_bc_align = bitcons_align_loss(
            logits_bc, logits_adv.detach(),
            config.bitcons_align, config.temperature or 1.0,
        )
        loss = loss + bc_alpha * (
            config.bitcons_ce_weight * loss_bc_ce
            + config.bitcons_align_weight * loss_bc_align
        )
```

Add the BitCons fields to `configs/training/<your_method>.yaml` with `bitcons: false` as default.
