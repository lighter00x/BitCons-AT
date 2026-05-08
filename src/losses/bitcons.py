"""
BitCons: Fragile Bit-Plane Masking Stream utilities.

Provides:
  - apply_bitplane_mask : zero out fragile bit-planes from clean images
  - bitcons_align_loss  : alignment loss between the BitCons stream and the
                          main adversarial stream (JS / KL / MSE / KL-Zscore)
  - get_bitcons_weight  : warmup schedule for the alignment loss coefficient α
"""
import math
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Bit-plane masking
# ─────────────────────────────────────────────────────────────────────────────

def apply_unreliable_bitplane_mask(images: torch.Tensor, fragile_planes: list) -> torch.Tensor:
    """Keep ONLY the fragile bit-planes (complement of apply_bitplane_mask).

    Args:
        images:         Float tensor in [0, 1], shape (B, C, H, W).
        fragile_planes: List of bit-plane indices to keep (0 = LSB, 7 = MSB).

    Returns:
        Masked float tensor in [0, 1] containing only the unreliable bit-planes,
        detached from the computation graph.
    """
    keep_mask = 0
    for p in fragile_planes:
        keep_mask |= (1 << p)
    keep_mask &= 0xFF
    images_int = (images.detach() * 255.0).long().clamp(0, 255)
    return (images_int & keep_mask).float() / 255.0


def apply_bitplane_mask(images: torch.Tensor, fragile_planes: list) -> torch.Tensor:
    """Zero out the specified bit-planes from images in [0, 1].

    Args:
        images:         Float tensor in [0, 1], shape (B, C, H, W).
        fragile_planes: List of bit-plane indices to mask out (0 = LSB, 7 = MSB).

    Returns:
        Masked float tensor in [0, 1], detached from the computation graph.
    """
    keep_mask = 0xFF
    for p in fragile_planes:
        keep_mask &= ~(1 << p) & 0xFF
    images_int = (images.detach() * 255.0).long().clamp(0, 255)
    return (images_int & keep_mask).float() / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Alignment losses
# ─────────────────────────────────────────────────────────────────────────────

def _kl_div(logit1: torch.Tensor, logit2: torch.Tensor) -> torch.Tensor:
    """KL(softmax(logit1) || softmax(logit2))."""
    return F.kl_div(
        F.log_softmax(logit1, dim=1),
        F.softmax(logit2, dim=1),
        reduction='batchmean',
    )


def _js_stable(logit1: torch.Tensor, logit2: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Numerically stable Jensen-Shannon divergence.

    Clamps prob1/prob2 before they are used as KL targets to prevent the
    0 * log(0) = NaN that appears when logits are extreme (e.g. OOD masked
    inputs) and temperature T is small.
    """
    prob1 = F.softmax(logit1 / T, dim=1).clamp(min=1e-8)
    prob2 = F.softmax(logit2 / T, dim=1).clamp(min=1e-8)
    mean_prob = 0.5 * (prob1 + prob2)          # ≥ 5e-9, safe to log directly
    log_mean = torch.log(mean_prob)
    jsd = F.kl_div(log_mean, prob1, reduction='batchmean')
    jsd = jsd + F.kl_div(log_mean, prob2, reduction='batchmean')
    return 0.5 * jsd


def bitcons_align_loss(
    logit_bc: torch.Tensor,
    logit_ref: torch.Tensor,
    align_type: str = 'js',
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute alignment loss between the BitCons stream and the reference stream.

    Args:
        logit_bc:    Raw logits from the BitCons (masked-clean) stream.
        logit_ref:   Raw logits from the reference (adversarial) stream.
                     Should be detached so gradients only flow through logit_bc.
        align_type:  One of {'js', 'kl', 'mse', 'kl_zscore'}.
        temperature: Softmax temperature (used by 'js').

    Returns:
        Scalar alignment loss.
    """
    if align_type == 'js':
        return _js_stable(logit_bc, logit_ref, temperature)
    elif align_type == 'kl':
        return _kl_div(logit_bc, logit_ref)
    elif align_type == 'mse':
        return F.mse_loss(logit_bc, logit_ref)
    elif align_type == 'kl_zscore':
        eps = 1e-8
        z1 = (logit_bc  - logit_bc.mean(1,  keepdim=True)) / (logit_bc.std(1,  keepdim=True) + eps)
        z2 = (logit_ref - logit_ref.mean(1, keepdim=True)) / (logit_ref.std(1, keepdim=True) + eps)
        return _kl_div(z1, z2)
    else:
        raise ValueError(
            f"Unknown BitCons alignment type: '{align_type}'. "
            "Choose from: 'js', 'kl', 'mse', 'kl_zscore'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature-space contrastive loss
# ─────────────────────────────────────────────────────────────────────────────

def bitcons_feature_contrastive_loss(
    feat_bc: torch.Tensor,
    feat_adv: torch.Tensor,
    feat_ub: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """NT-Xent contrastive loss on penultimate-layer features.

    For each sample i in the batch:
      - Anchor   : feat_bc_i   (reliable-bit stream)
      - Positive : feat_adv_i  (adversarial stream, same image → same semantics)
      - Negatives:
          * feat_adv_j  for j ≠ i  (cross-sample adversarial features)
          * feat_ub_j   for all j  (unreliable-bit stream, always noise)

    Gradients flow only through feat_bc; feat_adv and feat_ub are detached.

    Args:
        feat_bc:     (B, D) features from the reliable-bit stream.
        feat_adv:    (B, D) features from the adversarial stream.
        feat_ub:     (B, D) features from the unreliable-bit stream.
        temperature: Contrastive temperature τ.

    Returns:
        Scalar NT-Xent loss.
    """
    z_bc  = F.normalize(feat_bc,           dim=1)   # (B, D)
    z_adv = F.normalize(feat_adv.detach(), dim=1)   # (B, D)
    z_ub  = F.normalize(feat_ub.detach(),  dim=1)   # (B, D)

    B = z_bc.size(0)

    # Similarity matrices scaled by temperature
    sim_adv = (z_bc @ z_adv.T) / temperature        # (B, B)
    sim_ub  = (z_bc @ z_ub.T)  / temperature        # (B, B)

    # Positive score: diagonal of sim_adv, shape (B, 1)
    pos = sim_adv.diagonal().unsqueeze(1)

    # Cross-sample adv negatives: mask out diagonal (= the positive)
    mask = torch.eye(B, dtype=torch.bool, device=z_bc.device)
    sim_adv_neg = sim_adv.masked_fill(mask, float('-inf'))   # (B, B)

    # Concatenate: [positive | adv cross-negatives | ub negatives]
    # Shape: (B, 1 + B + B)
    logits = torch.cat([pos, sim_adv_neg, sim_ub], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=z_bc.device)

    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup weight scheduler
# ─────────────────────────────────────────────────────────────────────────────

def get_bitcons_weight(config, epoch: int) -> float:
    """Return the effective BitCons alignment loss weight at the given epoch.

    The weight linearly (or cosine) warms up from 0 to `bitcons_alpha` over
    the first `bitcons_warmup` epochs, then stays constant at `bitcons_alpha`.

    Config fields used:
        bitcons_alpha            (float, default 1.0)  : final weight value
        bitcons_warmup           (int,   default 0)    : warmup duration in epochs
        bitcons_warmup_schedule  (str,   default 'linear') : 'linear' or 'cosine'
    """
    alpha    = float(getattr(config, 'bitcons_alpha',           1.0)     or 1.0)
    warmup   = int(  getattr(config, 'bitcons_warmup',          0)       or 0)
    schedule =       getattr(config, 'bitcons_warmup_schedule', 'linear') or 'linear'

    if warmup <= 0 or epoch >= warmup:
        return alpha

    progress = epoch / warmup          # in [0, 1)
    if schedule == 'cosine':
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:                              # linear
        factor = progress

    return alpha * factor