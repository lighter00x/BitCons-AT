import torch
import torch.nn.functional as F

from .pgd import PGD


def _validate_low_planes(planes):
    planes = sorted(set(int(plane) for plane in planes))
    if not planes or planes != list(range(planes[-1] + 1)):
        raise ValueError(
            'BitMax planes must be contiguous low bits, e.g. [0, 1, 2]'
        )
    return planes


def generate_bit_candidates(images, clean_images, planes, eps, count):
    """Generate low-bit variants and project every candidate into the L-inf ball."""
    planes = _validate_low_planes(planes)
    if count < 1:
        raise ValueError('BitMax candidate count must be positive')

    low_mask = (1 << (planes[-1] + 1)) - 1
    high_mask = 0xFF ^ low_mask
    quantized = (images.detach() * 255.0).round().long().clamp(0, 255)
    high_bits = quantized & high_mask

    low_values = [torch.zeros_like(quantized)]
    if count >= 2:
        low_values.append(torch.full_like(quantized, low_mask))
    for _ in range(max(count - 2, 0)):
        low_values.append(
            torch.randint(0, low_mask + 1, quantized.shape, device=images.device)
        )

    lower = (clean_images.detach() - eps).clamp(0.0, 1.0)
    upper = (clean_images.detach() + eps).clamp(0.0, 1.0)
    candidates = []
    for low_bits in low_values:
        candidate = ((high_bits | low_bits).float() / 255.0)
        candidate = torch.maximum(torch.minimum(candidate, upper), lower)
        candidates.append(candidate.detach())
    return candidates


def select_bitmax_candidate(
    model,
    clean_images,
    adv_images,
    labels,
    planes,
    eps,
    count,
    refine_alpha=None,
    refine_steps=0,
):
    """Select the highest-CE candidate per sample, including standard PGD."""
    candidates = [adv_images.detach()]
    bit_seeds = generate_bit_candidates(
        adv_images, clean_images, planes, eps, count
    )
    if refine_steps > 0:
        if refine_alpha is None:
            raise ValueError('refine_alpha is required when refine_steps > 0')
        refiner = PGD(
            eps=eps,
            alpha=refine_alpha,
            steps=refine_steps,
            random_start=False,
            model_eval=True,
        )
        candidates.extend(
            refiner(model, clean_images, labels, initial=seed)
            for seed in bit_seeds
        )
    else:
        candidates.extend(bit_seeds)

    with torch.no_grad():
        was_training = model.training
        model.eval()
        try:
            losses = [
                F.cross_entropy(model(candidate), labels, reduction='none')
                for candidate in candidates
            ]
        finally:
            model.train(was_training)

    loss_matrix = torch.stack(losses, dim=0)
    selected_indices = loss_matrix.argmax(dim=0)
    candidate_stack = torch.stack(candidates, dim=0)
    batch_indices = torch.arange(labels.size(0), device=labels.device)
    selected = candidate_stack[selected_indices, batch_indices]

    pgd_losses = loss_matrix[0]
    selected_losses = loss_matrix[selected_indices, batch_indices]
    selected_bit = selected_indices.ne(0)
    stats = {
        'bitmax_selection_rate': selected_bit.float().mean().item(),
        'bitmax_loss_gain': (selected_losses - pgd_losses).mean().item(),
        'bitmax_delta_linf': (
            selected.sub(clean_images).abs().flatten(1).amax(1).mean().item()
        ),
    }
    return selected.detach(), stats
