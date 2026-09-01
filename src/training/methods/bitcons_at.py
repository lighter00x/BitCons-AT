import torch
import torch.nn.functional as F

from attacks import PGD, select_bitmax_candidate
from losses import (
    bitcons_discrepancy_weights,
    bitcons_js_per_sample,
    bitcons_risk_weights,
    get_risk_adaptive_bitcons_weight,
)
from .utils import (
    freeze_batchnorm_stats,
    get_classifier_parameters,
    loss_gradient_cosine,
)


def bitcons_at_train(
    config,
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    perturbation=None,
    epoch=0,
):
    """Train on the worst PGD/bit view plus risk-gated bit consistency."""
    model.train()
    eps = config.epsilon / 255
    pgd_attack = PGD(
        eps=eps,
        alpha=config.alpha / 255,
        steps=config.n_steps,
        loss_fn=criterion,
        model_eval=True,
    )
    planes = list(getattr(config, 'bitmax_planes', [0, 1, 2]))
    candidate_count = int(getattr(config, 'bitmax_candidates', 2))
    refine_steps = int(getattr(config, 'bitmax_refine_steps', 1))
    family_search = bool(getattr(config, 'bitmax_family_search', False))
    refine_best_only = bool(
        getattr(config, 'bitmax_refine_best_only', False)
    )
    return_best_bit = (
        getattr(config, 'bitmax_bit_view', 'selected') == 'best_bit'
    )
    risk_mode = getattr(config, 'bitcons_risk_mode', 'gain')
    gain_tau = float(getattr(config, 'bitcons_gain_tau', 0.5))
    discrepancy_tau = float(
        getattr(config, 'bitcons_discrepancy_tau', 0.01)
    )
    normalize_discrepancy_loss = bool(
        getattr(config, 'bitcons_normalize_discrepancy_loss', False)
    )
    margin_threshold = float(
        getattr(config, 'bitcons_margin_threshold', 0.0)
    )
    temperature = float(getattr(config, 'temperature', 1.0) or 1.0)
    consistency_enabled = bool(getattr(config, 'bitcons', True))
    conflict_mode = getattr(config, 'bitcons_conflict_mode', 'none')
    conflict_scale = float(getattr(config, 'bitcons_conflict_scale', 0.1))
    max_loss_ratio = float(
        getattr(config, 'bitcons_max_loss_ratio', 1.0)
    )
    classifier_parameters = (
        get_classifier_parameters(model)
        if consistency_enabled and conflict_mode != 'none'
        else None
    )
    consistency_weight = (
        get_risk_adaptive_bitcons_weight(config, epoch)
        if consistency_enabled
        else 0.0
    )

    totals = {
        'loss': 0.0,
        'robust': 0.0,
        'pgd_ce': 0.0,
        'bit_ce': 0.0,
        'consistency': 0.0,
        'js': 0.0,
        'weighted_consistency': 0.0,
        'selection_rate': 0.0,
        'loss_gain': 0.0,
        'positive_gain_rate': 0.0,
        'reliable_rate': 0.0,
        'gate_rate': 0.0,
        'risk_weight': 0.0,
        'margin': 0.0,
        'delta_linf': 0.0,
        'gradient_cosine': 0.0,
        'gradient_probe_count': 0.0,
        'conflict': 0.0,
        'aux_scale': 0.0,
        'loss_ratio': 0.0,
    }
    correct = 0
    sample_count = 0

    for images, labels in train_loader:
        clean_images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, clean_images, labels)
        bit_images, bitmax_stats, _ = select_bitmax_candidate(
            model,
            clean_images,
            adv_images,
            labels,
            planes,
            eps,
            candidate_count,
            refine_alpha=config.alpha / 255,
            refine_steps=refine_steps,
            return_details=True,
            family_search=family_search,
            refine_best_only=refine_best_only,
            return_best_bit=return_best_bit,
        )

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(bit_images, labels)
            perturbation.perturb(diff)

        logits_adv = model(adv_images)
        with freeze_batchnorm_stats(model):
            logits_bit = model(bit_images)

        pgd_losses = F.cross_entropy(logits_adv, labels, reduction='none')
        bit_losses = F.cross_entropy(logits_bit, labels, reduction='none')
        robust_losses = torch.maximum(pgd_losses, bit_losses)
        robust_loss = robust_losses.mean()

        js_losses = bitcons_js_per_sample(
            logits_bit,
            logits_adv,
            temperature=temperature,
            detach_reference=True,
        )
        gains = (bit_losses.detach() - pgd_losses.detach()).clamp_min(0.0)
        if risk_mode == 'discrepancy':
            risk_weights, margins, reliable = bitcons_discrepancy_weights(
                js_losses,
                logits_adv,
                labels,
                discrepancy_tau,
                margin_threshold,
            )
        else:
            risk_weights, gains, margins, reliable = bitcons_risk_weights(
                bit_losses,
                pgd_losses,
                logits_adv,
                labels,
                gain_tau,
                margin_threshold,
            )
        # Average over the full batch so rare/low-gain gates also reduce the
        # total auxiliary gradient, not just its sample composition.
        consistency_values = js_losses
        if risk_mode == 'discrepancy' and normalize_discrepancy_loss:
            consistency_values = (js_losses / discrepancy_tau).clamp(max=1.0)
        consistency_loss = (risk_weights * consistency_values).mean()
        raw_weighted_consistency = consistency_weight * consistency_loss
        gradient_cosine = 0.0
        conflict = False
        auxiliary_scale = 1.0
        if consistency_weight > 0 and consistency_loss.detach().item() > 0:
            if conflict_mode != 'none':
                gradient_cosine = loss_gradient_cosine(
                    robust_loss,
                    consistency_loss,
                    classifier_parameters,
                )
                conflict = gradient_cosine < 0.0
                if conflict and conflict_mode == 'suppress':
                    auxiliary_scale *= conflict_scale

            raw_aux_value = raw_weighted_consistency.detach().item()
            robust_value = robust_loss.detach().item()
            if raw_aux_value > 0:
                auxiliary_scale *= min(
                    1.0,
                    max_loss_ratio * robust_value / raw_aux_value,
                )
            totals['gradient_probe_count'] += 1.0

        weighted_consistency = (
            auxiliary_scale * raw_weighted_consistency
        )

        loss = robust_loss + weighted_consistency
        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if diff is not None:
            perturbation.restore(diff)

        use_bit_logits = bit_losses.detach().ge(pgd_losses.detach())[:, None]
        worst_logits = torch.where(use_bit_logits, logits_bit, logits_adv)
        correct += worst_logits.argmax(1).eq(labels).sum().item()
        sample_count += labels.size(0)

        totals['loss'] += loss.detach().item()
        totals['robust'] += robust_loss.detach().item()
        totals['pgd_ce'] += pgd_losses.detach().mean().item()
        totals['bit_ce'] += bit_losses.detach().mean().item()
        totals['consistency'] += consistency_loss.detach().item()
        totals['js'] += js_losses.detach().mean().item()
        totals['weighted_consistency'] += weighted_consistency.detach().item()
        totals['selection_rate'] += bitmax_stats['bitmax_selection_rate']
        totals['loss_gain'] += gains.mean().item()
        totals['positive_gain_rate'] += gains.gt(0).float().mean().item()
        totals['reliable_rate'] += reliable.float().mean().item()
        totals['gate_rate'] += risk_weights.gt(0).float().mean().item()
        totals['risk_weight'] += risk_weights.mean().item()
        totals['margin'] += margins.mean().item()
        totals['delta_linf'] += bitmax_stats['bitmax_delta_linf']
        totals['gradient_cosine'] += gradient_cosine
        totals['conflict'] += float(conflict)
        totals['aux_scale'] += auxiliary_scale
        totals['loss_ratio'] += (
            weighted_consistency.detach().item()
            / max(robust_loss.detach().item(), 1e-12)
        )

    batches = max(len(train_loader), 1)
    components = {
        'host_loss': totals['robust'] / batches,
        'pgd_ce_loss': totals['pgd_ce'] / batches,
        'bit_ce_loss': totals['bit_ce'] / batches,
        'bitcons_weight': consistency_weight,
        'bitcons_align_loss': totals['consistency'] / batches,
        'bitcons_js_loss': totals['js'] / batches,
        'bitcons_weighted_loss': totals['weighted_consistency'] / batches,
        'bitcons_gate_rate': totals['gate_rate'] / batches,
        'bitcons_risk_weight_mean': totals['risk_weight'] / batches,
        'bitcons_reliable_rate': totals['reliable_rate'] / batches,
        'bitcons_positive_gain_rate': totals['positive_gain_rate'] / batches,
        'bitcons_loss_gain': totals['loss_gain'] / batches,
        'bitcons_adv_margin': totals['margin'] / batches,
        'bitmax_selection_rate': totals['selection_rate'] / batches,
        'bitmax_delta_linf': totals['delta_linf'] / batches,
        'bitcons_gradient_cosine': (
            totals['gradient_cosine']
            / max(totals['gradient_probe_count'], 1.0)
        ),
        'bitcons_conflict_rate': (
            totals['conflict']
            / max(totals['gradient_probe_count'], 1.0)
        ),
        'bitcons_aux_scale': totals['aux_scale'] / batches,
        'bitcons_loss_ratio': totals['loss_ratio'] / batches,
    }
    return (
        totals['loss'] / batches,
        100.0 * correct / max(sample_count, 1),
        components,
    )
