import torch
from attacks import PGD
from losses import consistency_loss, apply_bitplane_mask, bitcons_align_loss, get_bitcons_weight
from .utils import LossComponentMeter, freeze_batchnorm_stats

bc_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.5)

def cons_at_train(
    config,
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    perturbation=None,
    epoch=0,
):
    model.train()
    pgd_attack = PGD(
        eps=config.epsilon / 255,
        alpha=config.alpha / 255,
        steps=config.n_steps,
        loss_fn=criterion,
    )

    use_bitcons = bool(getattr(config, 'bitcons', False))
    if use_bitcons:
        bc_planes = list(getattr(config, 'bitcons_planes', [0, 1, 2]))
        bc_align  =      getattr(config, 'bitcons_align',  'js')
        bc_temp   = float(getattr(config, 'temperature',    1.0) or 1.0)
        bc_ce_weight_value = getattr(config, 'bitcons_ce_weight', None)
        bc_align_weight_value = getattr(config, 'bitcons_align_weight', None)
        bc_ce_weight = float(1.0 if bc_ce_weight_value is None else bc_ce_weight_value)
        bc_align_weight = float(1.0 if bc_align_weight_value is None else bc_align_weight_value)

    total_loss = 0
    correct = 0
    total = 0
    component_meter = LossComponentMeter()

    for images, labels in train_loader:
        # images is a list [aug1, aug2] from the paired augmentation dataset
        images_aug1, images_aug2 = images[0].to(device), images[1].to(device)
        images_aug  = torch.cat([images_aug1, images_aug2], dim=0)
        labels_aug  = labels.repeat(2).to(device)

        with torch.no_grad():
            images_adv = pgd_attack(model, images_aug, labels_aug)

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels_aug)
            perturbation.perturb(diff)

        logits_adv = model(images_adv)
        logits_adv1, logits_adv2 = torch.chunk(logits_adv, 2, dim=0)

        loss_ce  = criterion(logits_adv, labels_aug)
        loss_con = consistency_loss(logits_adv1, logits_adv2, config.temperature)
        host_loss = loss_ce + config.lam * loss_con
        loss = host_loss
        loss_bc_ce = None
        loss_bc_align = None
        loss_bc_weighted = None
        effective_bc_alpha = 0.0

        ### BitCons stream ###
        # Apply masking to the final adversarial anchor view.
        if use_bitcons:
            bc_alpha = get_bitcons_weight(config, epoch)
            if bc_alpha > 0:
                effective_bc_alpha = bc_alpha
                images_adv1, _ = torch.chunk(images_adv, 2, dim=0)
                images_bc = apply_bitplane_mask(images_adv1, bc_planes)
                with freeze_batchnorm_stats(model):
                    logits_bc = model(images_bc)
                loss_bc_ce    = bc_criterion(logits_bc, labels.to(device))

                loss_bc_align = bitcons_align_loss(
                    logits_bc, logits_adv1.detach(), bc_align, bc_temp
                )
                loss_bc_total = (
                    bc_ce_weight * loss_bc_ce
                    + bc_align_weight * loss_bc_align
                )
                loss_bc_weighted = bc_alpha * loss_bc_total
                loss = loss + loss_bc_weighted

        component_meter.update(
            host_loss,
            effective_bc_alpha,
            loss_bc_ce,
            loss_bc_align,
            None,
            loss_bc_weighted,
        )
        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if diff is not None:
            perturbation.restore(diff)

        total_loss += loss.item()
        _, predicted = logits_adv.max(1)
        correct += predicted.eq(labels_aug).sum().item()
        total += labels_aug.size(0)

    return (
        total_loss / len(train_loader),
        100.0 * correct / total,
        component_meter.averages(),
    )
