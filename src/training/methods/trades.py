import torch
from attacks import generate_trades
from losses import (
    trades_loss,
    apply_bitplane_mask,
    apply_unreliable_bitplane_mask,
    bitcons_align_loss,
    bitcons_feature_contrastive_loss,
    get_bitcons_weight,
)

bc_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.5)


def trades_train(
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

    use_bitcons = bool(getattr(config, 'bitcons', False))
    if use_bitcons:
        bc_planes    = list(getattr(config, 'bitcons_planes',        [0, 1, 2]))
        bc_align     =      getattr(config, 'bitcons_align',         'js')
        bc_temp      = float(getattr(config, 'temperature',           1.0) or 1.0)
        use_contrast = bool(getattr(config, 'bitcons_contrast',       False))
        bc_ctr_lam   = float(getattr(config, 'bitcons_contrast_lam',  1.0) or 1.0)
        bc_ctr_temp  = float(getattr(config, 'bitcons_contrast_temp', 0.5) or 0.5)
    else:
        use_contrast = False

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            images_adv = generate_trades(
                model,
                images,
                labels,
                eps=config.epsilon / 255,
                alpha=config.alpha / 255,
                steps=config.n_steps,
            )

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels)
            perturbation.perturb(diff)

        if use_contrast:
            logits_clean, feat_clean = model(images, return_feat=True)
        else:
            logits_clean = model(images)
        logits_adv = model(images_adv)
        loss = trades_loss(logits_clean, logits_adv, labels, config.beta)

        ### BitCons stream ###
        if use_bitcons:
            bc_alpha = get_bitcons_weight(config, epoch)
            if bc_alpha > 0:
                images_bc = apply_bitplane_mask(images, bc_planes)

                if use_contrast:
                    logits_bc, feat_bc = model(images_bc, return_feat=True)
                    images_ub          = apply_unreliable_bitplane_mask(images, bc_planes)
                    logits_ub, feat_ub = model(images_ub, return_feat=True)
                    loss_bc_contrast = bitcons_feature_contrastive_loss(
                        feat_bc, feat_clean.detach(), feat_ub, bc_ctr_temp
                    )
                else:
                    logits_bc = model(images_bc)

                loss_bc_ce    = bc_criterion(logits_bc, labels)
                loss_bc_align = bitcons_align_loss(
                    logits_bc, logits_clean.detach(), bc_align, bc_temp
                )
                # loss = loss + bc_alpha * (loss_bc_ce + loss_bc_align)
                loss = loss + bc_alpha * (loss_bc_align)

                if use_contrast:
                    loss = loss + bc_alpha * bc_ctr_lam * loss_bc_contrast
                

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if diff is not None:
            perturbation.restore(diff)

        total_loss += loss.item()
        _, predicted = logits_clean.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total