import torch
from attacks import PGD, PGD_v2
from losses import consistency_loss, apply_bitplane_mask, bitcons_align_loss, get_bitcons_weight

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
    pgd_attack = PGD_v2(
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

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # images is a list [aug1, aug2] from the paired augmentation dataset
        images_aug1, images_aug2 = images[0].to(device), images[1].to(device)
        images_aug  = torch.cat([images_aug1, images_aug2], dim=0)
        labels_aug  = labels.repeat(2).to(device)

        with torch.no_grad():
            images_adv, logits_orig = pgd_attack(model, images_aug, labels_aug)

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels_aug)
            perturbation.perturb(diff)

        logits_adv = model(images_adv)
        logits_adv1, logits_adv2 = torch.chunk(logits_adv, 2, dim=0)

        loss_ce  = criterion(logits_adv, labels_aug)
        loss_con = consistency_loss(logits_adv1, logits_adv2, config.temperature)
        loss = loss_ce + config.lam * loss_con

        ### BitCons stream ###
        # Apply masking to aug1 (the "anchor" clean view).
        # Alignment target: mean of the two adversarial streams (detached).
        if use_bitcons:
            bc_alpha = get_bitcons_weight(config, epoch)
            if bc_alpha > 0:
                images_bc  = apply_bitplane_mask(images_aug1, bc_planes)
                logits_bc  = model(images_bc)
                loss_bc_ce    = bc_criterion(logits_bc, labels.to(device))

                loss_bc_align = bitcons_align_loss(
                    logits_bc, logits_orig, bc_align, bc_temp
                )
                loss = loss + bc_alpha * (loss_bc_ce + loss_bc_align)

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

    return total_loss / len(train_loader), 100.0 * correct / total
