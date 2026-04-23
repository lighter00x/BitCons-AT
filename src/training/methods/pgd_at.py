import torch
from attacks import PGD, PGD_v2
from losses import apply_bitplane_mask, bitcons_align_loss, get_bitcons_weight

bc_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.5)


def pgd_at_train(
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
        bc_planes   = list(getattr(config, 'bitcons_planes',  [0, 1, 2]))
        bc_align    =      getattr(config, 'bitcons_align',   'js')
        bc_temp     = float(getattr(config, 'temperature',     1.0) or 1.0)

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        benign_images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            adv_images, logits_orig = pgd_attack(model, benign_images, labels)

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(adv_images, labels)
            perturbation.perturb(diff)

        logits_adv = model(adv_images)
        loss = criterion(logits_adv, labels)

        ### BitCons stream ###
        if use_bitcons:
            bc_alpha = get_bitcons_weight(config, epoch)
            if bc_alpha > 0:
                B = benign_images.size(0)
                p_mix = torch.rand(1, device=device).item()  # scalar float in [0, 1)
                use_benign = torch.rand(B, device=device) < p_mix  # [B], bool
                use_benign = use_benign.view(B, 1, 1, 1)  # [B, 1, 1, 1]
                mixed_batch = torch.where(use_benign, benign_images, adv_images)
                images_bc  = apply_bitplane_mask(mixed_batch, bc_planes)
                logits_bc  = model(images_bc)
                loss_bc_ce    = bc_criterion(logits_bc, labels)
                loss_bc_align = bitcons_align_loss(
                    logits_bc, logits_orig, bc_align, bc_temp
                )
                # loss = loss + bc_alpha * (loss_bc_ce + loss_bc_align)
                loss = loss + bc_alpha * (loss_bc_align)
                

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if diff is not None:
            perturbation.restore(diff)

        total_loss += loss.item()
        _, predicted = logits_adv.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total
