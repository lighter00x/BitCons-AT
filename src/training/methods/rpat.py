import torch
import torch.nn as nn
from attacks import PGD, PGD_v2
from losses import apply_bitplane_mask, bitcons_align_loss, get_bitcons_weight

criterion_ra = nn.MSELoss()
bc_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.5)


def rpat_train(
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
        benign_images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            images_adv, logits_orig = pgd_attack(model, benign_images, labels)

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels)
            perturbation.perturb(diff)

        adv_outputs = model(images_adv)
        loss = criterion(adv_outputs, labels)

        ### Robust Perception loss ###
        if epoch >= config.RA_start:
            ip_rate = config.RA_ip_rate
            interp_images    = ip_rate * benign_images + (1 - ip_rate) * images_adv
            interp_output_1  = model(interp_images)
            benign_outputs   = model(benign_images)
            interp_output_2  = ip_rate * benign_outputs + (1 - ip_rate) * adv_outputs
            loss_ra = criterion_ra(interp_output_1, interp_output_2)
            loss = loss + config.lam * loss_ra

        ### BitCons stream ###
        if use_bitcons:
            bc_alpha = get_bitcons_weight(config, epoch)
            if bc_alpha > 0:
                images_bc  = apply_bitplane_mask(benign_images, bc_planes)
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
        _, predicted = adv_outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total
