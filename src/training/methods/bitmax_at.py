import torch

from attacks import PGD, select_bitmax_candidate


def bitmax_at_train(
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
    refine_steps = int(getattr(config, 'bitmax_refine_steps', 2))

    total_loss = 0.0
    correct = 0
    total = 0
    selection_rate_total = 0.0
    loss_gain_total = 0.0
    delta_linf_total = 0.0

    for images, labels in train_loader:
        clean_images, labels = images.to(device), labels.to(device)

        adv_images = pgd_attack(model, clean_images, labels)
        train_images, bitmax_stats = select_bitmax_candidate(
            model,
            clean_images,
            adv_images,
            labels,
            planes,
            eps,
            candidate_count,
            refine_alpha=config.alpha / 255,
            refine_steps=refine_steps,
        )

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(train_images, labels)
            perturbation.perturb(diff)

        logits = model(train_images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if diff is not None:
            perturbation.restore(diff)

        total_loss += loss.item()
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        selection_rate_total += bitmax_stats['bitmax_selection_rate']
        loss_gain_total += bitmax_stats['bitmax_loss_gain']
        delta_linf_total += bitmax_stats['bitmax_delta_linf']

    batches = len(train_loader)
    loss_components = {
        'host_loss': total_loss / batches,
        'bitmax_selection_rate': selection_rate_total / batches,
        'bitmax_loss_gain': loss_gain_total / batches,
        'bitmax_delta_linf': delta_linf_total / batches,
    }
    return total_loss / batches, 100.0 * correct / total, loss_components
