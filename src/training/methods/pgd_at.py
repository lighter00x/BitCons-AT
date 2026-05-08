import torch
from attacks import PGD, PGD_v2
from losses import (
    apply_bitplane_mask,
    apply_unreliable_bitplane_mask,
    bitcons_align_loss,
    bitcons_feature_contrastive_loss,
    get_bitcons_weight,
)

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
        bc_planes    = list(getattr(config, 'bitcons_planes',        [0, 1, 2]))
        bc_align     =      getattr(config, 'bitcons_align',         'js')  # 散度
        bc_temp      = float(getattr(config, 'temperature',           1.0) or 1.0)  # 对齐损失的温度系数
        use_contrast = bool(getattr(config, 'bitcons_contrast',       False))   # 是否额外启用特征级的对比损失
        bc_ctr_lam   = float(getattr(config, 'bitcons_contrast_lam',  1.0) or 1.0)
        bc_ctr_temp  = float(getattr(config, 'bitcons_contrast_temp', 0.5) or 0.5)
    else:
        use_contrast = False

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        benign_images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            # 扰动图像 + 模型对初始扰动样本的输出 logit
            adv_images, logits_orig = pgd_attack(model, benign_images, labels)

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(adv_images, labels)
            perturbation.perturb(diff)

        if use_contrast:
            # logits是最后对img_adv进行classify的分数， feat_adv是输出logits之前展平的feature tensor
            logits_adv, feat_adv = model(adv_images, return_feat=True)
        else:
            logits_adv = model(adv_images)
        # 1. 基础对抗训练损失
        loss = criterion(logits_adv, labels)

        ### BitCons stream ###
        if use_bitcons:
            bc_alpha = get_bitcons_weight(config, epoch)
            if bc_alpha > 0:
                B = benign_images.size(0)
                p_mix = torch.rand(1, device=device).item()  # scalar float in [0, 1)
                use_benign = torch.rand(B, device=device) < p_mix  # [B], bool 掩码：判断每张图是否使用干净样本
                use_benign = use_benign.view(B, 1, 1, 1)  # [B, 1, 1, 1]
                mixed_batch = torch.where(use_benign, benign_images, adv_images)
                images_bc = apply_bitplane_mask(mixed_batch, bc_planes)

                if use_contrast:
                    logits_bc, feat_bc = model(images_bc, return_feat=True)
                    images_ub          = apply_unreliable_bitplane_mask(mixed_batch, bc_planes)
                    logits_ub, feat_ub = model(images_ub, return_feat=True)
                    # 对比学习
                    # feat_bc:  高位特征（最后logits之前的）
                    # feat_adv: 对adv样本的特征
                    # feat_ub:  低位特征
                    # 3. BitCons 特征对比损失
                    loss_bc_contrast = bitcons_feature_contrastive_loss(
                        feat_bc, feat_adv.detach(), feat_ub, bc_ctr_temp
                    )   
                else:
                    logits_bc = model(images_bc)

                loss_bc_ce    = bc_criterion(logits_bc, labels)
                # 2. BitCons 对齐损失
                # 它衡量掩码后的高位图像输出 (logits_bc) 与初始模型输出 (logits_orig) 之间的分布对齐程度（如 JS散度或 KL散度）
                loss_bc_align = bitcons_align_loss(
                    logits_bc, logits_orig, bc_align, bc_temp
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
        _, predicted = logits_adv.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total
