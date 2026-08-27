import torch
from tqdm import tqdm
from losses import apply_bitplane_mask


def evaluate_aa(model, device, test_loader, norm='Linf', eps=8/255, verbose=False):
    """
    使用AutoAttack评估模型的对抗鲁棒性

    标准 AutoAttack 组合包含 APGD-CE、APGD-DLR、FAB 和 Square Attack。

    Args:
        model: 要评估的模型
        device: 计算设备
        test_loader: 测试数据加载器
        norm: 范数类型，'Linf' (L-infinity), 'L2', 'L1'
        eps: 扰动限制
        verbose: 是否打印详细信息

    Returns:
        对抗精度（百分比）
    """
    try:
        import autoattack
    except ImportError:
        raise ImportError(
            "AutoAttack not installed. Install with:\n"
            "  pip install git+https://github.com/fra31/auto-attack.git"
        )

    model.eval()

    # 收集所有图像和标签
    all_images = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())

    x_test = torch.cat(all_images).to(device)
    y_test = torch.cat(all_labels).to(device)

    if verbose:
        print(f"Running AutoAttack evaluation...")
        print(f"  Norm: {norm}")
        print(f"  Epsilon: {eps:.4f}")
        print(f"  Total samples: {len(x_test)}")

    # 创建 AutoAttack 对象
    adversary = autoattack.AutoAttack(
        model,
        norm=norm,
        eps=eps,
        version='standard'  # 标准版本（推荐用于评估）
    )

    # 运行攻击
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)

    # 计算鲁棒精度
    with torch.no_grad():
        logits = model(x_adv)
        _, predicted = logits.max(1)
        correct = predicted.eq(y_test).sum().item()
        total = len(y_test)

    robust_accuracy = 100.0 * correct / total

    if verbose:
        print(f"AutoAttack Robust Accuracy: {robust_accuracy:.2f}%")

    return robust_accuracy


def evaluate_aa_masked(model, device, test_loader, planes, norm='Linf', eps=8/255, verbose=False):
    """Generate adversarial examples against original model, apply bit-plane mask, then evaluate."""
    try:
        import autoattack
    except ImportError:
        raise ImportError(
            "AutoAttack not installed. Install with:\n"
            "  pip install git+https://github.com/fra31/auto-attack.git"
        )

    model.eval()

    all_images = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())

    x_test = torch.cat(all_images).to(device)
    y_test = torch.cat(all_labels).to(device)

    adversary = autoattack.AutoAttack(model, norm=norm, eps=eps, version='standard')
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)

    x_masked = apply_bitplane_mask(x_adv, planes)

    with torch.no_grad():
        logits = model(x_masked)
        _, predicted = logits.max(1)
        correct = predicted.eq(y_test).sum().item()

    robust_accuracy = 100.0 * correct / len(y_test)

    if verbose:
        print(f"AutoAttack (masked) Robust Accuracy: {robust_accuracy:.2f}%")

    return robust_accuracy
