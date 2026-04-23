import torch
import torch.nn as nn
from losses import apply_bitplane_mask


def evaluate_natural_masked(model, device, test_loader, planes):
    """
    评估模型在自然样本上的准确率

    Args:
        model: 模型
        device: 计算设备
        test_loader: 测试数据加载器

    Returns:
        accuracy: 准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images_masked = apply_bitplane_mask(images, planes)
            logits = model(images_masked)
            _, predicted = logits.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy

def evaluate_natural(model, device, test_loader):
    """
    评估模型在自然样本上的准确率

    Args:
        model: 模型
        device: 计算设备
        test_loader: 测试数据加载器

    Returns:
        accuracy: 准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = logits.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy
