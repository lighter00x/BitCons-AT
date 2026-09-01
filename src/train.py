import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import time

# Ensure src is in path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from common.config import Config
from common.args import get_args
from common.utils import set_seed, get_optimizer, get_scheduler
from datasets import get_loaders
from models import get_model
from training.methods import get_train_fn
from training.perturbations import get_perturbation
from losses import get_criterion
from evals import evaluate_natural, evaluate_pgd_10, evaluate_pgd_20, evaluate_pgd_10_masked, evaluate_natural_masked
from evals.pgd import evaluate_pgd_10_classwise
from utils import Logger, save_checkpoint


def main():
    args = get_args()
    config = Config()
    config.load_from_args(args)
    print("Configuration:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    # time.sleep(5)  
    set_seed(config.seed)

    project_root = Path(__file__).parent.parent
    data_dir = Path(config.data_dir).expanduser() if getattr(config, "data_dir", None) else (project_root / "dataset")
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    if not data_dir.exists():
        fallback_data_dirs = [
            project_root / "dataset" / "data",
            project_root / "dataset",
        ]
        for fallback_data_dir in fallback_data_dirs:
            if fallback_data_dir.exists():
                print(f"[WARN] data_dir not found: {data_dir}")
                print(f"[WARN] Falling back to: {fallback_data_dir}")
                data_dir = fallback_data_dir
                break
    config["data_dir"] = str(data_dir)

    if config.device == "cuda" and torch.cuda.is_available():
        if config.gpu_id is not None:
            device = torch.device(f"cuda:{config.gpu_id}")
            torch.cuda.set_device(config.gpu_id)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 生成实验名称和文件夹结构
    # 主文件夹格式: {dataset}_{model}_{method}_{perturbation}
    # 子文件夹格式: seed_{seed}_{timestamp}
    bitcons_enabled = bool(getattr(config, 'bitcons', False))
    contrast_enabled = bitcons_enabled and bool(getattr(config, 'bitcons_contrast', False))
    experiment_parts = [
        config.dataset,
        config.model,
        config.method,
        config.perturbation,
        f"bitcons_{str(bitcons_enabled).lower()}",
        f"contrast_{str(contrast_enabled).lower()}",
    ]
    if args.desc:
        experiment_parts.append(args.desc)
    main_exp_dir = "_".join(experiment_parts)
    if config.exp_name is not None:
        sub_exp_dir = f"seed_{config.seed}_{config.exp_name}"
    else:
        sub_exp_dir = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if config.exp_name is None:
        config.exp_name = sub_exp_dir

    # 确保 out_dir 相对于项目根目录（FAWP 文件夹），而不是脚本所在的 src 文件夹
    # 如果 out_dir 是相对路径，将其转换为相对于项目根目录的绝对路径
    out_dir = Path(config.out_dir)
    if not out_dir.is_absolute():
        # 获取项目根目录（src 的父目录）
        out_dir = project_root / out_dir

    # 创建 Logger 时传入主文件夹和子文件夹
    logger = Logger(str(out_dir), main_exp_dir, sub_exp_dir)
    logger.save_config(config.to_dict())
    checkpoint_dir = logger.get_checkpoint_dir()

    train_loader, test_loader = get_loaders(config)
    model = get_model(config).to(device)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    criterion = get_criterion(config)

    if config.perturbation and config.perturbation != "none":
        perturbation = get_perturbation(config, model, optimizer, device)
    else:
        perturbation = None

    train_fn = get_train_fn(config)

    start_epoch = 0
    best_robust_acc = float('-inf')

    if config.resume:
        from utils import load_checkpoint
        # 会记录保存的时候的训练epoch，可恢复到之前的训练状态（包括模型权重、优化器状态、学习率调度器状态等）
        start_epoch, best_robust_acc = load_checkpoint(
            model, optimizer, config.resume, scheduler=scheduler
        )

    print(f"Training {config.method} on {config.dataset} with {config.model}")
    print(f"Perturbation: {config.perturbation}")
    print(f"Device: {device}")
    print(f"Experiment: {config.exp_name}")

    # 记录训练开始时间
    training_start_time = time.time()

    for epoch in range(start_epoch, config.epochs):

        train_loss, train_acc, loss_components = train_fn(
            config,
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            perturbation,
            epoch,
        )

        scheduler.step()

        test_acc = evaluate_natural(model, device, test_loader) # 自然样本上的原始准确率
        pgd10_acc = evaluate_pgd_10(model, device, test_loader) # 自然样本 + pgd10 准确率

        pgd10_masked_acc = None
        natural_masked_acc = None
        if (
            getattr(config, 'bitcons', False)
            and getattr(config, 'method', None) != 'bitcons_at'
        ):
            bc_planes = list(getattr(config, 'bitcons_planes', [0, 1, 2]))
            pgd10_masked_acc = evaluate_pgd_10_masked(model, device, test_loader, bc_planes)    # 在pgd10 + bitcons下的准确率
            natural_masked_acc = evaluate_natural_masked(model, device, test_loader, bc_planes) # 直接是干净样本 + bitcons下的准确率

        logger.log_metrics(epoch, train_loss, train_acc, test_acc, pgd10_acc,
                           pgd10_masked_acc=pgd10_masked_acc, natural_masked_acc=natural_masked_acc)
        logger.log_loss_components(epoch, loss_components)
        # Base and BitCons runs must use the same checkpoint-selection metric.
        # Masked accuracy is a mechanism diagnostic, not the threat-model score.
        robust_acc = pgd10_acc
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_robust_acc,
                checkpoint_dir,
                scheduler=scheduler,
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            masked_str = f" | PGD-10 Masked Acc: {pgd10_masked_acc:.2f}%" if pgd10_masked_acc is not None else ""
            natural_masked_str = f" | Natural Masked Acc: {natural_masked_acc:.2f}%" if natural_masked_acc is not None else ""
            print(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}% | PGD-10 Acc: {pgd10_acc:.2f}%"
                + masked_str
                + natural_masked_str
            )

    save_checkpoint(
        model,
        optimizer,
        epoch,
        best_robust_acc,
        checkpoint_dir,
        filename="final_model.pt",
        scheduler=scheduler,
    )

    # 计算总训练时间
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    print(f"\nTraining completed.")
    print(f"  Best PGD-10 Accuracy:  {best_robust_acc:.2f}%")
    print(
        f"Total training time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)"
    )

    # 训练完成后生成最终报告和图表（传入总训练时间）
    logger.finalize(total_training_time=total_training_time)


if __name__ == "__main__":
    main()
