import argparse


def get_args():
    parser = argparse.ArgumentParser(description='BitCons Adversarial Training')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinynet'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Dataset root directory (overrides dataset YAML)')
    parser.add_argument('--model', type=str, default='preactresnet18',
                        choices=['resnet18', 'wrn28_10', 'wrn34_10', 'preactresnet18'],
                        help='Model architecture')
    parser.add_argument('--config', type=str, default='pgd_at',
                        help='Training config name (in configs/training/)')
    parser.add_argument('--desc', type=str, default=None,
                        help='Description for the experiment')
    
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr_init', type=float, default=None)
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=['multi_step', 'cosine'])
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['sgd', 'adam'])
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)

    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)

    parser.add_argument('--perturbation', type=str, default=None,
                        choices=['none', 'awp', 'rwp'],
                        help='Weight perturbation method')

    parser.add_argument('--awp_gamma', type=float, default=None)
    parser.add_argument('--awp_warmup', type=int, default=None)
    parser.add_argument('--awp_lr', type=float, default=None)
    parser.add_argument('--rwp_gamma', type=float, default=None)
    parser.add_argument('--rwp_warmup', type=int, default=None)
    parser.add_argument('--rwp_lr', type=float, default=None)

    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)

    # ── BitCons: Fragile Bit-Plane Masking Stream ──────────────────────────
    parser.add_argument('--bitcons', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Enable or disable the BitCons training stream')
    parser.add_argument('--bitcons_contrast', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Enable or disable BitCons feature contrastive loss')
    parser.add_argument('--bitcons_planes', nargs='+', type=int, default=None,
                        metavar='P',
                        help='Bit-plane indices to mask (0=LSB … 7=MSB). '
                             'E.g. --bitcons_planes 0 1 2')
    parser.add_argument('--bitcons_align', type=str, default=None,
                        choices=['js', 'kl', 'mse', 'kl_zscore'],
                        help='Alignment loss type between BitCons and main stream')
    parser.add_argument('--bitcons_alpha', type=float, default=None,
                        help='Final weight α for the BitCons alignment loss')
    parser.add_argument('--bitcons_ce_weight', type=float, default=None,
                        help='Masked-view classification loss weight inside BitCons')
    parser.add_argument('--bitcons_align_weight', type=float, default=None,
                        help='Logit alignment loss weight inside BitCons')
    parser.add_argument('--bitcons_warmup', type=int, default=None,
                        help='Epochs to warm up α from 0 → bitcons_alpha')
    parser.add_argument('--bitcons_warmup_schedule', type=str, default=None,
                        choices=['linear', 'cosine'],
                        help='Warmup schedule shape (default: linear)')
    parser.add_argument('--bitcons_contrast_lam', type=float, default=None,
                        help='Feature contrastive loss weight relative to BitCons alpha')
    parser.add_argument('--bitcons_contrast_temp', type=float, default=None,
                        help='Feature contrastive temperature')
    # ──────────────────────────────────────────────────────────────────────

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU ID to use (default: auto-select)')
    parser.add_argument('--out_dir', type=str, default='outputs/',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')

    return parser.parse_args()
