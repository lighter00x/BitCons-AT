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
    parser.add_argument('--bitcons_start_epoch', type=int, default=None,
                        help='Epoch that starts risk-adaptive BitCons consistency')
    parser.add_argument('--bitcons_gain_tau', type=float, default=None,
                        help='Loss-gain scale used by risk-adaptive BitCons')
    parser.add_argument('--bitcons_risk_mode', type=str, default=None,
                        choices=['gain', 'discrepancy'],
                        help='Signal used to gate risk-adaptive BitCons')
    parser.add_argument('--bitcons_discrepancy_tau', type=float, default=None,
                        help='JS scale used by discrepancy-gated BitCons')
    parser.add_argument('--bitcons_normalize_discrepancy_loss',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='Normalize and cap discrepancy consistency by tau')
    parser.add_argument('--bitcons_conflict_mode', type=str, default=None,
                        choices=['none', 'monitor', 'suppress'],
                        help='Classifier-gradient conflict handling mode')
    parser.add_argument('--bitcons_conflict_scale', type=float, default=None,
                        help='Auxiliary scale for conflicting batches')
    parser.add_argument('--bitcons_max_loss_ratio', type=float, default=None,
                        help='Maximum weighted BitCons/robust loss ratio')
    parser.add_argument('--bitcons_margin_threshold', type=float, default=None,
                        help='Minimum adversarial true-class margin for alignment')
    parser.add_argument('--bitcons_contrast_lam', type=float, default=None,
                        help='Feature contrastive loss weight relative to BitCons alpha')
    parser.add_argument('--bitcons_contrast_temp', type=float, default=None,
                        help='Feature contrastive temperature')
    # ──────────────────────────────────────────────────────────────────────

    parser.add_argument('--bitmax_planes', nargs='+', type=int, default=None,
                        metavar='P',
                        help='Contiguous low bit-planes explored by BitMax')
    parser.add_argument('--bitmax_candidates', type=int, default=None,
                        help='Number of projected low-bit candidates per sample')
    parser.add_argument('--bitmax_refine_steps', type=int, default=None,
                        help='PGD refinement steps after each discrete bit jump')
    parser.add_argument('--bitmax_family_search',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='Search P0/P01/... low-bit candidate families')
    parser.add_argument('--bitmax_refine_best_only',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='Refine only the per-sample strongest bit seed')
    parser.add_argument('--bitmax_bit_view', type=str, default=None,
                        choices=['selected', 'best_bit'],
                        help='Return PGD-inclusive winner or strongest bit view')
    parser.add_argument('--bitplane_planes', nargs='+', type=int, default=None,
                        metavar='P',
                        help='Low bit-planes masked by the BPDA model wrapper')

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
