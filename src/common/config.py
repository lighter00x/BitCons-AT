import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy


class Config:
    def __init__(self):
        self.data = {}

    def load_yaml(self, path: str) -> None:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # 处理科学记数法和类型转换
        for key, value in config.items():
            if isinstance(value, str):
                # 尝试转换为float或int
                try:
                    # 先尝试float（支持科学记数法如5e-4）
                    if 'e' in value.lower() or '.' in value:
                        value = float(value)
                    else:
                        # 尝试int
                        value = int(value)
                except (ValueError, AttributeError):
                    # 保持原值
                    pass
            config[key] = value

        # 处理权重扰动子配置
        # 如果 perturbation 被指定（不是 'none'），自动加载对应的子配置
        if 'perturbation' in config and config['perturbation'] != 'none':
            perturb_type = config['perturbation']
            if perturb_type in config and isinstance(config[perturb_type], dict):
                perturb_config = config[perturb_type]
                # 将子配置的参数添加到主配置中（带前缀）
                for sub_key, sub_value in perturb_config.items():
                    config[f"{perturb_type}_{sub_key}"] = sub_value

        self.data.update(config)

    def load_from_args(self, args: argparse.Namespace) -> None:
        # 获取项目根目录（src的父目录）
        src_dir = Path(__file__).parent.parent
        project_dir = src_dir.parent

        dataset_config = project_dir / f'configs/datasets/{args.dataset}.yaml'
        if not dataset_config.exists():
            raise FileNotFoundError(f'Dataset config not found: {dataset_config}')
        self.load_yaml(str(dataset_config))

        # 模型配置通过 get_model() 函数直接处理，无需加载 YAML 文件
        # 只需通过 args.model 指定模型名称

        train_config = project_dir / f'configs/training/{args.config}.yaml'
        if not train_config.exists():
            raise FileNotFoundError(f'Training config not found: {train_config}')
        self.load_yaml(str(train_config))

        self._apply_overrides(args)
        self._normalize_and_validate(args)

    def _apply_overrides(self, args: argparse.Namespace) -> None:
        # 先添加必要的参数（dataset, model, config）
        if args.dataset:
            self.data['dataset'] = args.dataset
        if args.model:
            self.data['model'] = args.model
        if args.config:
            self.data['config'] = args.config

        # 然后添加其他非None的参数
        for key, value in vars(args).items():
            if value is None or key in ['dataset', 'model', 'config']:
                continue
            self.data[key] = value

        for perturbation in ('awp', 'rwp'):
            nested = deepcopy(self.data.get(perturbation, {}))
            for field in ('gamma', 'warmup'):
                override = self.data.get(f'{perturbation}_{field}')
                if override is not None:
                    nested[field] = override
            if nested:
                self.data[perturbation] = nested

    def _normalize_and_validate(self, args: argparse.Namespace) -> None:
        bitcons = bool(self.data.get('bitcons', False))
        contrast = bool(self.data.get('bitcons_contrast', False))

        if not bitcons:
            if getattr(args, 'bitcons_contrast', None) is True:
                raise ValueError('BitCons contrast requires --bitcons')
            self.data['bitcons_contrast'] = False
            contrast = False
        elif contrast:
            self.data['bitcons_contrast'] = True

        if contrast and self.data.get('method') == 'cons_at':
            raise ValueError('BitCons contrast is not implemented for cons_at')
        if contrast and self.data.get('method') == 'bitcons_at':
            raise ValueError(
                'Legacy BitCons contrast is not supported by bitcons_at'
            )

        planes = self.data.get('bitcons_planes', [])
        if any(plane < 0 or plane > 7 for plane in planes):
            raise ValueError('bitcons_planes values must be between 0 and 7')

        align_type = self.data.get('bitcons_align', 'js')
        if align_type not in ('js', 'kl', 'mse', 'kl_zscore'):
            raise ValueError(f'Unknown bitcons_align: {align_type}')

        warmup_schedule = self.data.get('bitcons_warmup_schedule', 'linear')
        if warmup_schedule not in ('linear', 'cosine'):
            raise ValueError(
                f'Unknown bitcons_warmup_schedule: {warmup_schedule}'
            )

        if self.data.get('bitcons_alpha', 0) < 0:
            raise ValueError('bitcons_alpha must be non-negative')
        if self.data.get('bitcons_ce_weight', 0) < 0:
            raise ValueError('bitcons_ce_weight must be non-negative')
        if self.data.get('bitcons_align_weight', 0) < 0:
            raise ValueError('bitcons_align_weight must be non-negative')
        if self.data.get('bitcons_warmup', 0) < 0:
            raise ValueError('bitcons_warmup must be non-negative')
        if self.data.get('bitcons_start_epoch', 0) < 0:
            raise ValueError('bitcons_start_epoch must be non-negative')
        if self.data.get('bitcons_conflict_mode', 'none') not in (
            'none', 'monitor', 'suppress'
        ):
            raise ValueError('Unknown bitcons_conflict_mode')
        conflict_scale = self.data.get('bitcons_conflict_scale', 0.1)
        if conflict_scale < 0 or conflict_scale > 1:
            raise ValueError('bitcons_conflict_scale must be between 0 and 1')
        if self.data.get('bitcons_max_loss_ratio', 1.0) <= 0:
            raise ValueError('bitcons_max_loss_ratio must be positive')
        if self.data.get('bitcons_contrast_lam', 0) < 0:
            raise ValueError('bitcons_contrast_lam must be non-negative')
        if self.data.get('bitcons_contrast_temp', 1) <= 0:
            raise ValueError('bitcons_contrast_temp must be positive')
        if self.data.get('temperature', 1) <= 0:
            raise ValueError('temperature must be positive')

        if self.data.get('method') in ('bitmax_at', 'bitcons_at'):
            bitmax_planes = sorted(set(self.data.get('bitmax_planes', [])))
            if (
                not bitmax_planes
                or bitmax_planes != list(range(bitmax_planes[-1] + 1))
            ):
                raise ValueError(
                    'bitmax_planes must be contiguous low bits starting at 0'
                )
            if self.data.get('bitmax_candidates', 0) <= 0:
                raise ValueError('bitmax_candidates must be positive')
            if self.data.get('bitmax_refine_steps', -1) < 0:
                raise ValueError('bitmax_refine_steps must be non-negative')
            if self.data.get('bitmax_bit_view', 'selected') not in (
                'selected', 'best_bit'
            ):
                raise ValueError('Unknown bitmax_bit_view')

        if self.data.get('method') == 'bitcons_at':
            if self.data.get('bitcons_gain_tau', 0) <= 0:
                raise ValueError('bitcons_gain_tau must be positive')
            risk_mode = self.data.get('bitcons_risk_mode', 'gain')
            if risk_mode not in ('gain', 'discrepancy'):
                raise ValueError('Unknown bitcons_risk_mode')
            if self.data.get('bitcons_discrepancy_tau', 0.01) <= 0:
                raise ValueError('bitcons_discrepancy_tau must be positive')

        if self.data.get('method') == 'bitplane_at':
            bitplane_planes = sorted(set(self.data.get('bitplane_planes', [])))
            if (
                not bitplane_planes
                or bitplane_planes != list(range(bitplane_planes[-1] + 1))
            ):
                raise ValueError(
                    'bitplane_planes must be contiguous low bits starting at 0'
                )

        for key in ('epochs', 'batch_size', 'n_steps'):
            if self.data.get(key, 0) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('lr_init', 'epsilon', 'alpha'):
            if self.data.get(key, 0) < 0:
                raise ValueError(f'{key} must be non-negative')

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            return super().__getattribute__(key)
        return self.data.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self.data)

    def __repr__(self) -> str:
        return f"Config({self.data})"
