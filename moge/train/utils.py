from typing import *
import fnmatch

import sympy
import torch
import torch.nn as nn


def any_match(s: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)


def build_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    named_param_groups = [
        {
            k: p for k, p in model.named_parameters() if any_match(k, param_group_config['params']['include']) and not any_match(k, param_group_config['params'].get('exclude', []))
        } for param_group_config in optimizer_config['params']
    ]
    excluded_params = [k for k, p in model.named_parameters() if p.requires_grad and not any(k in named_params for named_params in named_param_groups)]
    assert len(excluded_params) == 0, f'The following parameters require grad but are excluded from the optimizer: {excluded_params}'
    optimizer_cls = getattr(torch.optim, optimizer_config['type'])
    optimizer = optimizer_cls([
        {
            **param_group_config,
            'params': list(params.values()), 
        } for param_group_config, params in zip(optimizer_config['params'], named_param_groups)
    ])
    return optimizer


def parse_lr_lambda(s: str) -> Callable[[int], float]:
    epoch = sympy.symbols('epoch')
    lr_lambda = sympy.sympify(s)
    return sympy.lambdify(epoch, lr_lambda, 'math')


def build_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    if scheduler_config['type'] == "SequentialLR":
        child_schedulers = [
            build_lr_scheduler(optimizer, child_scheduler_config)
                for child_scheduler_config in scheduler_config['params']['schedulers']
        ]
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=child_schedulers, milestones=scheduler_config['params']['milestones'])
    elif scheduler_config['type'] == "LambdaLR":
        lr_lambda = scheduler_config['params']['lr_lambda']
        if isinstance(lr_lambda, str):
            lr_lambda = parse_lr_lambda(lr_lambda)
        elif isinstance(lr_lambda, list):
            lr_lambda = [parse_lr_lambda(l) for l in lr_lambda]
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
    else:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config['type'])
        scheduler = scheduler_cls(optimizer, **scheduler_config.get('params', {}))
    return scheduler