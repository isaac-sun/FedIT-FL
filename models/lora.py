"""Minimal LoRA (Hu et al., 2021) for a frozen Hugging Face causal LM.

Each targeted ``nn.Linear`` becomes ``LoRALinear``:

    y = W x + (alpha / r) * B (A dropout(x)),   A: r x in,  B: out x r

``A`` starts Kaiming-uniform (seeded), ``B`` at zero, so the adapted model
starts equal to the base model.  Only ``lora_A`` / ``lora_B`` are trained,
uploaded and averaged; the state dict used everywhere in this project is the
ordered dict of those tensors (fp32, CPU), so FedAvg, the attacks and the
Shapley coalitions work on it exactly as on the full model in ECL-FL.

FedAvg averages ``A`` and ``B`` separately (as FedIT / OpenFedLLM do); the
product of the averages is not the average of the products, and the Shapley
coalitions use the same rule, so both see the same aggregated model.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        bound = 1.0 / math.sqrt(base.in_features)   # kaiming_uniform_(a=sqrt(5))
        with torch.no_grad():
            self.lora_A.uniform_(-bound, bound, generator=generator)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        result = self.base(x)
        x = self.dropout(x)
        if not _autocast_on(x):   # under autocast the matmuls pick the dtype
            x = x.to(self.lora_A.dtype)
        update = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return result + update.to(result.dtype)


def _autocast_on(x: torch.Tensor) -> bool:
    try:
        return torch.is_autocast_enabled(x.device.type)
    except TypeError:   # torch < 2.4
        return x.is_cuda and torch.is_autocast_enabled()


def inject_lora(model: nn.Module, target_modules, r: int, alpha: float, dropout: float,
                seed: int = 0) -> list[str]:
    """Replace every ``nn.Linear`` whose name ends in a target with LoRA."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator().manual_seed(seed)
    targets = tuple(target_modules)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                lora = LoRALinear(child, r, alpha, dropout, generator=generator)
                lora.to(child.weight.device)
                setattr(module, child_name, lora)
                replaced.append(f"{name}.{child_name}" if name else child_name)
    if not replaced:
        raise ValueError(f"no linear layers named {targets} were found")
    return replaced


def lora_parameters(model: nn.Module) -> "OrderedDict[str, nn.Parameter]":
    return OrderedDict((name, p) for name, p in model.named_parameters()
                       if name.endswith("lora_A") or name.endswith("lora_B"))


def lora_state_dict(model: nn.Module) -> OrderedDict:
    """The trainable state: fp32 CPU copies of every LoRA tensor."""
    return OrderedDict((name, p.detach().float().cpu().clone())
                       for name, p in lora_parameters(model).items())


@torch.no_grad()
def load_lora_state(model: nn.Module, state: dict, strict: bool = True) -> None:
    params = lora_parameters(model)
    if strict and set(params) != set(state):
        missing = sorted(set(params) - set(state))[:3]
        extra = sorted(set(state) - set(params))[:3]
        raise KeyError(f"LoRA state mismatch (missing {missing}, unexpected {extra})")
    for name, value in state.items():
        params[name].copy_(value.to(params[name].device, params[name].dtype),
                           non_blocking=True)
