"""Base causal LM + LoRA, and the per-example response loss.

The base model (default Qwen2.5-0.5B, Apache-2.0) is loaded from a local
folder (``model_cache/Qwen2.5-0.5B``, see ``fetch_model.sh``), frozen, and
kept in bf16 on CUDA (fp32 on CPU).  LoRA tensors stay fp32.

``response_losses`` returns one number per example: the mean token
cross-entropy over its response tokens (and EOS).  Only those positions go
through the LM head, in chunks, so the 151,936-way vocabulary never builds a
full ``batch x length x vocab`` logit tensor.  Training minimises the mean of
these per-example losses; the Shapley value functions, the test loss and the
per-class losses are means of the same quantity over sets of examples.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lora import inject_lora, load_lora_state, lora_parameters, lora_state_dict

logger = logging.getLogger("fedit-fl")


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class FedLM(nn.Module):
    """A frozen causal LM with LoRA adapters; the state dict is LoRA only."""

    def __init__(self, base, config):
        super().__init__()
        self.lm = base
        self.config = config
        self.targets = inject_lora(base, config.lora_target_modules, config.lora_r,
                                   config.lora_alpha, config.lora_dropout,
                                   seed=config.lora_init_seed)
        self.decoder = base.get_decoder()
        self.head = base.get_output_embeddings()

    # same names as the ECL-FL / 20NEWS-FL model wrappers
    def get_state_dict(self) -> OrderedDict:
        return lora_state_dict(self)

    def load_state_dict_from(self, state: dict) -> None:
        load_lora_state(self, state)

    def trainable_parameters(self):
        return list(lora_parameters(self).values())

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def response_losses(self, input_ids, attention_mask, labels) -> torch.Tensor:
        """Mean response-token cross-entropy of each example (float32, [B])."""
        hidden = self.decoder(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False).last_hidden_state
        targets = labels[:, 1:]
        mask = targets != -100
        selected = hidden[:, :-1][mask]
        flat_targets = targets[mask]
        owner = torch.arange(labels.shape[0], device=labels.device
                             ).unsqueeze(1).expand_as(targets)[mask]
        token_loss = []
        chunk = self.config.lm_head_chunk
        for start in range(0, selected.shape[0], chunk):
            logits = self.head(selected[start:start + chunk]).float()
            token_loss.append(F.cross_entropy(logits, flat_targets[start:start + chunk],
                                              reduction="none"))
        token_loss = torch.cat(token_loss) if token_loss else selected.new_zeros(0)
        sums = torch.zeros(labels.shape[0], dtype=torch.float32, device=labels.device)
        sums.index_add_(0, owner, token_loss.float())
        counts = mask.sum(dim=1).clamp_min(1).float()
        return sums / counts


def create_model(config, device: str) -> FedLM:
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device == "cuda" and config.bf16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=dtype,
                                                attn_implementation="sdpa")
    base.config.use_cache = False
    model = FedLM(base, config)
    model.to(device)
    logger.info("model %s (%s) with LoRA r=%d on %d layers: %.2fM trainable parameters",
                config.model_path, dtype, config.lora_r, len(model.targets),
                model.num_trainable() / 1e6)
    return model
