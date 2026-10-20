"""Federated client, adapted from ECL-FL / 20NEWS-FL ``fl/client.py``.

Same contract: load the global (LoRA) state, train locally, return the delta
on CPU.  Clients share one model object (a 0.5B base model per client would
not fit), so only the data loader and its random generator belong to the
client.  Every round runs ``local_steps`` optimizer steps of ``batch_size``
examples (``micro_batch_size`` per forward pass) with a fresh AdamW, on a
fresh shuffle of the client's data.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("fedit-fl")


class NonFiniteLoss(FloatingPointError):
    pass


class FLClient:
    def __init__(self, client_id: int, dataset, collate, config):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.last_train_loss = float("nan")   # mean local loss of the last train()
        if config.batch_size % config.micro_batch_size:
            raise ValueError("batch_size must be a multiple of micro_batch_size")
        loader_kwargs = dict(
            batch_size=config.micro_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(
                int(getattr(config, "seed", 0)) * 10_000 + int(client_id)
            ),
        )
        if config.device == "cuda":
            loader_kwargs.update(num_workers=config.num_workers, pin_memory=True)
        self.data_loader = DataLoader(dataset, **loader_kwargs)
        if len(self.data_loader) == 0:
            raise ValueError(f"client {client_id} has no local data")

    def _micro_batches(self):
        while True:   # a client with fewer examples than one round needs cycles
            yield from self.data_loader

    def train(self, model, global_state_dict: OrderedDict) -> OrderedDict:
        config = self.config
        device = config.device
        use_amp = device == "cuda"
        model.load_state_dict_from(global_state_dict)
        model.train()
        params = model.trainable_parameters()
        optimizer = torch.optim.AdamW(params, lr=config.local_lr,
                                      weight_decay=config.weight_decay)
        accumulation = config.batch_size // config.micro_batch_size
        total_steps = int(config.local_steps)
        warmup_steps = int(total_steps * config.warmup_ratio)
        batches = self._micro_batches()
        loss_sum = torch.zeros((), device=device)
        for step in range(total_steps):
            if config.local_lr_schedule == "constant":
                lr = config.local_lr
            elif step < warmup_steps:
                lr = config.local_lr * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = config.local_lr * (1.0 - 0.9 * progress)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                input_ids, attention, labels, _ = next(batches)
                input_ids = input_ids.to(device, non_blocking=use_amp)
                attention = attention.to(device, non_blocking=use_amp)
                labels = labels.to(device, non_blocking=use_amp)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = model.response_losses(input_ids, attention, labels).mean()
                if not torch.isfinite(loss):
                    raise NonFiniteLoss(f"client {self.client_id} produced a non-finite "
                                        f"training loss at local step {step}")
                (loss / accumulation).backward()
                loss_sum += loss.detach() / accumulation
            torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
            optimizer.step()
        self.last_train_loss = float(loss_sum) / max(total_steps, 1)

        local_state = model.get_state_dict()
        update = OrderedDict(
            (key, value - global_state_dict[key].to(value.dtype))
            for key, value in local_state.items()
        )
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        model.eval()
        return update
