"""Server state and evaluation, adapted from ECL-FL / 20NEWS-FL ``fl/server.py``."""

from __future__ import annotations

from collections import OrderedDict, deque

import numpy as np

from contribution.shapley import example_losses


class FLServer:
    def __init__(self, model, test_evalset, test_groups, num_test_groups: int, config):
        self.model = model
        self.config = config
        self.global_state_dict = model.get_state_dict()
        # Up to three preceding global states (needed by SDFR / AFR).
        self.global_history: deque = deque(maxlen=3)
        self.test_evalset = test_evalset
        self.test_groups = np.asarray(test_groups, dtype=np.int64)
        self.num_test_groups = num_test_groups

    def get_global_state_dict(self) -> OrderedDict:
        return OrderedDict((k, v.clone()) for k, v in self.global_state_dict.items())

    def update_global_model(self, new_state_dict: OrderedDict) -> None:
        self.global_history.append(self.get_global_state_dict())
        self.global_state_dict = OrderedDict(
            (k, v.clone()) for k, v in new_state_dict.items()
        )

    def evaluate_detailed(self, state_dict=None):
        """Mean test loss (per-example response cross-entropy) and its value
        per test group (the native Dolly categories)."""
        state = self.global_state_dict if state_dict is None else state_dict
        losses = example_losses(self.model, state, self.test_evalset, self.config.device)
        counts = np.bincount(self.test_groups, minlength=self.num_test_groups)
        sums = np.bincount(self.test_groups, weights=losses, minlength=self.num_test_groups)
        return float(losses.mean()), sums / np.maximum(counts, 1)
