"""
Base Implementation of Semi Supervised approach
"""

import argparse
from typing import Dict
from typing import Sequence

import torch


def base_semi_wrapper(Method=object):
    class BaseSemiWrapper(Method):
        def training_step(self, batch: Dict[str, Sequence[torch.Tensor]], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            return out

    return BaseSemiWrapper
