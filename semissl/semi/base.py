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
            index1, (X1, X2), label1 = batch[f"ssl"]
            index2, (X1s, X2s), label2 = batch[f"semi"]
            batch[f"ssl"] = (
                torch.cat((index1.reshape(-1), index2.reshape(-1))),
                [torch.cat((X1, X2s)), torch.cat((X2, X1s))],
                torch.cat((label1.reshape(-1), label2.reshape(-1))),
            )
            out = super().training_step(batch, batch_idx)
            return out

    return BaseSemiWrapper
