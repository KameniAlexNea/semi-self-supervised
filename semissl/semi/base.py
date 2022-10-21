"""
Base Implementation of Semi Supervised approach
"""

from typing import Dict
from typing import Sequence

import torch


def base_semi_wrapper(Method=object):
    class BaseSemiWrapper(Method):
        def training_step(
            self, batch: Dict[str, Sequence[torch.Tensor]], batch_idx: int
        ) -> torch.Tensor:
            index1, (X1, X2), label1 = batch["ssl"]
            index2, (X1r, X2r), label2 = batch["semi"]
            batch["ssl"] = (
                torch.cat((index1.reshape(-1), index2.reshape(-1))),
                [torch.cat((X1, X2r)), torch.cat((X2, X1r))],
                torch.cat((label1.reshape(-1), label2.reshape(-1))),
            )
            out = super().training_step(batch, batch_idx)
            return out

    return BaseSemiWrapper
