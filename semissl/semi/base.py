"""
Base Implementation of Semi Supervised approach
"""

from typing import Any
from typing import Sequence

import torch


def base_distill_wrapper(Method=object):
    class BaseDistillWrapper(Method):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

            self.output_dim = kwargs["output_dim"]

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:

            index1, (X1, X2), label1 = batch[f"ssl"]
            index2, (X1r, X2r), label2 = batch[f"semi"]
            batch[f"ssl"] = (
                torch.cat((index1.reshape(-1), index2.reshape(-1))),
                [torch.cat((X1, X2r)), torch.cat((X2, X1r))],
                torch.cat((label1.reshape(-1), label2.reshape(-1))),
            )
            out = super().training_step(batch, batch_idx)
            return out

    return BaseDistillWrapper
