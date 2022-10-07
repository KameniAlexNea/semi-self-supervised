"""
MSE Loss between center and elements of same class
"""

import argparse
from typing import Any, List
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from semissl.semi.base import base_semi_wrapper

def cross_entropy_semi_wrapper(Method=object):
    class CrossEntropySemiWrapper(base_semi_wrapper(Method)):
        def __init__(self, n_class: int, semi_lamb: float, **kwargs) -> None:
            super().__init__(**kwargs)
            self.output_dim = self.encoder.inplanes
            self.n_class = n_class
            self.semi_lamb = semi_lamb
            self.linear = nn.Linear(self.output_dim, self.n_class)

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("cross_entropy_semissl")

            parser.add_argument("--semi_lamb", type=float, default=10)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds supervised predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {
                    "name": "linear",
                    "params": self.linear.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                },
            ]
            return super().learnable_params + extra_learnable_params
        
        def forward(self, X, *args, **kwargs):
            out = super().forward(X, *args, **kwargs)
            z = self.linear(out["feats"])
            return {**out, "p": z}

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)

            *_, labels = batch["semi"]
            p1, p2 = out["p"][-len(labels):]
            p1 = p1[-len(labels):]
            p2 = p2[-len(labels):]
            
            cross_entropy = F.cross_entropy(torch.cat([p1, p2]), torch.cat([labels, labels]))

            self.log(
                "train_cross_entropy_semi_loss",
                cross_entropy,
                on_epoch=True,
                sync_dist=True,
            )

            return out["loss"] + self.semi_lamb * cross_entropy

    return CrossEntropySemiWrapper