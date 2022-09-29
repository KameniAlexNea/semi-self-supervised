"""
Correlation 1 between same class and zero with others
"""

import argparse
from typing import Any
from typing import Sequence

import torch
from functorch import vmap

def feature_corralation(z1: torch.Tensor, z2: torch.Tensor):
    if z1.dim() == 1:
        z1 = z1.view(1, -1)
        z2 = z2.view(1, -1)
    N, D = z1.size()

    z1 = z1 - z1.mean(dim=1, keepdim=True)
    z2 = z2 - z2.mean(dim=1, keepdim=True)
    cov_z1 = (z1 @ z1.T) / (D - 1)
    cov_z2 = (z2 @ z2.T) / (D - 1)
    diag = torch.eye(N)
    return ((1 - cov_z1)[~diag.bool()].pow(2).mean() + (1 - cov_z2)[~diag.bool()].pow(2).mean()) / 2

def correlated_semi_wrapper(Method=object):
    class CorrelatedSemiWrapper(Method):
        def __init__(self, semi_lamb: float, **kwargs) -> None:
            super().__init__(**kwargs)

            self.semi_lamb = semi_lamb
        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")

            parser.add_argument("--semi_lamb", type=float, default=1)

            return parent_parser

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)

            *_, labels = batch["semi"]
            z1, z2 = out["feats"]
            z1 = z1[-len(labels):]
            z2 = z2[-len(labels):]
            labels = labels[labels != -1]
            unique_labels: torch.Tensor = torch.unique(labels)

            result: torch.Tensor = vmap(lambda label: feature_corralation(z1[labels==label], z2[labels==label]))(unique_labels)
            correlation_loss = result.mean()

            self.log(
                "train_correlation_semi_loss",
                correlation_loss,
                on_epoch=True,
                sync_dist=True,
            )

            return out["loss"] + self.semi_lamb * correlation_loss

    return CorrelatedSemiWrapper