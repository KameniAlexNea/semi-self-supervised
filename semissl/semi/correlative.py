"""
Correlation 1 between same class and zero with others
"""

import argparse
from typing import Any, List
from typing import Sequence

import torch
from torch import nn

from semissl.losses.correlation import cross_correlation_loss_func
from semissl.losses.correlation import feature_correlation_loss
from semissl.semi.base import base_semi_wrapper


def correlative_semi_wrapper(Method=object):
    class CorrelatedSemiWrapper(base_semi_wrapper(Method)):
        def __init__(
            self, n_class: int, lamb: float, scale: float, cross_weight: float, semi_proj_hidden_dim: int, **kwargs
        ) -> None:
            super().__init__(**kwargs)

            self.lamb = lamb
            self.scale = scale
            self.cross_weight = cross_weight

            self.output_dim = self.encoder.inplanes

            self.classifier = nn.Sequential(
                nn.Linear(self.output_dim, semi_proj_hidden_dim),
                nn.BatchNorm1d(semi_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(semi_proj_hidden_dim, n_class),
            )

        @property
        def learnable_params(self) -> List[dict]:
            """Adds supervised predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {
                    "name": "classifier",
                    "params": self.classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                },
            ]
            return super().learnable_params + extra_learnable_params

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("correlative_semi")

            parser.add_argument("--lamb", type=float, default=5e-3)
            parser.add_argument("--scale", type=float, default=0.025)
            parser.add_argument("--cross_weight", type=float, default=10)

            parser.add_argument("--semi_proj_hidden_dim", type=int, default=2048)

            return parent_parser

        def forward(self, X, *args, **kwargs):
            out = super().forward(X, *args, **kwargs)
            z = self.classifier(out["feats"])
            return {**out, "p": z}

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)

            *_, labels = batch["semi"]
            z1, z2 = out["feats"]
            z1 = z1[-len(labels) :]
            z2 = z2[-len(labels) :]

            p1, p2 = out["p"][-len(labels) :]
            p1 = p1[-len(labels) :]
            p2 = p2[-len(labels) :]

            cross_loss = cross_correlation_loss_func(p1, p2, labels, labels)

            corr_loss = feature_correlation_loss(z1, z2, labels, self.lamb, self.scale)

            self.log(
                "train_correlation_semi_loss",
                corr_loss,
                on_epoch=True,
                sync_dist=True,
            )

            self.log(
                "train_cross_entropy_semi_loss",
                cross_loss,
                on_epoch=True,
                sync_dist=True,
            )

            return out["loss"] + corr_loss + self.cross_weight * cross_loss

    return CorrelatedSemiWrapper
