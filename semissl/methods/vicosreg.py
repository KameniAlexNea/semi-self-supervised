import argparse
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

import torch
import torch.nn as nn

from semissl.losses.vicosreg import cosine_reg_loss_func
from semissl.methods.base import BaseModel


class VICosReg(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        sim_loss_weight: float,
        var_loss_weight: float,
        cov_loss_weight: float,
        **kwargs
    ):
        """Implements VICosReg

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
        """

        super().__init__(**kwargs)

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parent_parser = super(VICosReg, VICosReg).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("vicosreg")

        # projector
        parser.add_argument("--output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--sim_loss_weight", default=25, type=float)
        parser.add_argument("--var_loss_weight", default=25, type=float)
        parser.add_argument("--cov_loss_weight", default=1.0, type=float)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the encoder and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICosReg reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICosReg loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # ------- cw reg loss -------
        cwreg_loss, (sim_loss, var_loss, cov_loss) = cosine_reg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        self.log("train_sim_loss", sim_loss, on_epoch=True, sync_dist=True)
        self.log("train_var_loss", var_loss, on_epoch=True, sync_dist=True)
        self.log("train_cov_loss", cov_loss, on_epoch=True, sync_dist=True)

        self.log("train_cwreg_loss", cwreg_loss, on_epoch=True, sync_dist=True)

        out.update({"loss": out["loss"] + cwreg_loss, "z": [z1, z2]})
        return out
