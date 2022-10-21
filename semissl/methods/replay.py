import argparse
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union

import torch

from semissl.methods.barlow_twins import BarlowTwins
from semissl.methods.byol import BYOL
from semissl.methods.vicreg import VICReg


def random_replay_model(Method=object):
    class RandReplayModel(Method):
        def __init__(self, threshold: float, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.threshold = threshold

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parent_parser = super(
                RandReplayModel, RandReplayModel
            ).add_model_specific_args(parent_parser)
            parser = parent_parser.add_argument_group("rand-replay")
            parser.add_argument("--threshold", default=0.4, type=float)
            return parent_parser

        def training_step(
            self, batch: Union[Dict[str, Sequence[Any]], Sequence[Any]], batch_idx: int
        ) -> torch.Tensor:
            if torch.rand(1) < self.threshold:
                index1, (X1, X2), label1 = batch[f"ssl"]
                # introduce probability here
                # symetry test
                index2, (X1r, X2r), label2 = batch[f"replay"]
                batch[f"ssl"] = (
                    torch.cat((index1.reshape(-1), index2.reshape(-1))),
                    [torch.cat((X1, X2r)), torch.cat((X2, X1r))],
                    torch.cat((label1.reshape(-1), label2.reshape(-1))),
                )
            return super().training_step(batch, batch_idx)

    return RandReplayModel


def replay_model(Method=object):
    class ReplayModel(Method):
        def training_step(
            self, batch: Union[Dict[str, Sequence[Any]], Sequence[Any]], batch_idx: int
        ) -> torch.Tensor:
            index1, (X1, X2), label1 = batch[f"ssl"]
            # introduce probability here
            # symetry test
            index2, (X1r, X2r), label2 = batch[f"replay"]
            batch[f"ssl"] = (
                torch.cat((index1.reshape(-1), index2.reshape(-1))),
                [torch.cat((X1, X1r)), torch.cat((X2, X2r))],
                torch.cat((label1.reshape(-1), label2.reshape(-1))),
            )
            return super().training_step(batch, batch_idx)

    return ReplayModel


RVICReg = replay_model(VICReg)

RBarlowTwins = replay_model(BarlowTwins)

RByol = replay_model(BYOL)

RandRVICReg = random_replay_model(VICReg)

RandRBarlowTwins = random_replay_model(BarlowTwins)

RandRByol = random_replay_model(BYOL)
