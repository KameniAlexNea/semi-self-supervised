import argparse

import pytorch_lightning as pl

from semissl.args.dataset import augmentations_args
from semissl.args.dataset import dataset_args
from semissl.args.semi import semi_ssl_args
from semissl.args.utils import additional_setup_linear
from semissl.args.utils import additional_setup_pretrain
from semissl.methods import METHODS
from semissl.semi import SEMISUPERVISED
from semissl.utils.checkpointer import Checkpointer

try:
    from semissl.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


def parse_args_pretrain(default_args: str = None) -> argparse.Namespace:
    """Parses dataset, augmentation, pytorch lightning, model specific and additional args.

    First adds shared args such as dataset, augmentation and pytorch lightning args, then pulls the
    model name from the command and proceeds to add model specific args from the desired class. If
    wandb is enabled, it adds checkpointer args. Finally, adds additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """
    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)
    augmentations_args(parser)
    semi_ssl_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add method-specific arguments
    parser.add_argument("--method", type=str)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args(default_args)

    # add model specific args
    parser = METHODS[temp_args.method].add_model_specific_args(parser)

    # add semissl args
    if temp_args.semissl:
        parser = SEMISUPERVISED[temp_args.semissl]().add_model_specific_args(parser)

    # add checkpoint and auto umap args
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--auto_umap", action="store_true")
    temp_args, _ = parser.parse_known_args(default_args)

    # optionally add checkpointer and AutoUMAP args
    if temp_args.save_checkpoint:
        parser = Checkpointer.add_checkpointer_args(parser)

    if _umap_available and temp_args.auto_umap:
        parser = AutoUMAP.add_auto_umap_args(parser)

    # parse args
    args = parser.parse_args(default_args)

    # prepare arguments with additional setup
    additional_setup_pretrain(args)

    return args


def parse_args_linear() -> argparse.Namespace:
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds and arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str)

    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["linear"].add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB
    temp_args, _ = parser.parse_known_args()

    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--domain", type=str, default=None)

    # add checkpointer args (only if logging is enabled)
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    # parse args
    args = parser.parse_known_args()[0]
    additional_setup_linear(args)

    return args
