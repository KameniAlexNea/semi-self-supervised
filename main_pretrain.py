import os
import types
from pprint import pprint

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from semissl.args.setup import parse_args_pretrain
from semissl.semi import SEMISUPERVISED
from semissl.methods import METHODS

try:
    from semissl.methods.dali import PretrainABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from semissl.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

from semissl.utils.checkpointer import Checkpointer
from semissl.utils.classification_dataloader import (
    prepare_data as prepare_data_classification,
)
from semissl.utils.pretrain_dataloader import mask_dataset, prepare_dataloader
from semissl.utils.pretrain_dataloader import prepare_datasets
from semissl.utils.pretrain_dataloader import prepare_multicrop_transform
from semissl.utils.pretrain_dataloader import prepare_n_crop_transform
from semissl.utils.pretrain_dataloader import prepare_transform


def main():
    seed_everything(5)

    args = parse_args_pretrain()

    # online eval dataset reloads when task dataset is over
    args.multiple_trainloader_mode = "max_size_cycle"

    # set online eval batch size and num workers
    args.online_eval_batch_size = (
        None  # int(args.batch_size) if args.dataset == "cifar100" else None
    )

    # pretrain and online eval dataloaders
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                args.dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        if args.multicrop:
            assert not args.unique_augs == 1

            if args.dataset in ["cifar10", "cifar100"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            # imagenet or custom dataset
            else:
                size_crops = [224, 96]

            transform = prepare_multicrop_transform(
                transform,
                size_crops=size_crops,
                num_crops=[args.num_crops, args.num_small_crops],
            )
        else:
            if args.num_crops != 2:
                assert args.method == "wmse"

            online_eval_transform = (
                transform[-1] if isinstance(transform, list) else transform
            )
            task_transform = prepare_n_crop_transform(
                transform, num_crops=args.num_crops
            )

        task_dataset, online_eval_dataset = prepare_datasets(
            args.dataset,
            task_transform=task_transform,
            online_eval_transform=online_eval_transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels=args.no_labels,
        )

        task_dataset, label_task_dataset = mask_dataset(
            task_dataset, args.dataset, args.semi_rate
        )

        label_loader = prepare_dataloader(
            label_task_dataset,
            batch_size=max(args.batch_size // 4, int(args.batch_size * len(label_task_dataset) / len(task_dataset))),
            num_workers=args.num_workers,
        )

        task_loader = prepare_dataloader(
            task_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        train_loaders = {"ssl": task_loader, "semi": label_loader}

        if args.online_eval_batch_size:
            online_eval_loader = prepare_dataloader(
                online_eval_dataset,
                batch_size=args.online_eval_batch_size,
                num_workers=args.num_workers,
            )
            train_loaders.update({"online_eval": online_eval_loader})

    # normal dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
        val_loader = None
    else:
        _, val_loader = prepare_data_classification(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # check method
    assert args.method in METHODS, f"Choose from {METHODS.keys()}"
    # build method
    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(
            f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass)
        )

    if args.semissl:
        MethodClass = SEMISUPERVISED[args.semissl](MethodClass)

    model: torch.nn.Module = MethodClass(
        **args.__dict__, n_class = 10 if args.dataset.lower() == "cifar10" else 100
    )

    callbacks = []

    # wandb logging
    wandb_logger = True
    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}",
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            reinit=True,
            # log_model=True,
        )
        if args.resume_from_checkpoint is None:
            wandb_logger.watch(model, log="gradients", log_freq=100)
            wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=args.checkpoint_dir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    trainer: Trainer = Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=callbacks, log_every_n_steps=25
    )

    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loaders, val_loader)


if __name__ == "__main__":
    main()
