from semissl.losses.barlow import barlow_loss_func
from semissl.losses.byol import byol_loss_func
from semissl.losses.deepclusterv2 import deepclusterv2_loss_func
from semissl.losses.dino import DINOLoss
from semissl.losses.moco import moco_loss_func
from semissl.losses.nnclr import nnclr_loss_func
from semissl.losses.ressl import ressl_loss_func
from semissl.losses.simclr import manual_simclr_loss_func
from semissl.losses.simclr import simclr_distill_loss_func
from semissl.losses.simclr import simclr_loss_func
from semissl.losses.simsiam import simsiam_loss_func
from semissl.losses.swav import swav_loss_func
from semissl.losses.vicosreg import cosine_reg_loss_func
from semissl.losses.vicreg import vicreg_loss_func
from semissl.losses.wmse import wmse_loss_func

__all__ = [
    "barlow_loss_func",
    "byol_loss_func",
    "deepclusterv2_loss_func",
    "DINOLoss",
    "moco_loss_func",
    "nnclr_loss_func",
    "ressl_loss_func",
    "simclr_loss_func",
    "manual_simclr_loss_func",
    "simclr_distill_loss_func",
    "simsiam_loss_func",
    "swav_loss_func",
    "vicreg_loss_func",
    "wmse_loss_func",
    "cosine_reg_loss_func",
]
