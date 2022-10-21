import torch
from torch.nn import functional as F


def feature_correlation_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    lamb: float = 5e-3,
    scale: float = 0.025,
):
    """
    Compute correlation between input regarding labels

    Inputs Features with same label should be close to 1 and 0 if different labels
    """
    N, D = z1.size()

    # to match the original code
    bn = torch.nn.BatchNorm1d(2 * N, affine=False).to(z1.device)
    labels = torch.cat([labels, labels])
    label_index = torch.argsort(labels)
    _, unique_counts = torch.unique(labels, return_counts=True)

    z = bn(torch.cat((z1, z2))[label_index].T)
    eye_mat: torch.Tensor = torch.block_diag(
        *[torch.ones(c, c, device=z1.device) for c in unique_counts]
    )
    corr = torch.einsum("bi, bj -> ij", z, z) / (2 * D)
    corr = (corr - eye_mat).pow(2)
    corr[~eye_mat.bool()] *= lamb
    return corr.sum() * scale


def cross_correlation_loss_func(
    p1: torch.Tensor, p2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor
):
    return F.cross_entropy(torch.cat([p1, p2]), torch.cat([y1, y2]))
