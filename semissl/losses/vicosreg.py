import torch
import torch.nn.functional as F


def variance_loss(z1: torch.Tensor, z2: torch.Tensor):
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor):
    return (z1 - z2).pow(2).sum(dim=-1).mean()


def pairwise_cosine_loss(z1: torch.Tensor, z2: torch.Tensor):
    """
    An approximate computation of Pairwise Squared Cosine Similarity
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: off-diagonal Cosine Similarity Loss.
    """
    D = z1.size(1)
    z = (z1.T @ z2).pow(2) / D  # (z1.pow(2).sum(0) @ z2.pow(2).sum(0) + eps)
    diag = torch.eye(D)
    return z[~diag.bool()].sum()


def cosine_reg_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
) -> torch.Tensor:
    """
    Computes VICosReg's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.

    Returns:
        torch.Tensor: VICosReg loss.
    """
    sim_loss = invariance_loss(z1, z2)
    var_loss = variance_loss(z1, z2)
    cov_loss = pairwise_cosine_loss(z1, z2)

    loss = (
        sim_loss_weight * sim_loss
        + var_loss_weight * var_loss
        + cov_loss_weight * cov_loss
    )
    return loss, (sim_loss, var_loss, cov_loss)
