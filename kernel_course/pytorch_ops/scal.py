from typing import Optional
import torch


def scal(
    y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Scales the contents of tensor `y` by a scalar `alpha` using PyTorch operations.

    Args:
        y (torch.Tensor): Tensor to be scaled.
        alpha (float): Scalar multiplier.
    Returns:
        torch.Tensor: Scaled tensor.
    """

    y = torch.mul(y, alpha)

    return y
