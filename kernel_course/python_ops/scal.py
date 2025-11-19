from typing import Optional
import torch


def scal(
    y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Scales the contents of tensor `y` by a scalar `alpha`.

    Args:
        y (torch.Tensor): Tensor to be scaled.
        alpha (float): Scalar multiplier.

    Returns:
        torch.Tensor: Scaled tensor.
    """

    y = alpha * y

    return y
