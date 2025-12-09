import torch


def geru(
    A: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
):
    """
    Updates tensor `A` by adding the outer product of vectors `x` and `y` scaled by `alpha` using PyTorch operations.

    Args:
        A (torch.Tensor): Matrix tensor to be updated.
        x (torch.Tensor): Vector tensor.
        y (torch.Tensor): Vector tensor.
        alpha (float): Scaling factor for the outer product of `x` and `y`.

    Returns:
        torch.Tensor: The updated tensor `A`.
    """

    A += torch.mul(torch.ger(x, y), alpha)

    return A
