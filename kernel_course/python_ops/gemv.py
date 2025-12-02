import torch


def gemv(
    A: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Updates tensor `y` by adding the product of matrix `A` and vector `x`
    scaled by `alpha`, and `y` scaled by `beta`.

    Args:
        A (torch.Tensor): Matrix tensor.
        x (torch.Tensor): Vector tensor to be multiplied with `A`.
        y (torch.Tensor): Vector tensor to be updated.
        alpha (float): Scaling factor for the product of `A` and `x`.
        beta (float): Scaling factor for `y`.
    """

    y = alpha * A @ x + beta * y

    return y
