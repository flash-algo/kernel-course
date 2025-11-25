import torch


def axpby(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Updates tensor `y` by adding `x` scaled by `alpha` and `y` scaled by `beta`.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor to be updated.
        alpha (float): Scaling factor for `x`.
        beta (float): Scaling factor for `y`.

    Returns:
        torch.Tensor: The updated tensor `y`.
    """

    y = alpha * x + beta * y

    return y
