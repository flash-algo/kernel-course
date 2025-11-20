import torch


def copy(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Copies the contents of tensor `x` into tensor `y`.

    Args:
        x (torch.Tensor): Source tensor.
        y (torch.Tensor): Destination tensor.

    Returns:
        torch.Tensor: The destination tensor `y`.
    """

    y = x

    return y
