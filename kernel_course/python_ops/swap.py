import torch


def swap(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Swaps the contents of tensor `x` with tensor `y`.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: The swapped tensor `y`.
    """

    x, y = y, x
    return y
