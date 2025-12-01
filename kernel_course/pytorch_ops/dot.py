import torch


def dot(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the dot product of two tensors using PyTorch operations.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: The dot product of `x` and `y`.
    """

    z = torch.sum(torch.mul(x, y))

    return z
