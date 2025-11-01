from typing import Optional
import torch


def add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds `b` to `a` element-wise using PyTorch operations.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        c (Optional[torch.Tensor], optional): An optional third tensor to store the result. Defaults to None.

    Returns:
        torch.Tensor: The result of the addition.
    """

    if c is None:
        c = torch.empty_like(a)

    c = torch.add(a, b)

    return c
    