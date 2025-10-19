from typing import Union, Optional
import torch


def add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds two tensors element-wise.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        c (Optional[torch.Tensor], optional): An optional third tensor to store the result. Defaults to None.

    Returns:
        torch.Tensor: The result of the addition.
    """
    if c is None:
        c = torch.empty_like(a)

    torch.add(a, b, out=c)
    return c
    