from typing import Optional
import torch


def axpy(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds `b` scaled by `alpha` to `a` element-wise.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        alpha (float): Scaling factor for `b`.
        c (Optional[torch.Tensor], optional): An optional third tensor to store the result. Defaults to None.

    Returns:
        torch.Tensor: The result of the addition.
    """

    if c is None:
        c = torch.empty_like(a)

    c = a + b * alpha

    return c
    