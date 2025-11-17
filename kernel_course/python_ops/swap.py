from typing import Optional, Tuple
import torch

def swap(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Swaps the contents of tensor `x` with tensor `y`.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The swapped tensors `(x, y)`.
    """

    x, y = y, x
    return x, y