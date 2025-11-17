from typing import Optional, Tuple
import torch

def swap(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Swaps the contents of tensor `x` with tensor `y` using PyTorch operations.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The swapped tensors `(x, y)`.
    """

    temp = x.clone()
    x.copy_(y)
    y.copy_(temp)
    return x, y