from typing import Optional
import torch

def swap(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Swaps the contents of tensor `x` with tensor `y` using PyTorch operations.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.
    
    Returns:
        torch.Tensor: The swapped tensor `y`.
    """

    temp = x.clone()
    x.copy_(y)
    y.copy_(temp)
    return y