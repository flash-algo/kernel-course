import torch


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Updates tensor `C` by adding the product of matrices `A` and `B`
    scaled by `alpha`, and `C` scaled by `beta`.

    Args:
        A (torch.Tensor): First matrix tensor.
        B (torch.Tensor): Second matrix tensor to be multiplied with `A`.
        C (torch.Tensor): Matrix tensor to be updated.
        alpha (float): Scaling factor for the product of `A` and `B`.
        beta (float): Scaling factor for `C`.

    Returns:
        torch.Tensor: The updated tensor `C`.
    """

    C = alpha * A @ B + beta * C

    return C
