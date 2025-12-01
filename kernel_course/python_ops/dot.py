import torch


def dot(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the dot product of two tensors.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: The dot product of `x` and `y`.
    """

    x = x.reshape(-1)
    y = y.reshape(-1)

    z = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    for i in range(len(x)):
        z += (x[i] * y[i]).to(torch.float32)
    z = z.to(x.dtype)

    return z
