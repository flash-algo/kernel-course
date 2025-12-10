import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ],
    key=["n_elements_M", "n_elements_N"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["n_elements_M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["n_elements_N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def geru_kernel(
    A,
    X,
    Y,
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    alpha,
    n_elements_M,
    n_elements_N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # There are multiple program processing different blocks of data
    # We identify which program we are in using program_id
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # This program will process inputs that offset from the initial pointer
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize pointers to the start of the blocks
    A_ptr = (
        A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    )
    x_ptr = (
        X + offs_m * stride_x
    )
    y_ptr = (
        Y + offs_n * stride_y
    )
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask_m = offs_m < n_elements_M
    mask_n = offs_n < n_elements_N
    # Load a block of A from DRAM, masking out any extra elements in case the input is not a multiple of the block size
    if EVEN_N & EVEN_M:
        a = tl.load(A_ptr)
    else:
        a = tl.load(
            A_ptr,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
    # Load x and y vectors from DRAM
    if EVEN_M:
        x = tl.load(x_ptr)
    else:
        x = tl.load(x_ptr, mask=mask_m, other=0.0)
    if EVEN_N:
        y = tl.load(y_ptr)
    else:
        y = tl.load(y_ptr, mask=mask_n, other=0.0)
    # Compute the outer product
    p = x[:, None] * y[None, :]
    # Scale by alpha and update A
    a += alpha * p
    # Store the updated block of A back to DRAM
    if EVEN_N & EVEN_M:
        tl.store(A_ptr, a)
    else:
        tl.store(
            A_ptr,
            a,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def geru(
    A: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
):
    """
    Updates tensor `A` by adding the outer product of vectors `x` and `y` scaled by `alpha` using a Triton kernel.

    Args:
        A (torch.Tensor): Matrix tensor to be updated.
        x (torch.Tensor): Vector tensor.
        y (torch.Tensor): Vector tensor.
        alpha (float): Scaling factor for the outer product of `x` and `y`.

    Returns:
        torch.Tensor: The updated tensor `A`.
    """

    # Calculate the number of elements in the input tensors
    n_elements_M, n_elements_N = A.shape

    # The SPMD launch grid is two-dimensional, with each program processing a block of A
    def grid(meta):
        return (triton.cdiv(n_elements_M, meta["BLOCK_M"]), triton.cdiv(n_elements_N, meta["BLOCK_N"]))

    # Launch the Triton kernel
    geru_kernel[grid](
        A,
        x,
        y,
        A.stride(0),
        A.stride(1),
        x.stride(0),
        y.stride(0),
        alpha,
        n_elements_M,
        n_elements_N,
    )

    return A
