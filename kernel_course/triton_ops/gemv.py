import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
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
def gemv_kernel(
    A,
    X,
    Y,
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    alpha,
    beta,
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
    # This program will process inputs that offset from the initial pointer
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize pointers to the start of the blocks
    A_ptr = A + offs_m[:, None] * stride_am
    x_ptr = X
    y_ptr = Y + offs_m * stride_y
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask_m = offs_m < n_elements_M
    # Initialize the accumulator to zero for each row
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    end_n = n_elements_N
    # Loop over the N dimension in blocks of BLOCK_N
    for start_n in range(0, end_n, BLOCK_N):
        # Align start_n to a multiple of BLOCK_N for efficient memory access
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # This program will process inputs that offset from the initial pointer
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # Create a mask to guard memory operations against out-of-bounds accesses
        mask_n = offs_n < n_elements_N
        # Load a block of A and x from DRAM, masking out any extra elements in case the input is not a multiple of the block size
        if EVEN_N & EVEN_M:
            a = tl.load(
                A_ptr + offs_n[None, :] * stride_an
            )
        else:
            a = tl.load(
                A_ptr + offs_n[None, :] * stride_an,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
        if EVEN_N:
            x = tl.load(x_ptr + offs_n * stride_x)
        else:
            x = tl.load(x_ptr + offs_n * stride_x, mask=mask_n, other=0.0)
        # Perform the matrix-vector multiplication for this block and accumulate the results
        acc += tl.sum(a * x[None, :], axis=1)
    # Load y from DRAM, masking out any extra elements in case the input is not a multiple of the block size
    if EVEN_M:
        y = tl.load(y_ptr)
    else:
        y = tl.load(y_ptr, mask=mask_m, other=0.0)
    # Compute y = alpha * A * x + beta * y
    y_new = beta * y
    y_new += alpha * acc
    # Write y back to DRAM
    if EVEN_M:
        tl.store(y_ptr, y_new)
    else:
        tl.store(y_ptr, y_new, mask=mask_m)


def gemv(
    A: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Updates tensor `y` by adding the product of matrix `A` and vector `x`
    scaled by `alpha`, and `y` scaled by `beta` using a Triton kernel.

    Args:
        A (torch.Tensor): Matrix tensor.
        x (torch.Tensor): Vector tensor to be multiplied with `A`.
        y (torch.Tensor): Vector tensor to be updated.
        alpha (float): Scaling factor for the product of `A` and `x`.
        beta (float): Scaling factor for `y`.

    Returns:
        torch.Tensor: The updated tensor `y`.
    """

    # Calculate the number of elements in the input tensors
    n_elements_M, n_elements_N = A.shape

    # The SPMD launch grid is one-dimensional, with each program processing a block of rows of A
    def grid(meta):
        return (triton.cdiv(n_elements_M, meta["BLOCK_M"]),)

    # Launch the Triton kernel
    gemv_kernel[grid](
        A,
        x,
        y,
        A.stride(0),
        A.stride(1),
        x.stride(0),
        y.stride(0),
        alpha,
        beta,
        n_elements_M,
        n_elements_N,
    )

    return y
