import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2
        ),
    ],
    key=["n_elements_M", "n_elements_K", "n_elements_N"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["n_elements_M"] % args["BLOCK_M"] == 0,
        "EVEN_K": lambda args: args["n_elements_K"] % args["BLOCK_K"] == 0,
        "EVEN_N": lambda args: args["n_elements_N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gemm_kernel(
    A,
    B,
    C,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    alpha,
    beta,
    n_elements_M,
    n_elements_K,
    n_elements_N,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # There are multiple program processing different blocks of data
    # We identify which program we are in using program_id
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # This program will process inputs that offset from the initial pointer
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_kb = tl.arange(0, BLOCK_K)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize pointers to the start of the blocks
    a_ptr = A + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptr = B + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask_m = offs_m < n_elements_M
    mask_n = offs_n < n_elements_N
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    end_k = n_elements_K
    # Loop over K dimension
    for start_k in range(0, end_k, BLOCK_K):
        # Align start_k to a multiple of BLOCK_K for efficient memory access
        start_k = tl.multiple_of(start_k, BLOCK_K)
        # This program will process inputs that offset from the initial pointer
        offs_k = start_k + offs_kb
        # Create a mask to guard memory operations against out-of-bounds accesses
        mask_k = offs_k < n_elements_K
        # Load a block of A and B from DRAM, masking out any extra elements in case the input is not a multiple of the block size
        if EVEN_M & EVEN_K:
            a = tl.load(a_ptr + start_k * stride_ak)
        else:
            a = tl.load(
                a_ptr + start_k * stride_ak,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
        if EVEN_N & EVEN_K:
            b = tl.load(b_ptr + start_k * stride_bk)
        else:
            b = tl.load(
                b_ptr + start_k * stride_bk,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
        # Perform the matrix multiplication for the current block and accumulate the result
        acc += tl.dot(a, b)
    # Load C from DRAM, masking out any extra elements in case the input is not a multiple of the block size
    if EVEN_M & EVEN_N:
        c = tl.load(c_ptr)
    else:
        c = tl.load(
            c_ptr,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
    # Compute C = alpha * A * B + beta * C
    c = beta * c
    c += alpha * acc
    # Store the updated C back to DRAM
    if EVEN_M & EVEN_N:
        tl.store(c_ptr, c)
    else:
        tl.store(
            c_ptr,
            c,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Updates tensor `C` by adding the product of matrices `A` and `B`
    scaled by `alpha`, and `C` scaled by `beta` using Triton operations.

    Args:
        A (torch.Tensor): First matrix tensor.
        B (torch.Tensor): Second matrix tensor to be multiplied with `A`.
        C (torch.Tensor): Matrix tensor to be updated.
        alpha (float): Scaling factor for the product of `A` and `B`.
        beta (float): Scaling factor for `C`.

    Returns:
        torch.Tensor: The updated tensor `C`.
    """

    # Calculate the number of elements in the input tensors
    n_elements_M, n_elements_K = A.shape
    n_elements_K, n_elements_N = B.shape

    # The SPMD grid is a 2D grid where each program computes a BLOCK_M x BLOCK_N block of the output matrix C
    def grid(meta):
        return (
            triton.cdiv(n_elements_M, meta["BLOCK_M"]),
            triton.cdiv(n_elements_N, meta["BLOCK_N"]),
        )

    # Launch the Triton kernel
    gemm_kernel[grid](
        A,
        B,
        C,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        C.stride(0),
        C.stride(1),
        alpha,
        beta,
        n_elements_M,
        n_elements_K,
        n_elements_N,
    )

    return C
