from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def copy_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # There are multiple program processing different blocks of data
    # We identify which program we are in using program_id
    pid = tl.program_id(axis=0)
    # This program will process inputs that offset from the initial pointer
    # For example, if you had a vector of size 256 and block_size of 64, the programs would each access the elements [0:64], [64:128], [128:192], [192:256]
    # We need note that offsets is a list of pointers
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
    # Load x from DRAM, masking out any extra elements in case the input is not a multiple of the block_size
    x = tl.load(x_ptr + offsets, mask=mask)
    # Write x to y in DRAM
    tl.store(y_ptr + offsets, x, mask=mask)


def copy(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Copies the contents of tensor `x` into tensor `y` using a Triton kernel.

    Args:
        x (torch.Tensor): Source tensor.
        y (torch.Tensor): Destination tensor.
    
    Returns:
        torch.Tensor: The destination tensor `y`.
    """

    # Calculate the number of elements in the input tensor
    n_elements = x.numel()

    # The SPMD launch grid denotes the number of kernel instances that run it parallelly
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    # In this case, we use a 1D grid where the size is the number of blocks needed to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Each torch.tensor object is implicitly converted into a pointer to its first element
    # `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel, Don't forget to pass meta-parameters as keyword arguments
    copy_kernel[grid](x, y, n_elements)

    return y