from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
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
    # Load a and b from DRAM, masking out any extra elements in case the input is not a multiple of the block_size
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    # Write c back to DRAM
    tl.store(c_ptr + offsets, c, mask=mask)


def add(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    c: Optional[torch.Tensor] = None,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Adds `b` to `a` element-wise using a Triton kernel.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        c (Optional[torch.Tensor], optional): An optional third tensor to store the result. Defaults to None.
        block_size (int, optional): The block size for the Triton kernel. Defaults to 1024.

    Returns:
        torch.Tensor: The result of the addition.
    """

    if c is None:
        # We need to preallocate the output tensor
        c = torch.empty_like(a)

    # Calculate the number of elements in the input tensors
    n_elements = a.numel()

    # The SPMD launch grid denotes the number of kernel instances that run it parallelly
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    # In this case, we use a 1D grid where the size is the number of blocks needed to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Each torch.tensor object is implicitly converted into a pointer to its first element
    # `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel, Don't forget to pass meta-parameters as keyword arguments
    add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=block_size)

    # We return a handle to c but, since `torch.cuda.synchronize()` has not been called, the kernel is still returning asynchronously at this point
    return c