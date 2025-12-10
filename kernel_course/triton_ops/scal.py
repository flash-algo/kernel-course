import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=2, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def scal_kernel(
    Y,
    alpha,
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
    # Initialize pointers to the start of the blocks
    y_ptr = Y + offsets
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
    # Load y from DRAM, masking out any extra elements in case the input is not a multiple of the block_size
    y = tl.load(y_ptr, mask=mask)
    # Scale y by alpha
    y = y * alpha
    # Write y back to DRAM
    tl.store(y_ptr, y, mask=mask)


def scal(
    y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Scales tensor `y` by a scalar `alpha` using a Triton kernel.

    Args:
        y (torch.Tensor): Input tensor to be scaled.
        alpha (float): Scalar value to scale the tensor.

    Returns:
        torch.Tensor: The scaled tensor `y`.
    """

    # Calculate the number of elements in the input tensor
    n_elements = y.numel()

    # The SPMD launch grid denotes the number of kernel instances that run it parallelly
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    # In this case, we use a 1D grid where the size is the number of blocks needed to cover all elements
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch the Triton kernel
    scal_kernel[grid](y, alpha, n_elements)

    return y
