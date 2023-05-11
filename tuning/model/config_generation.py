from itertools import product
from functools import reduce
from operator import mul

from typing import List, Optional
from utils.data_types import Pipeline, OperationType, DataType, DispatchConfig, Dispatch, TargetBackend

###################################################################################################
# This file contains library for producing configs to annotate mlir models.
###################################################################################################


# CUDA Tensorcore Configs
###################################################################################################

def generate_tile_sizes(pipeline: Pipeline, data_type: DataType, input_shape: List[int]) -> List[List[int]]:
    """"Returns list of possible tile sizes for input shape and pipeline"""
    tile_sizes = []

    if pipeline == Pipeline.GPU_TENSORCORE:
        # Tensorcor main dims M, N, K
        tensorcore_mn_sizes = [16, 32, 64, 128, 256]
        tensorcore_k_sizes = [16, 32, 64, 128]
        tile_sizes = list(product(tensorcore_mn_sizes,
                                  tensorcore_mn_sizes, tensorcore_k_sizes))

        # K dim is always smaller than the picked M or N
        tile_sizes = [tile_size for tile_size in tile_sizes if tile_size[2]
                      <= tile_size[0] and tile_size[2] <= tile_size[1]]

    # Batch dim tile is always 1 if it exists
    if len(input_shape) == 4:
        tile_sizes = [[1] + list(tile_size) for tile_size in tile_sizes]

    return tile_sizes


def generate_workgroup_sizes(pipeline: Pipeline, input_shape: List[int], tile_size: List[int]) -> List[List[int]]:
    """"Returns list of possible workgroup sizes for tile size"""
    workgroup_sizes = []

    if pipeline == Pipeline.GPU_TENSORCORE:
        # Tensorcore main dims X, Y, Z
        tensorcore_x_sizes = [32, 64, 128, 256, 512]
        tensorcore_y_sizes = [1, 2]
        # For tensorcore, workgroup Z is always 1
        tensorcore_z_sizes = [1]
        workgroup_sizes = list(product(
            tensorcore_x_sizes, tensorcore_y_sizes, tensorcore_z_sizes))

        tensorcore_x_sizes = [32, 64, 128, 256]
        tensorcore_y_sizes = [4]
        workgroup_sizes2 = list(product(
            tensorcore_x_sizes, tensorcore_y_sizes, tensorcore_z_sizes))
        workgroup_sizes.extend(workgroup_sizes2)

    return workgroup_sizes


def generate_pipeline_depth(pipeline: Pipeline, input_shape: List[int], tile_size: List[int]) -> List[int]:
    """"Returns list of possible pipeline depth"""
    if pipeline != Pipeline.GPU_TENSORCORE:
        return []

    # For tensorcore, usually between 1 and 12, increments of 1
    return [x for x in range(1, 6)]


def cuda_tensorcore_verify(dispatch: Dispatch, config: DispatchConfig) -> bool:
    """"Verifies the CUDA config on Tensorcor. Returns true if pass."""
    # Fixed constants
    dim_x = 0
    dim_y = 1
    dim_z = 2
    dim_m = 0
    dim_n = 1
    dim_k = 2
    warp_size = 32

    iree_tensorcore_shape = None
    if dispatch.data_type == DataType.F16:
        iree_tensorcore_shape = [16, 16, 16]
    elif dispatch.data_type == DataType.F32:
        iree_tensorcore_shape = [16, 16, 8]
    else:
        raise RuntimeError("Unsupported tensorcore shape")

    input_shape = [dispatch.m, dispatch.n, dispatch.k]
    thread_block_shape = [config.tile_size[0],
                          config.tile_size[1],
                          config.tile_size[2]]
    if dispatch.b:
        input_shape.insert(0, dispatch.b)
        # Remove the batch dimension from the thread_block_shape
        thread_block_shape = [config.tile_size[1],
                              config.tile_size[2],
                              config.tile_size[3]]

    # Total workgroup size > 1024?
    if reduce(mul, config.workgroup_size) > 1024:
        return False

    # Number of threads in z-dim is 1
    if config.workgroup_size[dim_z] != 1:
        return False

    # x-dim multiple of warp size (32)
    if config.workgroup_size[dim_x] % warp_size != 0:
        return False

    #  Number of warps in x, y, and z dim.
    num_warps = [config.workgroup_size[dim_x] / warp_size,
                 config.workgroup_size[dim_y],
                 config.workgroup_size[dim_z]]

    # Matrix-multiply problem shape in number of elements in M, N, and K dim.
    matmul_shape = [dispatch.m,
                    dispatch.n,
                    dispatch.k]

    # Warp tile shape in number of elements in M, N, and K dim.
    # Note that num warp in (x, y, z) dim are mapped to problem (M, N, K) dim as:
    # DimY -> ProblemDimM, DimX -> ProblemDimN, DimZ -> ProblemDimK.
    warp_shape = [thread_block_shape[dim_m] / num_warps[dim_y],
                  thread_block_shape[dim_n] / num_warps[dim_x],
                  thread_block_shape[dim_k] / num_warps[dim_z]]

    # Verify that matmul problem shape can be tiled with the thread block shape.
    if matmul_shape[dim_m] % thread_block_shape[dim_m] != 0 or matmul_shape[dim_n] % thread_block_shape[dim_n] != 0 or matmul_shape[dim_k] % thread_block_shape[dim_k] != 0:
        return False

    # Verify that if warp shape can be tiled using warp-level Tensor core
    # instruction shape.
    if warp_shape[dim_m] % iree_tensorcore_shape[dim_m] != 0 or warp_shape[dim_n] % iree_tensorcore_shape[dim_n] != 0 or warp_shape[dim_k] % iree_tensorcore_shape[dim_k] != 0:
        return False

    # Ensure shared memory usage <=163KB (limit for A100)
    shared_mem_bytes = 163 * 1024
    # Shared memory usage is determined by the two input operands to the matmul times the pipelined length
    lhs_mem_bytes = thread_block_shape[dim_m] * \
        thread_block_shape[dim_k] * dispatch.data_type.bytes_size
    rhs_mem_bytes = thread_block_shape[dim_n] * \
        thread_block_shape[dim_k] * dispatch.data_type.bytes_size
    matmul_mem_bytes = (lhs_mem_bytes + rhs_mem_bytes) * config.pipeline_depth
    if matmul_mem_bytes > shared_mem_bytes:
        return False

    # # Can only software pipeline if tile size is smaller than K
    # if len(input_shape) == 4:
    #     if tile_size[3] == input_shape[2]:
    #         return [1]
    # if len(input_shape) == 3:
    #     if tile_size[2] == input_shape[1]:
    #         return [1]

    return True

# General Configs
###################################################################################################


def generate_cuda_configs(dispatch: Dispatch) -> List[DispatchConfig]:
    """Generates configs for CUDA.
    """
    if dispatch.pipeline_name != Pipeline.GPU_TENSORCORE:
        raise RuntimeError("Only GPU Tensorcore supported for configs.")
    configs = []
    pipeline = dispatch.pipeline_name
    operation = dispatch.operation
    data_type = dispatch.data_type
    input_shape = [dispatch.m, dispatch.n, dispatch.k]
    if dispatch.b:
        input_shape.insert(0, dispatch.b)
    tile_sizes = generate_tile_sizes(pipeline, data_type, input_shape)

    for tile_size in tile_sizes:
        workgroup_sizes = generate_workgroup_sizes(
            pipeline, input_shape, tile_size)
        pipeline_depths = generate_pipeline_depth(
            pipeline, input_shape, tile_size)

        # Create a config for each combination of tile, workgroup and pipeline
        for workgroup_size in workgroup_sizes:
            for pipeline_depth in pipeline_depths:
                dispatch_config = DispatchConfig(
                    pipeline_name=pipeline,
                    operation=operation,
                    tile_size=tile_size,
                    workgroup_size=workgroup_size,
                    pipeline_depth=pipeline_depth,
                )
                if cuda_tensorcore_verify(dispatch, dispatch_config):
                    configs.append(dispatch_config)

    return configs


def generate_configs(target_backend: TargetBackend, dispatch: Dispatch) -> List[DispatchConfig]:
    """Generates configs compatible with the target backend and dispatch.
    """
    if target_backend == TargetBackend.CUDA:
        return generate_cuda_configs(dispatch)
    else:
        raise RuntimeError("Only configs for CUDA supported.")
