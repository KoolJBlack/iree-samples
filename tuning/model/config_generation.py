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

    if pipeline == Pipeline.GPU_TENSORCORE or pipeline == Pipeline.GPU_TENSORCORE_MMASYNC:
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

    if pipeline == Pipeline.GPU_TENSORCORE or pipeline == Pipeline.GPU_TENSORCORE_MMASYNC:
        # Tensorcore main dims X, Y, Z
        def to_threads(warp):
            return warp * 32
        # tensorcore_x_sizes = list(map(to_threads, [1, 2, 4, 8, 16]))
        tensorcore_x_sizes = list(map(to_threads, [1, 2, 4, 8]))
        tensorcore_y_sizes = [1, 2, 4]
        # For tensorcore, workgroup Z is always 1
        tensorcore_z_sizes = [1]
        workgroup_sizes = list(product(
            tensorcore_x_sizes, tensorcore_y_sizes, tensorcore_z_sizes))

    return workgroup_sizes


def generate_pipeline_depth(pipeline: Pipeline, input_shape: List[int], tile_size: List[int]) -> List[int]:
    """"Returns list of possible pipeline depth"""
    if pipeline == Pipeline.GPU_TENSORCORE:
        # For tensorcore, usually between 1 and 12, increments of 1
        return [x for x in range(2, 6)]
    elif pipeline == Pipeline.GPU_TENSORCORE_MMASYNC:
        return [x for x in range(3, 6)]
    return []

def cuda_tensorcore_verify(dispatch: Dispatch, config: DispatchConfig) -> bool:
    """"Verifies the CUDA config on Tensorcore, based on Verifiers.cpp. Returns true if pass."""
    # Fixed constants
    dim_x = 0
    dim_y = 1
    dim_z = 2
    dim_m = 0
    dim_n = 1
    dim_k = 2
    warp_size = 32

    iree_tensorcore_shape = None
    if dispatch.pipeline_name == Pipeline.GPU_TENSORCORE:
        if dispatch.data_type == DataType.F16:
            iree_tensorcore_shape = [16, 16, 16]
        elif dispatch.data_type == DataType.F32:
            iree_tensorcore_shape = [16, 16, 8]
    if dispatch.pipeline_name == Pipeline.GPU_TENSORCORE_MMASYNC:
        if dispatch.data_type == DataType.F16:
            iree_tensorcore_shape = [16, 8, 16]
        elif dispatch.data_type == DataType.F32:
            iree_tensorcore_shape = [16, 8, 8]  
    if not iree_tensorcore_shape:
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
    # if (thread_block_shape[dim_m] + thread_block_shape[dim_n]) * thread_block_shape[dim_k] * config.pipeline_depth * dispatch.data_type.bytes_size > shared_mem_bytes:
    #     return False

    return True

def cuda_tensorcore_prune(dispatch: Dispatch, config: DispatchConfig) -> bool:
    """"A pruning pass to remove configs that aren't expected to be performant."""
    # if dispatch.pipeline_name == Pipeline.GPU_TENSORCORE_MMASYNC:
    #     num_warps = reduce(mul, config.workgroup_size) / 32
    #     if num_warps < 4: 
    #         return False
    return True

# General Configs
###################################################################################################


def generate_cuda_configs(dispatch: Dispatch) -> List[DispatchConfig]:
    """Generates configs for CUDA.
    """
    if dispatch.pipeline_name != Pipeline.GPU_TENSORCORE and dispatch.pipeline_name != Pipeline.GPU_TENSORCORE_MMASYNC:
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
                    tile_size=tile_size,
                    workgroup_size=workgroup_size,
                    pipeline_depth=pipeline_depth,
                )
                configs.append(dispatch_config)
    # configs = [
    #     DispatchConfig(
    #         Pipeline.GPU_TENSORCORE_MMASYNC,
    #         OperationType.MATMUL,
    #         tile_size=[128, 256, 16],
    #         workgroup_size=[128, 2, 1],
    #         pipeline_depth=4 
    #         )
    # ]
 
    configs = []

    # CUTLASS f32 configs: https://github.com/NVIDIA/cutlass/blob/main/tools/library/scripts/generator.py#L2544
    configs.extend([ 
      DispatchConfig(tile_size=[256, 128, 16], pipeline_depth= 3, workgroup_size=[4, 2, 1]),
      DispatchConfig(tile_size=[128, 256, 16], pipeline_depth= 3, workgroup_size=[2, 4, 1]),
      DispatchConfig(tile_size=[256,  64, 16], pipeline_depth= 4, workgroup_size=[4, 1, 1]),
      DispatchConfig(tile_size=[ 64, 256, 16], pipeline_depth= 4, workgroup_size=[1, 4, 1]),
      DispatchConfig(tile_size=[128, 128, 16], pipeline_depth= 5, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128, 128, 16], pipeline_depth= 4, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128, 128, 16], pipeline_depth= 3, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128,  64, 16], pipeline_depth= 6, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[ 64, 128, 16], pipeline_depth= 6, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[ 64,  64, 16], pipeline_depth=10, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[256, 128, 32], pipeline_depth= 3, workgroup_size=[4, 2, 1]),
      DispatchConfig(tile_size=[128, 256, 32], pipeline_depth= 3, workgroup_size=[2, 4, 1]),
      DispatchConfig(tile_size=[256,  64, 32], pipeline_depth= 4, workgroup_size=[4, 1, 1]),
      DispatchConfig(tile_size=[ 64, 256, 32], pipeline_depth= 4, workgroup_size=[1, 4, 1]),
      DispatchConfig(tile_size=[128, 128, 32], pipeline_depth= 4, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128, 128, 32], pipeline_depth= 3, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128,  64, 32], pipeline_depth= 3, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[64,  128, 32], pipeline_depth= 3, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[ 64,  64, 32], pipeline_depth= 5, workgroup_size=[2, 2, 1]),
    ])

    # CUTLASS f16 configs: https://github.com/NVIDIA/cutlass/blob/main/tools/library/scripts/generator.py#L1908
    configs.extend([ 
      DispatchConfig(tile_size=[256, 128, 32], pipeline_depth= 3, workgroup_size=[4, 2, 1]),
      DispatchConfig(tile_size=[128, 256, 32], pipeline_depth= 3, workgroup_size=[2, 4, 1]),
      DispatchConfig(tile_size=[256,  64, 32], pipeline_depth= 3, workgroup_size=[4, 1, 1]),
      DispatchConfig(tile_size=[256,  64, 32], pipeline_depth= 4, workgroup_size=[4, 1, 1]),
      DispatchConfig(tile_size=[ 64, 256, 32], pipeline_depth= 4, workgroup_size=[1, 4, 1]),
      DispatchConfig(tile_size=[128, 128, 32], pipeline_depth= 3, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128, 128, 32], pipeline_depth= 4, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128, 128, 32], pipeline_depth= 5, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128,  64, 32], pipeline_depth= 6, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[ 64, 128, 32], pipeline_depth= 6, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[ 64,  64, 32], pipeline_depth=10, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[256, 128, 64], pipeline_depth= 3, workgroup_size=[4, 2, 1]),
      DispatchConfig(tile_size=[128, 256, 64], pipeline_depth= 3, workgroup_size=[2, 4, 1]),
      DispatchConfig(tile_size=[256,  64, 64], pipeline_depth= 4, workgroup_size=[4, 1, 1]),
      DispatchConfig(tile_size=[ 64, 256, 64], pipeline_depth= 4, workgroup_size=[1, 4, 1]),
      DispatchConfig(tile_size=[128, 128, 64], pipeline_depth= 4, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[256,  64, 64], pipeline_depth= 3, workgroup_size=[4, 1, 1]),
      DispatchConfig(tile_size=[ 64, 256, 64], pipeline_depth= 3, workgroup_size=[1, 4, 1]),
      DispatchConfig(tile_size=[128, 128, 64], pipeline_depth= 3, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[128,  64, 64], pipeline_depth= 3, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[ 64, 128, 64], pipeline_depth= 3, workgroup_size=[2, 2, 1]),
      DispatchConfig(tile_size=[ 64,  64, 64], pipeline_depth= 5, workgroup_size=[2, 2, 1]),
    ])

    # Triton configs: https://github.com/openai/triton/blob/b5d32896b1f89fc44a82f8df3bb010934c53f4f5/python/triton/ops/matmul.py#L29
    configs.extend([
        DispatchConfig([128, 256, 32], pipeline_depth=3,  workgroup_size=[8, 1, 1]),
        DispatchConfig([256, 128, 32], pipeline_depth=3,  workgroup_size=[8, 1, 1]),
        DispatchConfig([256, 64, 32], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([64, 256, 32], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([128, 128, 32], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([128, 64, 32], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([64, 128, 32], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([128, 32, 32], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([64, 32, 32], pipeline_depth=5,  workgroup_size=[2, 1, 1]),
        # good for int8
        DispatchConfig([128, 256, 128], pipeline_depth=3,  workgroup_size=[8, 1, 1]),
        DispatchConfig([256, 128, 128], pipeline_depth=3,  workgroup_size=[8, 1, 1]),
        DispatchConfig([256, 64, 128], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([64, 256, 128], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([128, 128, 128], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([128, 64, 64], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([64, 128, 64], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([128, 32, 64], pipeline_depth=4,  workgroup_size=[4, 1, 1]),
        DispatchConfig([64, 32, 64], pipeline_depth=5,  workgroup_size=[2, 1, 1]),
    ])

    # For batch matmul, update tile size.
    if dispatch.b:
        for config in configs:
            config.tile_size.insert(0, 1)

    print(f"Total custom configs: {len(configs)}")
    unique_configs_set = set(configs)
    configs = list(unique_configs_set)
    print(f"Unique custom configs (before filter): {len(configs)}")

    configs = [dispatch_config for dispatch_config in configs if cuda_tensorcore_verify(dispatch, dispatch_config)]
    configs = [dispatch_config for dispatch_config in configs if cuda_tensorcore_prune(dispatch, dispatch_config)]
    return configs


def generate_configs(target_backend: TargetBackend, dispatch: Dispatch) -> List[DispatchConfig]:
    """Generates configs compatible with the target backend and dispatch.
    """
    if target_backend == TargetBackend.CUDA:
        return generate_cuda_configs(dispatch)
    else:
        raise RuntimeError("Only configs for CUDA supported.")
