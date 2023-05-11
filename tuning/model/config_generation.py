import sys
from pathlib import Path
import json
from itertools import product
from functools import reduce
from operator import mul

from typing import List, Optional
from utils.data_types import Pipeline, OperationType, DataType, DispatchConfig, Dispatch, TargetBackend

###################################################################################################
# This file contains library for producing configs to annotate mlir models.
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

    # Can only software pipeline if tile size is smaller than K
    if len(input_shape) == 4:
        if tile_size[3] == input_shape[2]:
            return [1]
    if len(input_shape) == 3:
        if tile_size[2] == input_shape[1]:
            return [1]
    # For tensorcore, usually between 1 and 12, increments of 1
    return [x for x in range(1, 6)]

def cuda_tensorcore_verify(dispatch: Dispatch, config: DispatchConfig) -> bool:
    """"Verifies the CUDA config on Tensorcor. Returns true if pass."""
    input_shape = [dispatch.m, dispatch.n, dispatch.k]
    if dispatch.b:
        input_shape.insert(0, dispatch.b)

    # Toss any config that does not divide the input shape
    def divides_shape(tile_size: List[int]):
        for tile, shape in zip(tile_size, input_shape):
            if shape % tile != 0:
                return False
        return True
    if not divides_shape(config.tile_size):
        return False

    # Ensure shared memory usage <=164KB
    shared_mem_bytes = 163 * 1024
    def fits_shared_mem(tile_size: List[int]):
        if len(tile_size) == 4:
            tile_size = tile_size[1:]
        total_shared_mem_size = (
            tile_size[0] * tile_size[2] + tile_size[1] * tile_size[2]) * dispatch.data_type.bytes_size
        return total_shared_mem_size < shared_mem_bytes
    if not fits_shared_mem(config.tile_size):
        return False
    
    # Total workgroup size < 1024
    if reduce(mul, config.workgroup_size) > 1024:
        return False

    # Only use if second level tiling size divides by tensorcore
    iree_tensorcore_size = [16, 16, 8]
    warp_size = 32

    def divides_tensorcore(tile_size, workgroup_size):
        if len(tile_size) == 4:
            tile_size = tile_size[1:]
        second_level_tile = [tile_size[0] / workgroup_size[1],
                                tile_size[1] / (workgroup_size[0] / warp_size), tile_size[2]]
        # print(second_level_tile, workgroup_size)
        return second_level_tile[0] % iree_tensorcore_size[0] == 0 and second_level_tile[1] % iree_tensorcore_size[1] == 0 and second_level_tile[2] % iree_tensorcore_size[2] == 0
    if not divides_tensorcore(config.tile_size, config.workgroup_size):
        return False

    return True


def generate_cuda_configs(dispatch: Dispatch)  -> List[DispatchConfig]:
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



def main():
    print("config_generation.py")
    configs = generate_configs(pipeline=Pipeline.GPU_TENSORCORE, operation=OperationType.MATMUL, input_shape=[
                               4096, 3072, 768], data_type=DataType.F32)
    print(f"Generated config count: {len(configs)}")


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
