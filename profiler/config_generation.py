#!/usr/bin/env python3

import sys
from pathlib import Path
import json
import enum
from itertools import product
from functools import reduce
from operator import mul, mod

from typing import List, Optional


@enum.unique
class Pipeline(enum.Enum):
    GPU_TENSORCORE = "GPU_TENSORCORE"
    GPU_SIMT = "GPU"

    def __str__(self):
        return self.name


@enum.unique
class OperationType(enum.Enum):
    MATMUL = "matmul"
    BATCH_MATMUL = "bmm"
    GENERIC = "generic"


@enum.unique
class DataType(enum.Enum):
    I8 = "i8"
    I32 = "i32"
    F32 = "f32"
    F16 = "f16"

    @staticmethod
    def from_string(s: str):
        try:
            return DataType[s]
        except KeyError:
            raise ValueError()


def generate_tile_sizes(pipeline: Pipeline, input_shape: List[int]) -> List[List[int]]:
    """"Returns list of possible tile sizes for input shape and pipeline"""
    tile_sizes = []

    if pipeline == Pipeline.GPU_TENSORCORE:
        # Tensorcor main dims M, N, K
        tensorcore_mn_sizes = [16, 32, 64, 128, 256, 512]
        tensorcore_k_sizes = [16, 32, 64, 128]
        tile_sizes = list(product(tensorcore_mn_sizes,
                                  tensorcore_mn_sizes, tensorcore_k_sizes))

        # K dim is always smaller than the picked M or N
        tile_sizes = [tile_size for tile_size in tile_sizes if tile_size[2]
                      <= tile_size[0] and tile_size[2] <= tile_size[1]]

    else:
        # Todo: SIMT
        # Simt
        simt_sizes = [1, 2, 4, 8]

    # Batch dim tile is always 1 if it exists
    if len(input_shape) == 4:
        for tile_size in tile_sizes:
            tile_size = [1] + tile_size

    # Toss any config that does not divide the input shape

    def divides_shape(input_shape, tile_size):
        for tile, shape in zip(tile_size, input_shape):
            if shape % tile != 0:
                return False
        return True
    tile_sizes = [tile_size for tile_size in tile_sizes if divides_shape(
        input_shape, tile_size)]

    return tile_sizes


def generate_workgroup_sizes(pipeline: Pipeline, input_shape: List[int], tile_size: List[int]) -> List[List[int]]:
    """"Returns list of possible workgroup sizes for tile size"""
    workgroup_sizes = []

    # print("generate_workgroup_sizes")

    if pipeline == Pipeline.GPU_TENSORCORE:
        # Tensorcore main dims X, Y, Z
        tensorcore_x_sizes = [32, 64, 128, 256, 512]
        tensorcore_y_sizes = [1, 2, 4, 8]
        # For tensorcore, workgroup Z is always 1
        tensorcore_z_sizes = [1]
        workgroup_sizes = list(product(
            tensorcore_x_sizes, tensorcore_y_sizes, tensorcore_z_sizes))

        # Total workgroup size < 1024
        workgroup_sizes = [workgroup_size for workgroup_size in workgroup_sizes if not reduce(
            mul, workgroup_size) > 1024]

        # Only use if second level tiling size divides by tensorcore
        iree_tensorcore_size = [16, 16, 8]
        warp_size = 32

        def divides_tensorcore(tile_size, workgroup_size):
            second_level_tile = [tile_size[0] / workgroup_size[1],
                                 tile_size[1] / (workgroup_size[0] / warp_size), tile_size[2]]
            # print(second_level_tile, workgroup_size)
            return second_level_tile[0] % iree_tensorcore_size[0] == 0 and second_level_tile[1] % iree_tensorcore_size[1] == 0 and second_level_tile[2] % iree_tensorcore_size[2] == 0
        workgroup_sizes = [
            workgroup_size for workgroup_size in workgroup_sizes if divides_tensorcore(tile_size, workgroup_size)]
    else:
        # Todo: SIMT
        pass

    return workgroup_sizes


def generate_pipeline_depth(pipeline: Pipeline, input_shape: List[int], tile_size: List[int]) -> List[int]:
    """"Returns list of possible pipeline depth"""
    if pipeline != Pipeline.GPU_TENSORCORE:
        return []
    # Can only software pipeline if tile size is smaller than K
    if tile_size[2] == input_shape[1]:
        return []
    # For tensorcore, usually between 1 and 12, increments of 1
    return [x for x in range(1, 13)]


def assemble_config_object(
    tile_size: List[int],
    workgroup_size: List[int],
    pipeline_name: str = "GPU_TENSORCORE",
    pipeline_depth: Optional[int] = 4,
    identifier: str = "matmul",
    b: Optional[int] = None,
    m: int = 4096,
    n: int = 3072,
        k: int = 768) -> dict:
    """Returns a shark config as a dict object."""
    config_object = dict()
    config_options = dict()
    # Options sub object
    config_options["work_group_tile_sizes"] = tile_size
    config_options["work_group_sizes"] = workgroup_size
    config_options["pipeline"] = pipeline_name
    if pipeline_name == "GPU_TENSORCORE" and pipeline_depth:
        config_options["pipeline_depth"] = pipeline_depth

    config_object["options"] = [config_options]
    config_object["identifier"] = identifier
    if b:
        config_object['b'] = b
    config_object['m'] = m
    config_object['n'] = n
    config_object['k'] = k

    return config_object


"""A blank control config that does not annotate the model."""
CONTROL_CONFIG = {
    "options": [{
        "work_group_tile_sizes": "(control)",
        "work_group_sizes": "(control)",
        "pipeline": "control",
    }],
    "identifier": "control",
    "m": "control",
    "n": "control",
    "k": "control",
}


def dump_shark_config_json(config: dict, output_path: Path):
    """Writes out a shark config dict object to a json file.
    Returns number of bytes written."""
    with open(output_path, "w") as f:
        return f.write(json.dumps(config))


def generate_configs(pipeline: Pipeline, operation: OperationType, input_shape: List[int], data_type: DataType, m: int = 4096, n: int = 3072, k: int = 768) -> List[dict]:
    """Generates a list of configs based on options.

    Configs are returned asa list of dictionaries. Each config can be used to annotate model using sharks model_annotation.py
    """

    input_shape = [m, n, k]
    configs = []
    tile_sizes = generate_tile_sizes(pipeline, input_shape)

    for tile_size in tile_sizes:
        workgroup_sizes = generate_workgroup_sizes(
            pipeline, input_shape, tile_size)
        pipeline_depths = generate_pipeline_depth(
            pipeline, input_shape, tile_size)

        # Create a config for each combination of tile, workgroup and pipeline
        for workgroup_size in workgroup_sizes:
            for pipeline_depth in pipeline_depths:
                b = None
                if len(input_shape) == 4:
                    b = input_shape[0]
                    input_shape = input_shape[1:]
                config_dict = assemble_config_object(
                    tile_size=tile_size,
                    workgroup_size=workgroup_size,
                    pipeline_name=pipeline.name,
                    pipeline_depth=pipeline_depth,
                    identifier="matmul",
                    b=b,
                    m=input_shape[0],
                    n=input_shape[1],
                    k=input_shape[2]
                )
                configs.append(config_dict)
    return configs


def main():
    print("config_generation.py")
    configs = generate_configs(pipeline=Pipeline.GPU_TENSORCORE, operation=OperationType.MATMUL, input_shape=[
                               4096, 3072, 768], data_type=DataType.F32)
    print(f"Generated config count: {len(configs)}")


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
