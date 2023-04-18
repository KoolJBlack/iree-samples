#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path
import json

from typing import List, Optional

## ====== Constants ======
IREE_DEV_PATH = Path(os.getenv('IREE_DEV'))
IREE_BUILD_PATH = Path(os.getenv('IREE_BUILD'))
IREE_DATA_PATH = Path(os.getenv('IREE_DATA'))
IREE_TMP_PATH = Path(os.getenv('IREE_TMP'))

REMOTE_IREE_DEV_PATH=Path("/home/kooljblack/Code/iree")
REMOTE_IREE_BUILD_PATH=Path("/home/kooljblack/Code/iree-build")
REMOTE_IREE_DATA_PATH=Path("/home/kooljblack/Code/iree-data")
REMOTE_IREE_TMP_PATH=Path("/home/kooljblack/Code/iree-tmp")
IREE_TOOLS_PATH = Path("/usr/local/google/home/kooljblack/Code/iree-tools/")

BATCH_SIZE_TMP_PATH = IREE_TMP_PATH.joinpath("batch_size/search")
BATCH_SIZE_TOOLS_PATH = IREE_TOOLS_PATH.joinpath("batch_size")


## ====== Json Loading + Saving ======


def load_model_configs(config_path: str):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            data = json.loads(line)

            if "identifier" not in data.keys():
                continue
            if data["identifier"] == "matmul":
                matrix_size = [data["m"], data["n"], data["k"]]
            elif data["identifier"] == "bmm":
                matrix_size = [data["b"], data["m"], data["n"], data["k"]]
            elif data["identifier"] == "generic":
                matrix_size = [1, data["b"], data["m"], data["n"], data["k"]]
            elif data["identifier"] == "conv":
                matrix_size = [
                    data["n"],
                    data["ih"],
                    data["iw"],
                    data["c"],
                    data["kh"],
                    data["kw"],
                    data["f"],
                    data["oh"],
                    data["ow"],
                    data["d"],
                    data["s"],
                    data["p"],
                ]
            config[shape_list_to_string(matrix_size)] = data
        f.close()
        return config

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
    config_object = dict()
    config_options = dict()
    # Options sub object
    config_options["work_group_tile_sizes"] = tile_size
    config_options["work_group_sizes"] = workgroup_size
    config_options["pipeline"] = pipeline_name
    if pipeline_name == "GPU_TENSORCORE" and pipeline_depth:
        config_options["pipeline_depth"] = pipeline_depth
    
    config_object["options"] =  [config_options]
    config_object["identifier"] = identifier
    if b:
        config_object['b'] = b
    config_object['m'] = m
    config_object['n'] = n
    config_object['k'] = k

    return config_object

def dump_shark_config_json(file_path : str):
    pass
    result = json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
    print(f"{type(result)}")

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            print(data.keys())
    
    # Test creating a mock config
    tile_size = [1, 32, 128]
    workgroup_size = [32, 1, 1]
    config_object = assemble_config_object(tile_size, workgroup_size)
    print(f"Json formatted string: {json.dumps(config_object)}")


def main():
  print("hello world")
  test_config_path = BATCH_SIZE_TOOLS_PATH.joinpath("config.json")
  dump_shark_config_json(test_config_path)

if __name__ == '__main__':
  sys.exit(main())  # next section explains the use of sys.exit
