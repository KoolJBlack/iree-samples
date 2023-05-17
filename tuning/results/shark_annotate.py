from typing import Optional, List, Tuple
import json

from utils.data_types import Pipeline, OperationType

def assemble_shark_config_object(
    tile_size: List[int],
    workgroup_size: List[int],
    pipeline : Pipeline,
    pipeline_depth: Optional[int],
    operation: OperationType,
    b: Optional[int],
    m: int,
    n: int,
    k: int) -> str:
    """Returns a shark config as json string-."""

    if operation == OperationType.MATMUL:
        identifier = "matmul"
    elif operation == OperationType.BATCH_MATMUL:
        identifier = "bmm"
    else: 
        raise RuntimeError("undefined operation")

    if pipeline == Pipeline.GPU_TENSORCORE:
        pipeline_name = "GPU_TENSORCORE"
    elif operation == Pipeline.GPU_SIMT:
        pipeline = "GPU_SIMT"
    else: 
        raise RuntimeError("undefined pipeline")

    config_object = dict()
    config_options = dict()
    # Options sub object
    config_options["work_group_tile_sizes"] = tile_size
    config_options["work_group_sizes"] = workgroup_size
    config_options["pipeline"] = pipeline_name
    if pipeline_depth:
        config_options["pipeline_depth"] = pipeline_depth
    config_object["options"] = [config_options]
    config_object["identifier"] = identifier
    if b:
        config_object['b'] = b
    config_object['m'] = m
    config_object['n'] = n
    config_object['k'] = k

    return json.dumps(config_object)
