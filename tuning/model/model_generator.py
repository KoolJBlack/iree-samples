from typing import Optional, List, Tuple
from utils.data_types import OperationType, Dispatch, DispatchConfig

###################################################################################################
# This file contains library for generating mlir models for a given operation and config.
###################################################################################################

COMPILATION_INFO_TAG = "{compilation_info = #compilation_info}"

COMPILATION_INFO_TEMPLATE = """#compilation_info = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [{tile_size}]>,
    translation_info = <{pipeline} pipeline_depth = {pipeline_depth}>, 
    workgroup_size = [{workgroup_x} : index, {workgroup_y} : index, {workgroup_z} : index]
>
"""

MATMUL_TEMPLATE = """{compilation_info}
func.func @benchmark_matmul() -> tensor<{m}x{n}x{data_type}> {{
    %cst = arith.constant 1.0 : {data_type}
    %lhs = arith.constant dense<1.0> : tensor<{m}x{k}x{data_type}>
    %rhs = arith.constant dense<1.0> : tensor<{k}x{n}x{data_type}>
    %empty = tensor.empty() : tensor<{m}x{n}x{data_type}>
    %filled = linalg.fill ins(%cst : {data_type}) outs(%empty : tensor<{m}x{n}x{data_type}>) -> tensor<{m}x{n}x{data_type}>
    %result = linalg.matmul {compilation_info_tag} ins(%lhs, %rhs : tensor<{m}x{k}x{data_type}>, tensor<{k}x{n}x{data_type}>) outs(%filled : tensor<{m}x{n}x{data_type}>) -> tensor<{m}x{n}x{data_type}>
    return %result : tensor<{m}x{n}x{data_type}>
}}
"""

BATCH_MATMUL_TEMPLATE = """{compilation_info}
func.func @benchmark_batch_matmul() -> tensor<{b}x{m}x{n}x{data_type}> {{
    %cst = arith.constant 1.0 : {data_type}
    %lhs = arith.constant dense<1.0> : tensor<{b}x{m}x{k}x{data_type}>
    %rhs = arith.constant dense<1.0> : tensor<{b}x{k}x{n}x{data_type}>
    %empty = tensor.empty() : tensor<{b}x{m}x{n}x{data_type}>
    %filled = linalg.fill ins(%cst : {data_type}) outs(%empty : tensor<{b}x{m}x{n}x{data_type}>) -> tensor<{b}x{m}x{n}x{data_type}>
    %result = linalg.batch_matmul {compilation_info_tag} ins(%lhs, %rhs : tensor<{b}x{m}x{k}x{data_type}>, tensor<{b}x{k}x{n}x{data_type}>) outs(%filled : tensor<{b}x{m}x{n}x{data_type}>) -> tensor<{b}x{m}x{n}x{data_type}>
    return %result : tensor<{b}x{m}x{n}x{data_type}>
}}
"""

def generate_model(dispatch: Dispatch, config: Optional[DispatchConfig]):
    """Produces the MLIR model based on Dispatch information. 
    If optional DispatchConfig is provided, the model is annotated wit config. 
    """
    compilation_info = ""
    compilation_info_tag = ""
    if config:
        compilation_info = COMPILATION_INFO_TEMPLATE.format(
            pipeline=dispatch.pipeline_name.value,
            tile_size=list(config.tile_size),
            pipeline_depth=config.pipeline_depth,
            workgroup_x=config.workgroup_size[0],
            workgroup_y=config.workgroup_size[1],
            workgroup_z=config.workgroup_size[2],
        )
        compilation_info_tag = COMPILATION_INFO_TAG
    if dispatch.operation == OperationType.MATMUL:
        model_text = MATMUL_TEMPLATE.format(
            compilation_info=compilation_info,
            compilation_info_tag=compilation_info_tag,
            data_type=dispatch.data_type.iree_type,
            m=dispatch.m,
            n=dispatch.n,
            k=dispatch.k,
        )
    elif dispatch.operation == OperationType.BATCH_MATMUL:
        model_text = BATCH_MATMUL_TEMPLATE.format(
            compilation_info=compilation_info,
            compilation_info_tag=compilation_info_tag,
            data_type=dispatch.data_type.iree_type,
            b=dispatch.b,
            m=dispatch.m,
            n=dispatch.n,
            k=dispatch.k,
        )
    else:
        raise RuntimeError("Unknown operation type: " + dispatch.operation)

    return model_text
