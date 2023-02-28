#!/usr/bin/env python3

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import re
import enum
from dataclasses import dataclass
import json

from config_generation import Pipeline, OperationType, DataType, generate_configs, CONTROL_CONFIG
from profiler import run_profile, TargetBackend


def dir_path(string) -> Optional[Path]:
    """Returns path to dir if it exists"""
    if os.path.isdir(string):
        return Path(string)
    else:
        raise NotADirectoryError(string)
        # return None


@enum.unique
class Operation(enum.Enum):
    MATMUL = "matmul"
    BATCH_MATMUL = "batch_matmul"
    GENERIC = "generic"


@dataclass
class ProfilerProgram:
    """Defines a run for the profiler and results to collect."""
    name: str
    b: int
    m: int
    n: int
    k: int
    data_type: DataType
    operation_type: OperationType
    target_backend: TargetBackend
    pipeline: Pipeline
    template_mlir_filename: str
    output_csv_filename: str

    def dump_json(self) -> str:
        json_dict = {
            "name": self.name,
            "b": self.b,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "data_type": self.data_type.iree_type,
            "operation_type": self.operation_type.value,
            "target_backend": self.target_backend.value,
            "pipeline": self.pipeline.value,
            "template_mlir_filename": self.template_mlir_filename,
            "output_csv_filename": self.output_csv_filename

        }
        return json.dumps(json_dict)

    @classmethod
    def load_json(cls, json_str: str):
        json_dict = json.loads(json_str)

        data_type = None
        if json_dict["data_type"] == DataType.I8.iree_type:
            data_type = DataType.I8
        if json_dict["data_type"] == DataType.F16.iree_type:
            data_type = DataType.F16
        if json_dict["data_type"] == DataType.I32.iree_type:
            data_type = DataType.I32
        if json_dict["data_type"] == DataType.F32.iree_type:
            data_type = DataType.F32

        # operation = None
        # if json_dict["operation"] == Operation.MATMUL.value:
        #     operation = Operation.MATMUL
        # if json_dict["operation"] == Operation.BATCH_MATMUL.value:
        #     operation = Operation.BATCH_MATMUL

        operation_type = None
        if json_dict["operation_type"] == OperationType.MATMUL.value:
            operation_type = OperationType.MATMUL
        if json_dict["operation_type"] == OperationType.BATCH_MATMUL.value:
            operation_type = OperationType.BATCH_MATMUL

        target_backend = TargetBackend.CUDA
        if json_dict["target_backend"] != TargetBackend.CUDA.value:
            raise ValueError("Only target backend CUDA supported")

        pipeline = None
        if json_dict["pipeline"] == Pipeline.GPU_TENSORCORE.value:
            pipeline = Pipeline.GPU_TENSORCORE
        if json_dict["pipeline"] == Pipeline.GPU_SIMT.value:
            pipeline = Pipeline.GPU_SIMT

        return ProfilerProgram(
            name=json_dict["name"],
            b=json_dict["b"],
            m=json_dict["m"],
            n=json_dict["n"],
            k=json_dict["k"],
            data_type=data_type,
            operation_type=operation_type,
            target_backend=target_backend,
            pipeline=pipeline,
            template_mlir_filename=json_dict["template_mlir_filename"],
            output_csv_filename=json_dict["output_csv_filename"],
        )


def generate_temp_file(
        b: int,
        m: int,
        n: int,
        k: int,
        data_type: DataType,
        operation: Operation,
        template_mlir_model_path: Path,
):
    """Spits out a template mlir model for given program params."""
    matmul_template = f"""
    func.func @benchmark_matmul_tensorcore() -> tensor<{m}x{n}x{data_type.iree_type}> {{
        %ins = arith.constant dense<1.0> : tensor<{m}x{k}x{data_type.iree_type}>
        %cst = arith.constant 0.707106769 : {data_type.iree_type}
        %cst_1 = arith.constant dense<1.0> : tensor<{k}x{n}x{data_type.iree_type}>
        %46 = tensor.empty() : tensor<{m}x{n}x{data_type.iree_type}>
        %47 = linalg.fill ins(%cst : {data_type.iree_type}) outs(%46 : tensor<{m}x{n}x{data_type.iree_type}>) -> tensor<{m}x{n}x{data_type.iree_type}>
        %48 = linalg.matmul ins(%ins, %cst_1 : tensor<{m}x{k}x{data_type.iree_type}>, tensor<{k}x{n}x{data_type.iree_type}>) outs(%47 : tensor<{m}x{n}x{data_type.iree_type}>) -> tensor<{m}x{n}x{data_type.iree_type}>
        return %48 : tensor<{m}x{n}x{data_type.iree_type}>
    }}
    """
    batch_matmul_template = f"""
    func.func @benchmark_batch_matmul_tensorcore() -> tensor<{b}x{m}x{n}x{data_type.iree_type}> {{
        %ins = arith.constant dense<1.0> : tensor<{b}x{m}x{k}x{data_type.iree_type}>
        %cst = arith.constant 0.707106769 : {data_type.iree_type}
        %cst_1 = arith.constant dense<1.0> : tensor<{b}x{k}x{n}x{data_type.iree_type}>
        %46 = tensor.empty() : tensor<{b}x{m}x{n}x{data_type.iree_type}>
        %47 = linalg.fill ins(%cst : {data_type.iree_type}) outs(%46 : tensor<{b}x{m}x{n}x{data_type.iree_type}>) -> tensor<{b}x{m}x{n}x{data_type.iree_type}>
        %48 = linalg.batch_matmul ins(%ins, %cst_1 : tensor<{b}x{m}x{k}x{data_type.iree_type}>, tensor<{b}x{k}x{n}x{data_type.iree_type}>) outs(%47 : tensor<{b}x{m}x{n}x{data_type.iree_type}>) -> tensor<{b}x{m}x{n}x{data_type.iree_type}>
        return %48 : tensor<{b}x{m}x{n}x{data_type.iree_type}>
    }}
    """
    selected_template = None
    if operation == OperationType.MATMUL:
        selected_template = matmul_template
    if operation == OperationType.BATCH_MATMUL:
        selected_template = batch_matmul_template

    print(f"Saving generated model to: {template_mlir_model_path}")
    with open(template_mlir_model_path, mode="w") as f:
        f.write(selected_template)


def dump_profile_programs(input_mlir_model: Path,
                          output_program_path: Path,
                          data_type: DataType = DataType.F32,
                          target_backend: TargetBackend = TargetBackend.CUDA,
                          pipeline: Pipeline = Pipeline.GPU_TENSORCORE):
    """Parse an mlir model dump after 'tile distribute workgroups pass'. Identifies all dispatches that match the given pipeline and dumps them to json file."""

    pipeline_marker = None
    if pipeline == Pipeline.GPU_TENSORCORE:
        pipeline_marker = "LLVMGPUMatmulTensorCore"
    if pipeline == Pipeline.GPU_SIMT:
        pipeline_marker = "LLVMGPUMatmulSimt"

    pipeline_lines = []
    lines_count = 0
    with open(input_mlir_model, mode="r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip()
            lines_count += 1
            if pipeline_marker in line:
                pipeline_lines.append(line)

    print(f"Analyzed input: {input_mlir_model} with {lines_count} lines.")
    print(
        f"Found {len(pipeline_lines)} dispatches with marker: {pipeline_marker}")

    # Regex match each dispatch name to get parameters
    pattern = re.compile(
        "@forward_dispatch_(?P<dispatch_index>\d+)_(?P<operation>matmul|batch_matmul|generic)_(?P<input_shape>(?:\d+)(?:x\d+)+)+\s", re.IGNORECASE)

    profiler_programs = []
    unique_names = set()
    for line in pipeline_lines:
        # Find function name. Split out parameters.
        match = pattern.search(line)

        dispatch_index = int(match.group("dispatch_index"))
        operation_str = match.group("operation")
        operation_type = None
        if operation_str == "matmul":
            operation_type = OperationType.MATMUL
        elif operation_str == "batch_matmul":
            operation_type = OperationType.BATCH_MATMUL
        elif operation_str == "generic":
            operation_type = OperationType.GENERIC
        input_shape = match.group("input_shape").split("x")

        dispatch_name = "forward_dispatch_" + operation_str + "_" + \
            ("x".join(input_shape)) + "x" + data_type.iree_type

        # Ignore repeated dispatches (we don't care about fused contents)
        if dispatch_name in unique_names:
            continue
        unique_names.add(dispatch_name)

        print(f"Dispatch name: {dispatch_name}")

        # Create profiler program
        b = 0
        if len(input_shape) == 4:
            b = input_shape[0]
            input_shape = input_shape[1:]
        m = input_shape[0]
        n = input_shape[1]
        k = input_shape[2]
        template_filename = dispatch_name + ".mlir"
        result_filename = dispatch_name + "_results.csv"
        profiler_program = ProfilerProgram(dispatch_name, b, m, n, k, data_type,
                                           operation_type, target_backend, pipeline, template_filename, result_filename)
        profiler_programs.append(profiler_program)

    # Dump the profiler programs
    with open(output_program_path, "w") as f:
        for profiler_program in profiler_programs:
            json_str = profiler_program.dump_json()
            f.write(json_str + "\n")

    print(f"{len(profiler_programs)} unique profiler programs written to {output_program_path}")


def run_program(
        input_program_path: Path,
        results_dir_path: Path,
        benchmark_repetitions: int,
        benchmark_dispatch_batch_size: int):
    """Run profiler programs from file."""

    profiler_programs = []
    with open(input_program_path, "r") as f:
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            profiler_programs.append(ProfilerProgram.load_json(line))

    print(f"Loaded {len(profiler_programs)} ProfilePrograms from: {input_program_path}")

    for profiler_program in profiler_programs:
        template_mlir_model_path = results_dir_path.joinpath(
            profiler_program.template_mlir_filename)
        output_csv_path = results_dir_path.joinpath(
            profiler_program.output_csv_filename)

        print("\n\n===========================================")
        print(
            f"Running program: {profiler_program.name}.\n Placing template in model in: {template_mlir_model_path}.\n Outputting csv results to: {output_csv_path}")
        generate_temp_file(profiler_program.b, profiler_program.m, profiler_program.n, profiler_program.k,
                           profiler_program.data_type, profiler_program.operation_type, template_mlir_model_path=template_mlir_model_path)
        print("===========================================")

        run_profile(
            profiler_program.b,
            profiler_program.m,
            profiler_program.n,
            profiler_program.k,
            profiler_program.data_type,
            profiler_program.target_backend,
            template_mlir_model_path,
            output_csv_path,
            benchmark_repetitions,
            benchmark_dispatch_batch_size,
            operation_type=profiler_program.operation_type)
        print(f"Finished program: {profiler_program.name}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Coordinator for running the profiler over one or more sessions.")
    parser.add_argument("--mode",
                        type=str,
                        choices=["dump", "combine", "run"],
                        help="Set mode for profiler coordinator.",
                        required=True)
    parser.add_argument("--input_mlir_model",
                        type=Path,
                        help="Input model to analyze for tuning",
                        required=False,
                        default="/usr/local/google/home/kooljblack/Code/iree-tmp/batch_size/ir_dump/B32/dump_after_tile_distribute_workgroups.mlir")
    parser.add_argument("--output_program_dump",
                        type=Path,
                        help="Output path to dump tuning programs",
                        required=False,
                        default="/usr/local/google/home/kooljblack/Code/iree-tmp/tuning/artifacts/tuning_program.json")
    parser.add_argument("--input_program",
                        type=Path,
                        help="Input path to read tuning programs",
                        required=False,
                        default="/usr/local/google/home/kooljblack/Code/iree-tmp/tuning/artifacts/tuning_program.json")
    parser.add_argument("--results_dir",
                        type=dir_path,
                        help="Path to dir to place results from profiler run",
                        required=False,
                        default="/usr/local/google/home/kooljblack/Code/iree-tmp/tuning/results/")
    parser.add_argument("--benchmark_repetitions",
                        type=int,
                        help="Number of time to repeat each benchmark. The final result times are averaged",
                        required=False,
                        default=3)
    parser.add_argument("--benchmark_dispatch_batch_size",
                        type=int,
                        help="Number of iterations for each dispatch in benchmark",
                        required=False,
                        default=400)
    return parser.parse_args()


def main(args: argparse.ArgumentParser):
    if args.mode == "dump":
        dump_profile_programs(args.input_mlir_model, args.output_program_dump)
    if args.mode == "combine":
        raise RuntimeError("Not implemented yet.")
    if args.mode == "run":
        run_program(
            args.input_program,
            args.results_dir,
            args.benchmark_repetitions,
            args.benchmark_dispatch_batch_size)


if __name__ == "__main__":
    main(parse_arguments())
