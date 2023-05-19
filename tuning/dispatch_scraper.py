
import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import re
from dataclasses import dataclass

from profiler import run_profile, TargetBackend
from utils.data_types import OperationType, ProfilerProgram, Pipeline, OperationType, DataType


def dump_profile_programs(input_mlir_model: Path,
                          output_program_path: Path,
                          target_backend: TargetBackend = TargetBackend.CUDA,
                          pipeline: Pipeline = Pipeline.GPU_TENSORCORE_MMASYNC):
    """Parse an mlir model dump after 'tile distribute workgroups pass'. Identifies all dispatches that match the given pipeline and dumps them to json file."""

    pipeline_marker = pipeline.value

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
        f"Found {len(pipeline_lines)} dispatches with marker: {pipeline_marker}\n")

    # Regex match each dispatch name to get parameters
    pattern = re.compile(
        "@forward_dispatch_(?P<dispatch_index>\d+)_(?P<operation>matmul|batch_matmul|generic)_(?P<input_shape>(?:\d+)(?:x\d+)+)+_(?P<data_type>f16|f32)\s", re.IGNORECASE)

    profiler_programs = []
    unique_names = set()
    for line in pipeline_lines:
        # Find function name. Split out parameters.
        # print(f"Analyzing line: {line}")
        match = pattern.search(line)

        dispatch_index = int(match.group("dispatch_index"))
        operation_str = match.group("operation")
        operation_type = None
        if operation_str == "matmul":
            operation_type = OperationType.MATMUL
        elif operation_str == "batch_matmul":
            operation_type = OperationType.BATCH_MATMUL
        input_shape = match.group("input_shape").split("x")
        data_type = DataType.from_string(match.group("data_type"))

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
            b = int(input_shape[0])
            input_shape = input_shape[1:]
        m = int(input_shape[0])
        n = int(input_shape[1])
        k = int(input_shape[2])
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


def combine_programs(
        input_program_dir: Path,
        output_program_path: Path):
    """Combines input programs from a dir into a single output program. Removes duplicate programs."""

    raw_program_count = 0
    input_file_count = 0
    profiler_programs_dict = {}
    profiler_programs_list = []  # Preserve order
    for child in input_program_dir.iterdir():
        if child.suffix != ".json":
            continue

        # Skip the output file if its in the same dir
        if child.absolute() == output_program_path.absolute():
            continue

        input_file_count += 1
        with open(child, "r") as f:
            for line in f.readlines():
                raw_program_count += 1
                profiler_program = ProfilerProgram.load_json(line.rstrip())
                program_name = profiler_program.name
                if program_name in profiler_programs_dict.keys():
                    continue

                profiler_programs_dict[program_name] = profiler_program
                profiler_programs_list.append(profiler_program)

    # Dump the profiler programs
    with open(output_program_path, "w") as f:
        for profiler_program in profiler_programs_list:
            json_str = profiler_program.dump_json()
            f.write(json_str + "\n")

    print(f"Combined {input_file_count} tuning programs with {raw_program_count} programs into {len(profiler_programs_dict.values())} unique programs.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Coordinator for running the profiler over one or more sessions.")
    parser.add_argument("--mode",
                        type=str,
                        choices=["dump", "combine", "run"],
                        help="Set mode for profiler coordinator.",
                        required=True)
    # Dump
    parser.add_argument("--input_mlir_model",
                        type=Path,
                        help="Input model to analyze for tuning",
                        required=False,
                        default=None)
    parser.add_argument("--output_program_dump",
                        type=Path,
                        help="Output path to dump tuning programs",
                        required=False,
                        default=None)
    # Combine
    parser.add_argument("--input_programs_dir",
                        type=Path,
                        help="Path to dir containing programs to combine",
                        required=False,
                        default=None)
    parser.add_argument("--output_combined_program",
                        type=Path,
                        help="Path to combined program",
                        required=False,
                        default=None)

    return parser.parse_args()


def main(args: argparse.ArgumentParser):
    if args.mode == "dump":
        dump_profile_programs(args.input_mlir_model, args.output_program_dump)
    if args.mode == "combine":
        combine_programs(args.input_programs_dir, args.output_combined_program)


if __name__ == "__main__":
    main(parse_arguments())
