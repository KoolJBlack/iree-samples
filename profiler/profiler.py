#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import enum
import dataclasses
import tempfile
from pathlib import Path
from typing import Optional, List
import csv
from datetime import date, datetime

from model_annotation import model_annotation

from iree.compiler import ir, CompilerToolError
from iree.compiler.transforms import ireec as ireec_trans
from iree.runtime import benchmark_module
from iree.runtime.benchmark import BenchmarkResult
import iree.runtime as ireert
import iree.compiler as ireec

from config_generation import Pipeline, OperationType, DataType, generate_configs, dump_shark_config_json


@enum.unique
class TargetBackend(enum.Enum):
    CUDA = "cuda"

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        try:
            return TargetBackend[s]
        except KeyError:
            raise ValueError()


class ProfilerResult:
    """Stores the results of each profile based on config."""
    # TODO: capture errors?

    def __init__(self,
                 config: dict,
                 compilation_successful: bool,
                 benchmark_successful: bool,
                 benchmark_results: List[BenchmarkResult] = [],
                 compiler_error: Optional[CompilerToolError] = None):
        self.config = config
        self.compilation_successful = compilation_successful
        self.benchmark_successful = benchmark_successful
        self.benchmark_results = benchmark_results
        self.compiler_error = None

    @staticmethod
    def create_with_result(config: dict, benchmark_results: List[BenchmarkResult]):
        return ProfilerResult(config, True, True, benchmark_results)

    @staticmethod
    def create_failed_compilation(config: dict, compiler_error: Optional[CompilerToolError] = None):
        return ProfilerResult(config, False, False, None, compiler_error=compiler_error)

    @staticmethod
    def create_failed_benchmark(config: dict):
        return ProfilerResult(config, True, False, None)

class ProfilerResultsWriter:
    """Class for writing benchmark results to CSV."""

    def __init__(self, output_csv_path: Path, benchmark_repetitions: int):
        self.output_csv_path = output_csv_path
        self.benchmark_repetitions = benchmark_repetitions
        self.field_names = [
            "benchmark_name",
            "tile_sizes", "work_group_sizes", "pipeline", "pipeline_depth", "identifier", "b", "m", "n", "k",
            "iterations",
            "time_mean", "cpu_time_mean",
            "time_median", "cpu_time_median",
            "time_std", "cpu_time_std",
            "time_cv", "cpu_time_cv",
            "error",
            # "time",  "cpu_time",  "iterations",  "user_counters",
        ]

    def initialize_output_csv(self):
        """Setup CSV output if it doesn't exist."""

        if not self.output_csv_path.exists():
            with open(self.output_csv_path, mode="w", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(self.field_names)

    def write_csv_result(self, profiler_result: ProfilerResult):
        """Save a profile result to csv."""
        # Config info
        config = profiler_result.config
        config_options = config["options"][0]
        pipeline_depth = 0
        if "pipeline_depth" in config_options.keys():
            pipeline_depth = config_options["pipeline_depth"]
        b = 0
        if "b" in config.keys():
            b = config["b"]

        bench_result = {
            "tile_sizes": str(config_options["work_group_tile_sizes"])[1:-1],
            "work_group_sizes": str(config_options["work_group_sizes"])[1:-1],
            "pipeline": config_options["pipeline"],
            "pipeline_depth": pipeline_depth,
            "identifier": config["identifier"],
            "b": b,
            "m": config["m"],
            "n": config["n"],
            "k": config["k"],
        }

        err = None
        if not profiler_result.compilation_successful:
            err = profiler_result.compiler_error
        if not profiler_result.benchmark_successful:
            #TODO: capture error
            err = "Failed to benchmark."
        else: 
            # Pull key benchmark metrics
            benchmark_results = profiler_result.benchmark_results
            benchmark_result_mean = benchmark_results[-4]
            benchmark_result_median = benchmark_results[-3]
            benchmark_result_std = benchmark_results[-2]
            benchmark_result_cv = benchmark_results[-1]
            benchmark_results_remaining = benchmark_results[:-4]
            benchmark_name = benchmark_results_remaining[0].benchmark_name
            iterations = benchmark_results_remaining[0].iterations

            bench_result.update({
                "benchmark_name": benchmark_name,
                "iterations": iterations,
                "time_mean": benchmark_result_mean.time,
                "cpu_time_mean": benchmark_result_mean.cpu_time,
                "time_median": benchmark_result_median.time,
                "cpu_time_median": benchmark_result_median.cpu_time,
                "time_std": benchmark_result_std.time,
                "cpu_time_std": benchmark_result_std.cpu_time,
                "time_cv": benchmark_result_cv.time,
                "cpu_time_cv": benchmark_result_cv.cpu_time,
            })
        bench_result.update({"error": err})
        
        with open(self.output_csv_path, mode="a", newline="") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=self.field_names)
            writer.writerow(bench_result)



def create_context() -> ir.Context:
    context = ir.Context()
    ireec_trans.register_all_dialects(context)
    context.allow_unregistered_dialects = True
    return context


def annotate_mlir_model(
        input_model_str: str,
        config_path: Path,
        annotated_model_output_path: Optional[Path] = None) -> ir.Module:
    """"Annotate model from with config. 
    Configs are consumed form a Path.
    Returns annotated IREE Model."""

    search_op = "matmul"
    with create_context() as ctx:
        annotated_model = model_annotation(
            ctx=None,
            input_contents=input_model_str,
            config_path=config_path,
            search_op=search_op,
        )

        mlir_str = str(annotated_model)

        if annotated_model_output_path:
            with open(annotated_model_output_path, "w") as f:
                f.write(mlir_str)
            print(f"Saved annotated mlir to: {annotated_model_output_path}.")

        return annotated_model


def compile_module_to_flatbuffer(
        module,
        target_device: str,
        frontend: str,
        benchmark_dispatch_batch_size: Optional[int] = None,
        extra_args: Optional[List] = []) -> tuple[Optional[bytes], Optional[CompilerToolError]]:
    """Compiles mlir module and returns the flatbuffer blob"""
    args = []
    if target_device == "cuda":
        args.extend(['--iree-llvm-target-cpu-features=host', '--iree-mhlo-demote-i64-to-i32=false', '--iree-flow-demote-i64-to-i32',
                    '--iree-hal-cuda-disable-loop-nounroll-wa', '--iree-stream-resource-index-bits=64', '--iree-vm-target-index-bits=64', '--iree-util-zero-fill-elided-attrs'])
        args.append('--iree-hal-cuda-llvm-target-arch=sm_80')

    else:
        raise ValueError(
            "Only `cuda` target device is supported for benchmarking")

    if benchmark_dispatch_batch_size:
        args.append(
            f'-iree-hal-benchmark-dispatch-repeat-count={benchmark_dispatch_batch_size}')

    args.extend(extra_args)
    input_type = frontend

    # Compile according to the input type, else just try compiling.
    try:
        flatbuffer_blob = ireec.compile_str(
            module,
            target_backends=[target_device],
            extra_args=args,
            input_type=input_type,
        )
    except CompilerToolError as err:
        flatbuffer_blob = None
        print(f"the compiler failed but we will still continue: {err}")
        return None, err

    return flatbuffer_blob, None


def run_benchmark_module(
        flatbuffer_blob: bytes,
        entry_function: str,
        function_input: List[str] = [],
        device: str = "cuda",
        benchmark_repetitions: Optional[int] = None,
        benchmark_dispatch_batch_size: Optional[int] = None) -> List[BenchmarkResult]:
    # Create a module
    config = ireert.Config(driver_name="cuda")
    vm_module = ireert.VmModule.from_flatbuffer(
        config.vm_instance, flatbuffer_blob)
    funcs = [a for a in vm_module.function_names if a != "__init"]
    print(f"Benchmarking module with funcs: {funcs}")

    benchmark_results = benchmark_module(
        vm_module, device=device, benchmark_repetitions=benchmark_repetitions, batch_size=benchmark_dispatch_batch_size)

    return benchmark_results


def dir_path(string) -> Optional[Path]:
    """Returns path to dir if it exists"""
    if os.path.isdir(string):
        return Path(string)
    else:
        # raise NotADirectoryError(string)
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Profiler for tuning dispatches in IREE")
    parser.add_argument(
        "--m", type=int, help="m dim for matmul", required=False, default=4096)
    parser.add_argument(
        "--n", type=int, help="n dim for matmul", required=False, default=3072)
    parser.add_argument(
        "--k", type=int, help="k dim for matmul", required=False, default=768)
    parser.add_argument("--data_type",
                        type=DataType,
                        help="Numeric type of input matrices",
                        default=DataType.F32,
                        required=False)
    parser.add_argument("--target_backend", type=TargetBackend,
                        help="Select target backend. Only `cuda` is supported at the moment", required=False, default=TargetBackend.CUDA)
    parser.add_argument("--template_mlir_model",
                        type=Path,
                        help="Path to input .mlir for profiling. Used as input to annotation and compilation. Args for dimensions and datatype must match.",
                        required=False,
                        default="/usr/local/google/home/kooljblack/Code/iree-tmp/batch_size/search/benchmark-tensorcore-input.mlir")
    parser.add_argument("--artifacts_dir",
                        type=dir_path,
                        help="Path to dir to place annotated models and compiled artifacts for benchmarking. If not provied, a temp dir is used isntead",
                        required=False,
                        default="/usr/local/google/home/kooljblack/Code/iree-tmp/batch_size/search/")
    parser.add_argument("--extra_compilation_args",
                        type=list,
                        help="Extra arguments to be added to compilation",
                        required=False,
                        default=[])
    parser.add_argument("--output_csv",
                        type=Path,
                        help="Path to store csv results",
                        required=False,
                        default=None)
    parser.add_argument("--benchmark_repetitions",
                        type=int,
                        help="Number of time to repeat each benchmark. The final result times are averaged",
                        required=False,
                        default=5)
    parser.add_argument("--benchmark_dispatch_batch_size",
                        type=int,
                        help="Number of iterations for each dispatch in benchmark",
                        required=False,
                        default=1000)
    return parser.parse_args()

def main(args: argparse.ArgumentParser):
    print(f"Profiling shape [{args.m},{args.n},{args.k}] on {args.target_backend} for optimal config.")

    # Setup temporary dir for artifacts
    temp_dir = None
    artifacts_dir_path = None
    if args.artifacts_dir:
        artifacts_dir_path = args.artifacts_dir
    else:
        temp_dir = tempfile.TemporaryDirectory()
        artifacts_dir_path = Path(temp_dir.name)

    # Paths needed
    annotated_model_path = artifacts_dir_path.joinpath("annotated-model.mlir")
    generated_config_path = artifacts_dir_path.joinpath(
        "generated_config.json")
    vmfb_path = artifacts_dir_path.joinpath("annotated_flatbuffer.vmfb")

    benchmark_repetitions = args.benchmark_repetitions
    benchmark_dispatch_batch_size = args.benchmark_dispatch_batch_size

    # Output CSV path
    output_csv_path = args.output_csv
    now = datetime.now()
    if not output_csv_path:
        dt = now.strftime("%Y-%m-%d_%I:%M:%S%p")
        output_csv_path = artifacts_dir_path.joinpath(dt+"_results.csv")

    benchmark_results_writer = ProfilerResultsWriter(
        output_csv_path, benchmark_repetitions)
    benchmark_results_writer.initialize_output_csv()

    # Load template model
    input_model_path = args.template_mlir_model
    template_model_str = ""
    with open(input_model_path, "r") as f:
        template_model_str = f.read()

    if not template_model_str:
        raise ValueError("Unable to read template model.")

    configs = generate_configs(pipeline=Pipeline.GPU_TENSORCORE, operation=OperationType.MATMUL, input_shape=[
                               args.m, args.n, args.k], data_type=args.data_type)
    print(f"Generated configs for model: {len(configs)}")

    profiler_results = []
    # For each config, annotate the model, compile and benchmark
    # Configs need to be saved to file for use.
    for index, config in enumerate(configs[0:10]):
        print(f"Testing config {index}/{len(configs)} : {config}")

        # Save the config to file
        print(f"Saved generated config to: {generated_config_path}.")
        bytes_written = dump_shark_config_json(config, generated_config_path)
        annotated_module = annotate_mlir_model(
            input_model_str=template_model_str, config_path=generated_config_path)

        # Compile model
        flatbuffer_blob, err = compile_module_to_flatbuffer(
            str(annotated_module), "cuda", "mhlo", benchmark_dispatch_batch_size, args.extra_compilation_args)

        # Was compilation successful?
        if not flatbuffer_blob:
            print("Failed to compile!")
            profiler_results.append(
                ProfilerResult.create_failed_compilation(config, err))
            benchmark_results_writer.write_csv_result(profiler_results[-1])
            continue
        
        # Benchmark model
        benchmark_results = run_benchmark_module(flatbuffer_blob, entry_function="forward",
                                                 benchmark_repetitions=benchmark_repetitions, benchmark_dispatch_batch_size=benchmark_dispatch_batch_size)

        # Was benchmark successful?
        if not benchmark_results:
            print("Failed to benchmark!")
            profiler_results.append(
                ProfilerResult.create_failed_benchmark(config))
            benchmark_results_writer.write_csv_result(profiler_results[-1])

            continue

        profiler_results.append(
            ProfilerResult.create_with_result(config, benchmark_results))
        benchmark_results_writer.write_csv_result(profiler_results[-1])

    print(f"Produced {len(profiler_results)} profile results.")


    # Cleanup any artifacts
    if temp_dir:
        temp_dir.cleanup()

    print(f"Results stored in: {output_csv_path}")
    print("Profiler Complete!")


if __name__ == "__main__":
    main(parse_arguments())
