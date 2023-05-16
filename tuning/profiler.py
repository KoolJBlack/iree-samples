import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
import concurrent.futures
from itertools import zip_longest
import time
from datetime import timedelta

from iree.compiler import CompilerToolError
from iree.compiler.tools import core as ireec
from iree.runtime import benchmark_module
from iree.runtime.benchmark import BenchmarkResult, BenchmarkToolError
import iree.runtime as ireert

from model.config_generation import generate_configs
from utils.data_types import TargetBackend, TargetDevice, TargetDriver, CompilerFrontend, CompilationResult, DispatchConfig, Dispatch, DEFAULT_CONFIG, Pipeline, OperationType, DataType
from utils.iree_utils import CudaFlavors, iree_compile_arguments
from model.model_generator import generate_model
from results.results import ProfilerResult, ProfilerResultsWriter


def compile_module_to_flatbuffer(
        module,
        target_device: TargetDevice,
        frontend: CompilerFrontend = CompilerFrontend.MHLO,
        benchmark_dispatch_batch_size: Optional[int] = None,
        extra_args: Optional[List] = []) -> tuple[Optional[bytes], Optional[CompilerToolError]]:
    """Compiles mlir module and returns the flatbuffer blob"""
    args = []
    if target_device == TargetDevice.CUDA:
        args.extend(iree_compile_arguments(TargetBackend.CUDA, [
                    CudaFlavors.SHARK_DEFAULT, CudaFlavors.CUDA_SM_80]))
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
        print(f"Compile tool failed: {err}")
        return None, err
    return flatbuffer_blob, None


def compile_with_configs(
        dispatch: Dispatch,
        target_device: TargetDevice,
        configs: List[DispatchConfig],
        operation_type: OperationType,
        benchmark_dispatch_batch_size: int,
        extra_compilation_args: List,
        parallel_threads: int = 1) -> List[CompilationResult]:
    """Parallel annotation and compiling for models."""

    def thread_compile_with_config(dispatch: Dispatch, config: DispatchConfig):
        start_time = time.time()
        annotated_model = generate_model(
            dispatch, None if config == DEFAULT_CONFIG else config)

        # Compile model
        flatbuffer_blob, err = compile_module_to_flatbuffer(
            str(annotated_model), target_device, CompilerFrontend.MHLO, benchmark_dispatch_batch_size, extra_compilation_args)
        elapsed_time = time.time() - start_time
        return CompilationResult(config, flatbuffer_blob, err, elapsed_time)

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        future_to_config = {executor.submit(
            thread_compile_with_config, dispatch, config): config for config in configs}
        completed_futures, failed_futures = concurrent.futures.wait(
            future_to_config)

        if failed_futures:
            raise Exception("Failed future from compilation")

        for future in completed_futures:
            config = future_to_config[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print('%r generated an exception: %s' % (config, exc))

    return results


def run_benchmark_module(
        flatbuffer_blob: bytes,
        entry_function: str,
        function_input: List[str] = [],
        device: TargetDevice = TargetDevice.CUDA,
        driver: TargetDriver = TargetDriver.CUDA,
        benchmark_repetitions: Optional[int] = None,
        benchmark_dispatch_batch_size: Optional[int] = None) -> Tuple[List[BenchmarkResult], Optional[BenchmarkToolError]]:
    # Create a module
    config = ireert.Config(driver_name=driver)
    vm_module = ireert.VmModule.from_flatbuffer(
        config.vm_instance, flatbuffer_blob)
    funcs = [a for a in vm_module.function_names if a != "__init"]
    # print(f"Benchmarking module with funcs: {funcs}")

    try:
        benchmark_results = benchmark_module(
            vm_module, device=device, benchmark_repetitions=benchmark_repetitions, batch_size=benchmark_dispatch_batch_size)
        return benchmark_results, None
    except BenchmarkToolError as err:
        flatbuffer_blob = None
        print(f"Benchmark tool failed: {err}")
        return None, err


def dir_path(string) -> Optional[Path]:
    """Returns path to dir if it exists"""
    if os.path.isdir(string):
        return Path(string)
    else:
        # raise NotADirectoryError(string)
        return None


def run_profile(
        b: Optional[int],
        m: int,
        n: int,
        k: int,
        data_type: DataType,
        target_backend: TargetBackend,
        output_csv_path: Path,
        benchmark_repetitions: int,
        benchmark_dispatch_batch_size: int,
        extra_compilation_args: List[str] = [],
        compilation_parallelism: int = 1,
        pipeline: Pipeline = Pipeline.GPU_TENSORCORE,
        operation_type: OperationType = OperationType.MATMUL,
        config_start_index: int = 0,
        config_end_index: Optional[int] = None):
    """Run the profiler."""
    print(
        f"Profiling {operation_type} with shape [{b},{m},{n},{k}] on {target_backend}/{pipeline} backend for optimal config.")
    compilation_parallelism = compilation_parallelism

    dispatch = Dispatch(
        pipeline_name=pipeline,
        operation=operation_type,
        data_type=data_type,
        b=b,
        m=m,
        n=n,
        k=k
    )

    # Output CSV path
    now = datetime.now()
    if not output_csv_path:
        raise ValueError("Output CSV path required.")

    benchmark_results_writer = ProfilerResultsWriter(
        output_csv_path)
    benchmark_results_writer.initialize_output_csv()

    input_shape = [int(m), int(n), int(k)]
    if b:
        input_shape.insert(0, int(b))

    target_device = None
    if target_backend == TargetBackend.CUDA:
        target_device = TargetDevice.CUDA

    configs = generate_configs(
        target_backend=target_backend, dispatch=dispatch)

    print(f"Generated {len(configs)} configs for dispatch.")

    # Control config for first benchmark
    configs.insert(0, DEFAULT_CONFIG)  # None represent default config

    profiler_results = []
    if not config_end_index:
        config_end_index = len(configs)
    config_count = config_end_index
    configs = configs[config_start_index:config_end_index]

    # Grab up to compilation_parallelism configs. Group for subsequent iterations.
    iter_configs = [iter(configs)] * compilation_parallelism
    grouped_configs = zip_longest(fillvalue=None, *iter_configs)

    tuning_start_time_s = time.time()

    for group_index, config_group in enumerate(grouped_configs):
        for index, config in enumerate(config_group):
            config_index = group_index * compilation_parallelism + \
                index + config_start_index
            print(f"Profiling config {config_index}/{config_count} : {config}")

        # For each config, annotate the model, compile and benchmark
        compilation_results = compile_with_configs(
            dispatch,
            target_device,
            config_group,
            operation_type,
            benchmark_dispatch_batch_size,
            extra_compilation_args,
            compilation_parallelism)

        for index, compilation_result in enumerate(compilation_results):
            config_index = group_index * compilation_parallelism + \
                index + config_start_index
            # Was compilation successful?
            if not compilation_result.flatbuffer_blob:
                print(f"Failed to compile {config_index}/{config_count}")
                tuning_elapsed_time_s = time.time() - tuning_start_time_s
                print(
                    f"Profiled config {config_index}/{config_count} - total elapsed time: {timedelta(seconds=tuning_elapsed_time_s)}, compilation time: {compilation_result.compilation_time_s:0.4f}sec")
                profiler_results.append(
                    ProfilerResult.create_with_err(
                        config_index,
                        compilation_result.config,
                        dispatch,
                        compilation_result.compilation_time_s,
                        None,
                        err))
                benchmark_results_writer.write_csv_result(profiler_results[-1])
                continue

            benchmark_start_time_s = time.time()

            # Benchmark model
            benchmark_results, err = run_benchmark_module(compilation_result.flatbuffer_blob, entry_function="forward",
                                                          benchmark_repetitions=benchmark_repetitions, benchmark_dispatch_batch_size=benchmark_dispatch_batch_size)

            benchmark_elapsed_time_s = time.time() - benchmark_start_time_s
            tuning_elapsed_time_s = time.time() - tuning_start_time_s

            # Was benchmark successful?
            if err:
                print(f"Failed to benchmark {config_index}/{config_count}")

                print(f"Profiled config {config_index}/{config_count} - total elapsed time: {timedelta(seconds=tuning_elapsed_time_s)}, compilation time: {compilation_result.compilation_time_s:0.4f}sec, benchmark time: {benchmark_elapsed_time_s:0.4f}sec")
                profiler_results.append(
                    ProfilerResult.create_with_err(
                        config_index,
                        compilation_result.config,
                        dispatch,
                        compilation_result.compilation_time_s,
                        benchmark_elapsed_time_s,
                        err))
                benchmark_results_writer.write_csv_result(profiler_results[-1])
            else:
                print(
                    f"Profiled config {config_index}/{config_count} - total elapsed time: {timedelta(seconds=tuning_elapsed_time_s)}, compilation time: {compilation_result.compilation_time_s:0.4f}sec, benchmark time: {benchmark_elapsed_time_s:0.4f}sec with {benchmark_results[0].iterations} iterations")
                profiler_results.append(
                    ProfilerResult.create_with_benchmark_results(
                        config_index,
                        compilation_result.config,
                        dispatch,
                        benchmark_results,
                        compilation_result.compilation_time_s,
                        benchmark_elapsed_time_s))
                benchmark_results_writer.write_csv_result(profiler_results[-1])

    print(f"Produced {len(profiler_results)} profile results.")

    print(f"Results stored in: {output_csv_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Profiler for tuning dispatches in IREE.")
    parser.add_argument(
        "--b", type=int, help="b dim for batch matmul. Must match template mlir model.", required=False, default=None)
    parser.add_argument(
        "--m", type=int, help="m dim for matmul. Must match template mlir model.", required=True)
    parser.add_argument(
        "--n", type=int, help="n dim for matmul. Must match template mlir model.", required=True)
    parser.add_argument(
        "--k", type=int, help="k dim for matmul. Must match template mlir model.", required=True)
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
                        required=True,
                        default=None)
    parser.add_argument("--extra_compilation_args",
                        type=list,
                        help="Extra arguments to be added to compilation",
                        required=False,
                        default=[])
    parser.add_argument("--output_csv",
                        type=Path,
                        help="Path to csv file for results",
                        required=True,
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
    parser.add_argument("--compilation_parallelism",
                        type=int,
                        help="Number of simultaneous compilations to run. Default=1",
                        required=False,
                        default=1)
    parser.add_argument("--config_start_index",
                        type=int,
                        help="Continue from a config number",
                        required=False,
                        default=0)
    parser.add_argument("--config_end_index",
                        type=int,
                        help="Continue from a config number",
                        required=False,
                        default=None)
    return parser.parse_args()


def main(args: argparse.ArgumentParser):
    run_profile(
        b=args.b,
        m=args.m,
        n=args.n,
        k=args.k,
        data_type=args.data_type,
        target_backend=args.target_backend,
        template_mlir_model_path=args.template_mlir_model,
        extra_compilation_args=args.extra_compilation_args,
        output_csv_path=args.output_csv,
        benchmark_repetitions=args.benchmark_repetitions,
        benchmark_dispatch_batch_size=args.benchmark_dispatch_batch_size,
        compilation_parallelism=args.compilation_parallelism,
        config_start_index=args.config_start_index,
        config_end_index=args.config_end_index)


if __name__ == "__main__":
    main(parse_arguments())
