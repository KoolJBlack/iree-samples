import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

from profiler import run_profile
from utils.data_types import ProfilerProgram

def dir_path(string) -> Optional[Path]:
    """Returns path to dir if it exists"""
    if os.path.isdir(string):
        return Path(string)
    else:
        raise NotADirectoryError(string)
        # return None

def run_program(
        input_program_path: Path,
        results_dir_path: Path,
        benchmark_repetitions: int,
        benchmark_dispatch_batch_size: int,
        benchmark_min_time: float):
    """Run profiler programs from file."""

    profiler_programs = []
    with open(input_program_path, "r") as f:
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            profiler_programs.append(ProfilerProgram.load_json(line))

    print(
        f"Loaded {len(profiler_programs)} ProfilePrograms from: {input_program_path}")

    for profiler_program in profiler_programs:
        template_mlir_model_path = results_dir_path.joinpath(
            profiler_program.template_mlir_filename)
        output_csv_path = results_dir_path.joinpath(
            profiler_program.output_csv_filename)

        print("\n\n===========================================")
        print(
            f"Tuning dispatch: {profiler_program.name}.\n Placing template in model in: {template_mlir_model_path}.\n Outputting csv results to: {output_csv_path}")
        print("===========================================")

        run_profile(
            b=profiler_program.b,
            m=profiler_program.m,
            n=profiler_program.n,
            k=profiler_program.k,
            data_type=profiler_program.data_type,
            target_backend=profiler_program.target_backend,
            output_csv_path=output_csv_path,
            benchmark_repetitions=benchmark_repetitions,
            benchmark_dispatch_batch_size=benchmark_dispatch_batch_size,
            benchmark_min_time = benchmark_min_time,
            pipeline=profiler_program.pipeline,
            operation_type=profiler_program.operation_type)
        print(f"Finished tuning dispatch: {profiler_program.name}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A tuner to run multiple configs over a dispatch and collect results.")
    # Run
    parser.add_argument("--input_program",
                        type=Path,
                        help="Input path to read tuning programs",
                        required=False,
                        default=None)
    parser.add_argument("--results_dir",
                        type=dir_path,
                        help="Path to dir to place results from profiler run",
                        required=False,
                        default=None)
    parser.add_argument("--benchmark_repetitions",
                        type=int,
                        help="Number of time to repeat each benchmark. The final result times are averaged",
                        required=False,
                        default=1)
    parser.add_argument("--benchmark_dispatch_batch_size",
                        type=int,
                        help="Number of iterations for each dispatch in benchmark",
                        required=False,
                        default=200)
    parser.add_argument("--benchmark_min_time",
                        type=float,
                        help="Minimum length of each benchmark run in seconds.",
                        required=False,
                        default=3.0)
    return parser.parse_args()


def main(args: argparse.ArgumentParser):
    run_program(
        args.input_program,
        args.results_dir,
        args.benchmark_repetitions,
        args.benchmark_dispatch_batch_size,
        args.benchmark_min_time)


if __name__ == "__main__":
    main(parse_arguments())
