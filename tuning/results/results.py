import enum
from typing import Optional, List, Tuple, Union
from pathlib import Path
import csv

from utils.data_types import DispatchConfig, Dispatch, OperationType
from iree.compiler import CompilerToolError
from iree.runtime.benchmark import BenchmarkResult, BenchmarkToolError

###################################################################################################
# Libraries for organizing and saving results from profile runs.
###################################################################################################


"""Enum for profiler result dict keys."""


class PROFILER_RESULT_KEYS(str, enum.Enum):
    CONFIG_INDEX = "config_index"
    # BENCHMARK_NAME = "benchmark_name"
    TILE_SIZE = "tile_sizes"
    WORK_GROUP_SIZES = "work_group_sizes"
    PIPELINE_DEPTH = "pipeline_depth"
    PIPELINE = "pipeline"
    OPERATION = "identifier"
    B = "b"
    M = "m"
    N = "n"
    K = "k"
    BENCHMARK_REPETITIONS = "benchmark_repetitions"
    ITERATIONS = "iterations"
    PERCENTAGE_OF_PEAK = "percent_of_peak"
    TIME_MEAN_MS = "time_mean_milliseconds"
    # TIME_MEDIAN_MS = "time_median_milliseconds"
    TIME_MIN_MS = "time_min_milliseconds"
    TIME_STD_MS = "time_std_milliseconds"
    # TIME_CV_MS = "time_cv_milliseconds"
    GFLOPS = "gflops"
    COMPILATION_TIME_S = "compilation_time_seconds"
    BENCHMARK_TIME_S = "benchmark_time_seconds"
    ERROR = "error"


def profiler_result_dict_keys() -> dict:
    return [e.value for e in PROFILER_RESULT_KEYS]


def calculate_gflops(dispatch: Dispatch, runtime_ms: float):
    """Returns the gflops"""
    if dispatch.operation == OperationType.MATMUL:
        matmul_floating_operations = 2 * dispatch.m * dispatch.n * dispatch.k
    elif dispatch.operation == OperationType.BATCH_MATMUL:
        matmul_floating_operations = 2 * dispatch.b * \
            dispatch.m * dispatch.n * dispatch.k
    else:
        raise RuntimeError("unable to calculate gflops for non matmul")
    gflops = float(matmul_floating_operations) / runtime_ms / 1.0e6
    return gflops


class ProfilerResult:
    """Stores the results of each profiler run and its config."""

    def __init__(self,
                 config_index: int,
                 config: DispatchConfig,
                 dispatch: Dispatch,
                 benchmark_repetitions: Optional[int] = None,
                 iterations: Optional[int] = None,
                 time_mean_ms: Optional[float] = None,
                 time_min_ms: Optional[float] = None,
                 time_std_ms: Optional[float] = None,
                 compilation_time: Optional[float] = None,
                 benchmark_time: Optional[float] = None,
                 err: Union[CompilerToolError, BenchmarkToolError] = None,
                 dict: Optional[dict] = None):
        # If the dict is provided, use that directly
        if dict:
            self.profiler_result_dict = dict
            return
        self.profiler_result_dict = {
            PROFILER_RESULT_KEYS.CONFIG_INDEX: config_index,
            PROFILER_RESULT_KEYS.TILE_SIZE: str(config.tile_size)[1:-1],
            PROFILER_RESULT_KEYS.WORK_GROUP_SIZES: str(config.workgroup_size)[1:-1],
            PROFILER_RESULT_KEYS.PIPELINE: config.pipeline_name,
            PROFILER_RESULT_KEYS.PIPELINE_DEPTH: config.pipeline_depth,
            PROFILER_RESULT_KEYS.OPERATION: config.operation,
            PROFILER_RESULT_KEYS.B: dispatch.b,
            PROFILER_RESULT_KEYS.M: dispatch.m,
            PROFILER_RESULT_KEYS.N: dispatch.n,
            PROFILER_RESULT_KEYS.K: dispatch.k,
            PROFILER_RESULT_KEYS.BENCHMARK_REPETITIONS: benchmark_repetitions,
            PROFILER_RESULT_KEYS.ITERATIONS: iterations,
            PROFILER_RESULT_KEYS.TIME_MEAN_MS: time_mean_ms,
            PROFILER_RESULT_KEYS.TIME_MIN_MS: time_min_ms,
            PROFILER_RESULT_KEYS.TIME_STD_MS: time_std_ms,
            PROFILER_RESULT_KEYS.COMPILATION_TIME_S: compilation_time,
            PROFILER_RESULT_KEYS.BENCHMARK_TIME_S: benchmark_time,
            PROFILER_RESULT_KEYS.ERROR: err,
        }

        runtime_ms = None
        if not err:
            runtime_ms = float(
                self.profiler_result_dict[PROFILER_RESULT_KEYS.TIME_MEAN_MS])
            self.profiler_result_dict.update({
                PROFILER_RESULT_KEYS.GFLOPS: float(round(calculate_gflops(dispatch, runtime_ms=runtime_ms), 2))})

    @staticmethod
    def create_with_err(
            config_index: int,
            config: DispatchConfig,
            dispatch: Dispatch,
            compilation_time: float,
            benchmark_time: Optional[float],
            err: Union[CompilerToolError, BenchmarkToolError]):
        return ProfilerResult(
            config_index,
            config,
            dispatch,
            benchmark_repetitions=None,
            iterations=None,
            time_mean_ms=None,
            time_min_ms=None,
            time_std_ms=None,
            compilation_time=compilation_time,
            benchmark_time=benchmark_time,
            err=err)

    @staticmethod
    def create_with_benchmark_results(
            config_index: int,
            config: DispatchConfig,
            dispatch: Dispatch,
            benchmark_results: List[BenchmarkResult],
            compilation_time: float,
            benchmark_time: float):
        def strip_ms(raw_time: str):
            return raw_time.split(" ")[0]

        benchmark_repetitions = 1
        iterations = 1
        time_mean_ms = None
        time_min_ms = None
        time_std_ms = None
        # Pull key benchmark metrics
        if len(benchmark_results) == 1:
            # There is only one result
            benchmark_result = benchmark_results[0]
            iterations = benchmark_results[0].iterations
            benchmark_repetitions = 1
            time_mean_ms = strip_ms(benchmark_result.time)
            time_min_ms = strip_ms(benchmark_result.time)
            time_std_ms = 0
        else:
            benchmark_result_mean = benchmark_results[-4]
            # benchmark_result_median = benchmark_results[-3]
            benchmark_result_std = benchmark_results[-2]
            # benchmark_result_cv = benchmark_results[-1]
            benchmark_results_remaining = benchmark_results[:-4]

            bench_times = [float(strip_ms(bench_result.time))
                           for bench_result in benchmark_results_remaining]

            iterations = benchmark_results_remaining[0].iterations
            benchmark_repetitions = len(benchmark_results_remaining)
            time_mean_ms = strip_ms(benchmark_result_mean.time)
            time_min_ms = min(bench_times)
            time_std_ms = strip_ms(benchmark_result_std.time)

        return ProfilerResult(
            config_index,
            config,
            dispatch,
            benchmark_repetitions,
            iterations,
            time_mean_ms,
            time_min_ms,
            time_std_ms,
            compilation_time=compilation_time,
            benchmark_time=benchmark_time,
            err=None)

    @staticmethod
    def create_from_dict(profiler_result_dict: dict):
        config_index = profiler_result_dict[PROFILER_RESULT_KEYS.CONFIG_INDEX]
        benchmark_repetitions = profiler_result_dict[PROFILER_RESULT_KEYS.BENCHMARK_REPETITIONS]
        iterations = profiler_result_dict[PROFILER_RESULT_KEYS.ITERATIONS]
        time_mean_ms = profiler_result_dict[PROFILER_RESULT_KEYS.TIME_MEAN_MS]
        time_min_ms = profiler_result_dict[PROFILER_RESULT_KEYS.TIME_MIN_MS]
        time_std_ms = profiler_result_dict[PROFILER_RESULT_KEYS.TIME_STD_MS]
        compilation_time = profiler_result_dict[PROFILER_RESULT_KEYS.COMPILATION_TIME_S]
        benchmark_time = profiler_result_dict[PROFILER_RESULT_KEYS.BENCHMARK_TIME_S]
        err = profiler_result_dict[PROFILER_RESULT_KEYS.ERROR]
        return ProfilerResult(
            config_index=config_index,
            config=None,
            dispatch=None,
            benchmark_repetitions=benchmark_repetitions,
            iterations=iterations,
            time_mean_ms=time_mean_ms,
            time_min_ms=time_min_ms,
            time_std_ms=time_std_ms,
            compilation_time=compilation_time,
            benchmark_time=benchmark_time,
            err=err,
            dict=profiler_result_dict)

    def set_percentage_of_peak(self, percentage_of_peak: float):
        self.profiler_result_dict[PROFILER_RESULT_KEYS.PERCENTAGE_OF_PEAK] = percentage_of_peak


class ProfilerResultsWriter:
    """Class for writing benchmark results to CSV."""

    def __init__(self, output_csv_path: Path):
        self.output_csv_path = output_csv_path
        self.field_names = profiler_result_dict_keys()

    def initialize_output_csv(self, force: bool = False):
        """Setup CSV output if it doesn't exist."""
        if not self.output_csv_path.exists() or force:
            with open(self.output_csv_path, mode="w", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(self.field_names)

    def write_csv_result(self, profiler_result: ProfilerResult):
        """Save a profile result to csv."""
        with open(self.output_csv_path, mode="a", newline="") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=self.field_names)
            profiler_result_dict = profiler_result.profiler_result_dict
            writer.writerow(profiler_result_dict)

    def write_empty_line(self):
        with open(self.output_csv_path, mode="a", newline="") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=self.field_names)
            writer.writerow({})


class ProfilerResultsReader:
    """Class for reading and analyzing profile results from CSV."""

    def __init__(self, input_csv_path: Path):
        self.input_csv_path = input_csv_path
        self.field_names = profiler_result_dict_keys()
        self.profiler_results: List[ProfilerResult] = []
        self.profiler_results_success: List[ProfilerResult] = []
        self.profiler_results_failed: List[ProfilerResult] = []
        self.profiler_result_control: ProfilerResult = None
        self.dispatch = None
        self.read_csv()

    def read_csv(self):
        with open(self.input_csv_path, "r") as csv_input_file:
            csv_input_file.readline()  # Dump the first line since it contains the column titles
            reader = csv.DictReader(
                csv_input_file, fieldnames=profiler_result_dict_keys())
            for row_dict in reader:
                self.profiler_results.append(
                    ProfilerResult.create_from_dict(row_dict))

        # Sort results. Separate success and failed
        self.profiler_results.sort(
            key=lambda profiler_result: profiler_result.profiler_result_dict[PROFILER_RESULT_KEYS.TIME_MEAN_MS])

        def succeeded(profiler_result: ProfilerResult):
            return str(profiler_result.profiler_result_dict[PROFILER_RESULT_KEYS.ERROR]) == ""
        self.profiler_results_success = [
            profiler_result for profiler_result in self.profiler_results if succeeded(profiler_result)]
        self.profiler_results_failed = [
            profiler_result for profiler_result in self.profiler_results if not succeeded(profiler_result)]

        peak_time_ms = float(
            self.profiler_results_success[0].profiler_result_dict[PROFILER_RESULT_KEYS.TIME_MEAN_MS])
        for profiler_result in self.profiler_results_success:
            percentage_of_peak = float(
                profiler_result.profiler_result_dict[PROFILER_RESULT_KEYS.TIME_MEAN_MS]) / peak_time_ms
            profiler_result.set_percentage_of_peak(percentage_of_peak)

        # Separate the control from the successful profiles
        self.profiler_result_control = [
            control for control in self.profiler_results_success if control.profiler_result_dict[PROFILER_RESULT_KEYS.OPERATION] == "default"][0]
        self.profiler_results_success.remove(self.profiler_result_control)

    def write_updated_results(self, csv_output_path: Path):
        """Writes the updated and sorted results to CSV."""
        with open(csv_output_path, mode="w", newline="") as csv_output_file:
            writer = csv.DictWriter(
                csv_output_file, fieldnames=profiler_result_dict_keys())
            writer.writeheader()
            profiler_results_combined = [self.profiler_result_control] + \
                self.profiler_results_success + self.profiler_results_failed
            for profiler_result in profiler_results_combined:
                writer.writerow(profiler_result.profiler_result_dict)

    def get_dispatch_shape(self) -> List[int]:
        b = int(
            self.profiler_results_success[0].profiler_result_dict[PROFILER_RESULT_KEYS.B])
        m = int(
            self.profiler_results_success[0].profiler_result_dict[PROFILER_RESULT_KEYS.M])
        n = int(
            self.profiler_results_success[0].profiler_result_dict[PROFILER_RESULT_KEYS.N])
        k = int(
            self.profiler_results_success[0].profiler_result_dict[PROFILER_RESULT_KEYS.K])
        if b != 0:
            return [b, m, n, k]
        else:
            return [m, n, k]
