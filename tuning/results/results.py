import enum
from typing import Optional, List, Tuple
from pathlib import Path
import csv

from utils.data_types import DispatchConfig, Dispatch
from iree.compiler import CompilerToolError
from iree.runtime.benchmark import BenchmarkResult, BenchmarkToolError

###################################################################################################
# Libraries for organizing and saving results from profile runs.
###################################################################################################


"""Enum for profiler result dict keys."""


class PROFILER_RESULT_KEYS(str, enum.Enum):
    CONFIG_INDEX = "config_index"
    BENCHMARK_NAME = "benchmark_name"
    TILE_SIZE = "tile_sizes"
    WORK_GROUP_SIZES = "work_group_sizes"
    PIPELINE = "pipeline"
    PIPELINE_DEPTH = "pipeline_depth"
    IDENTIFIER = "identifier"
    B = "b"
    M = "m"
    N = "n"
    K = "k"
    BENCHMARK_REPETITIONS = "benchmark_repetitions"
    ITERATIONS = "iterations"
    PERCENTAGE_OF_PEAK = "percent_of_peak"
    TIME_MEAN_MS = "time_mean_milliseconds"
    CPU_TIME_MEAN_MS = "cpu_time_mean_milliseconds"
    # TIME_MEDIAN_MS = "time_median_milliseconds"
    # CPU_TIME_MEDIAN_MS = "cpu_time_median_milliseconds"
    TIME_STD_MS = "time_std_milliseconds"
    CPU_TIME_STD_MS = "cpu_time_std_milliseconds"
    # TIME_CV_MS = "time_cv_milliseconds"
    # CPU_TIME_CV_MS = "cpu_time_cv_milliseconds"
    COMPILATION_TIME_S = "compilation_time_seconds"
    BENCHMARK_TIME_S = "benchmark_time_seconds"
    ERROR = "error"


def profiler_result_dict_keys() -> dict:
    return [e.value for e in PROFILER_RESULT_KEYS]

class ProfilerResult:
    """Stores the results of each profiler run and its config."""

    def __init__(self,
                 config_index: int,
                 config: DispatchConfig,
                 compilation_successful: bool,
                 benchmark_successful: bool,
                 benchmark_results: List[BenchmarkResult] = [],
                 compiler_error: Optional[CompilerToolError] = None,
                 benchmark_error: Optional[BenchmarkToolError] = None,
                 compilation_time: Optional[float] = None,
                 benchmark_time: Optional[float] = None):
        self.config_index = config_index
        self.config = config
        self.compilation_successful = compilation_successful
        self.benchmark_successful = benchmark_successful
        self.benchmark_results = benchmark_results
        self.compiler_error = compiler_error
        self.benchmark_error = benchmark_error
        self.compilation_time = compilation_time
        self.benchmark_time = benchmark_time

    def set_percentage_of_peak(self, percentage_of_peak: float):
        self.profiler_result_dict[PROFILER_RESULT_KEYS.PERCENTAGE_OF_PEAK] = percentage_of_peak

    @staticmethod
    def create_with_result(config_index: int, config: DispatchConfig, benchmark_results: List[BenchmarkResult], compilation_time: float, benchmark_time: float):
        return ProfilerResult(config_index, config, True, True, benchmark_results, compilation_time=compilation_time, benchmark_time=benchmark_time)

    @staticmethod
    def create_failed_compilation(config_index: int, config: DispatchConfig, compiler_error: CompilerToolError, compilation_time: float):
        return ProfilerResult(config_index, config, False, False, None, compiler_error=compiler_error, compilation_time=compilation_time)

    @staticmethod
    def create_failed_benchmark(config_index: int, config: DispatchConfig, benchmark_error: BenchmarkToolError, compilation_time: float, benchmark_time: float):
        return ProfilerResult(config_index, config, True, False, None, benchmark_error=benchmark_error, compilation_time=compilation_time, benchmark_time=benchmark_time)

    @staticmethod
    def create_from_dict(profiler_result_dict: dict):
        return ProfilerResult(profiler_result_dict)


def build_profiler_result_dict(profiler_result: ProfilerResult, dispatch : Dispatch) -> dict:
    def strip_ms(raw_time: str):
        return raw_time.split(" ")[0]

    config = profiler_result.config
    profiler_result_dict = {
        PROFILER_RESULT_KEYS.CONFIG_INDEX: profiler_result.config_index,
        PROFILER_RESULT_KEYS.TILE_SIZE: str(config.tile_size)[1:-1],
        PROFILER_RESULT_KEYS.WORK_GROUP_SIZES: str(config.workgroup_size)[1:-1],
        PROFILER_RESULT_KEYS.PIPELINE: config.pipeline_name,
        PROFILER_RESULT_KEYS.PIPELINE_DEPTH: config.pipeline_depth,
        PROFILER_RESULT_KEYS.IDENTIFIER: config.operation,
        PROFILER_RESULT_KEYS.B: dispatch.b,
        PROFILER_RESULT_KEYS.M: dispatch.m,
        PROFILER_RESULT_KEYS.N: dispatch.n,
        PROFILER_RESULT_KEYS.K: dispatch.k,
        PROFILER_RESULT_KEYS.COMPILATION_TIME_S: profiler_result.compilation_time,
        PROFILER_RESULT_KEYS.BENCHMARK_TIME_S: profiler_result.benchmark_time,
    }

    err = None
    if not profiler_result.compilation_successful:
        err = str(profiler_result.compiler_error).replace('\n', '|')
    elif not profiler_result.benchmark_successful:
        err = str(profiler_result.benchmark_error)
    else:
        # Pull key benchmark metrics
        benchmark_results = profiler_result.benchmark_results
        if len(benchmark_results) == 1:
            # There is only one result
            benchmark_result_mean = benchmark_results[0]
            benchmark_name = benchmark_results[0].benchmark_name
            iterations = benchmark_results[0].iterations

            profiler_result_dict.update({
                PROFILER_RESULT_KEYS.BENCHMARK_NAME: benchmark_name,
                PROFILER_RESULT_KEYS.BENCHMARK_REPETITIONS: 1,
                PROFILER_RESULT_KEYS.ITERATIONS: iterations,
                PROFILER_RESULT_KEYS.TIME_MEAN_MS: strip_ms(benchmark_result_mean.time),
                PROFILER_RESULT_KEYS.CPU_TIME_MEAN_MS: strip_ms(benchmark_result_mean.cpu_time),
                # PROFILER_RESULT_KEYS.TIME_MEDIAN_MS: "N/A",
                # PROFILER_RESULT_KEYS.CPU_TIME_MEDIAN_MS: "N/A",
                PROFILER_RESULT_KEYS.TIME_STD_MS: "N/A",
                PROFILER_RESULT_KEYS.CPU_TIME_STD_MS: "N/A",
                # PROFILER_RESULT_KEYS.TIME_CV_MS: "N/A",
                # PROFILER_RESULT_KEYS.CPU_TIME_CV_MS: "N/A",
            })
        else:
            benchmark_result_mean = benchmark_results[-4]
            benchmark_result_median = benchmark_results[-3]
            benchmark_result_std = benchmark_results[-2]
            benchmark_result_cv = benchmark_results[-1]
            benchmark_results_remaining = benchmark_results[:-4]
            benchmark_name = benchmark_results_remaining[0].benchmark_name
            iterations = benchmark_results_remaining[0].iterations

            profiler_result_dict.update({
                PROFILER_RESULT_KEYS.BENCHMARK_NAME: benchmark_name,
                PROFILER_RESULT_KEYS.BENCHMARK_REPETITIONS: len(benchmark_results_remaining),
                PROFILER_RESULT_KEYS.ITERATIONS: iterations,
                PROFILER_RESULT_KEYS.TIME_MEAN_MS: strip_ms(benchmark_result_mean.time),
                PROFILER_RESULT_KEYS.CPU_TIME_MEAN_MS: strip_ms(benchmark_result_mean.cpu_time),
                # PROFILER_RESULT_KEYS.TIME_MEDIAN_MS: strip_ms(benchmark_result_median.time),
                # PROFILER_RESULT_KEYS.CPU_TIME_MEDIAN_MS: strip_ms(benchmark_result_median.cpu_time),
                PROFILER_RESULT_KEYS.TIME_STD_MS: strip_ms(benchmark_result_std.time),
                PROFILER_RESULT_KEYS.CPU_TIME_STD_MS: strip_ms(benchmark_result_std.cpu_time),
                # PROFILER_RESULT_KEYS.TIME_CV_MS: strip_ms(benchmark_result_cv.time),
                # PROFILER_RESULT_KEYS.CPU_TIME_CV_MS: strip_ms(benchmark_result_cv.cpu_time),
            })

    profiler_result_dict.update({PROFILER_RESULT_KEYS.ERROR: err})
    return profiler_result_dict

class ProfilerResultsWriter:
    """Class for writing benchmark results to CSV."""

    def __init__(self, output_csv_path: Path, dispatch: Dispatch):
        self.output_csv_path = output_csv_path
        self.dispatch = dispatch
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
            profiler_result_dict = build_profiler_result_dict(profiler_result, self.dispatch)
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
            self.profiler_results_success[0].profiler_result_dict[PROFILER_RESULT_KEYS.CPU_TIME_MEAN_MS])
        for profiler_result in self.profiler_results_success:
            percentage_of_peak = float(
                profiler_result.profiler_result_dict[PROFILER_RESULT_KEYS.CPU_TIME_MEAN_MS]) / peak_time_ms
            profiler_result.set_percentage_of_peak(percentage_of_peak)

        # Separate the control from the successful profiles
        self.profiler_result_control = [
            control for control in self.profiler_results_success if control.profiler_result_dict[PROFILER_RESULT_KEYS.IDENTIFIER] == "control"][0]
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

