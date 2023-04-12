#!/usr/bin/env python3

"""Script for converting from the old csv format to new."""

from pathlib import Path
import csv
import glob

from results import ProfilerResult, ProfilerResultsWriter

profiler_result_keys_old = [
            "config_index",
            "benchmark_name",
            "tile_sizes", "work_group_sizes", "pipeline", "pipeline_depth", "identifier", "b", "m", "n", "k",
            "benchmark_repetitions",
            "iterations",
            "time_mean_milliseconds", "cpu_time_mean_milliseconds",
            "time_median_milliseconds", "cpu_time_median_milliseconds",
            "time_std_milliseconds", "cpu_time_std_milliseconds",
            "time_cv_milliseconds", "cpu_time_cv_milliseconds",
            "compilation_time_seconds", "benchmark_time_seconds",
            "error",
]

profiler_result_keys = [
            "config_index",
            "benchmark_name",
            "tile_sizes", "work_group_sizes", "pipeline", "pipeline_depth", "identifier", "b", "m", "n", "k",
            "benchmark_repetitions",
            "iterations",
            "percent_of_peak",
            "time_mean_milliseconds", "cpu_time_mean_milliseconds",
            "time_median_milliseconds", "cpu_time_median_milliseconds",
            "time_std_milliseconds", "cpu_time_std_milliseconds",
            "time_cv_milliseconds", "cpu_time_cv_milliseconds",
            "compilation_time_seconds", "benchmark_time_seconds",
            "error",
]

def conversion_script():
    print([e.value for e in PROFILER_RESULT_KEYS])
    input_dir = Path("/usr/local/google/home/kooljblack/Code/iree-tmp/tuning/remote/results")
    output_dir = Path("/usr/local/google/home/kooljblack/Code/iree-tmp/tuning/remote/results-fixed")
    csv_paths = [Path(x) for x in glob.glob('/usr/local/google/home/kooljblack/Code/iree-tmp/tuning/remote/results/*.csv', recursive=True)]
 
    for csv_input_path in csv_paths:
        csv_output_path = output_dir.joinpath(Path(csv_input_path).name)
        print(csv_output_path)
        with open(csv_input_path, "r") as csv_input_file:
            with open(csv_output_path, mode="w", newline="") as csv_output_file:
                csv_input_file.readline() # Dump the first line since it contains the column titles
                reader = csv.DictReader(csv_input_file, fieldnames=profiler_result_keys_old)
                writer = csv.DictWriter(csv_output_file, fieldnames=profiler_result_keys)
                writer.writeheader()
                for row_dict in reader:
                    profiler_result = ProfilerResult.create_from_dict(row_dict)
                    writer.writerow(profiler_result.profiler_result_dict)

def main():
    print(f"Hello World")


if __name__ == "__main__":
    main()
