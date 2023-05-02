
from pathlib import Path
import argparse
import glob
from typing import Optional, List, Tuple
import math

from results import ProfilerResultsReader, ProfilerResultsWriter

###################################################################################################
# Script for updating a profiler csv with percentage of peak and reordering the results as such.
###################################################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for updating a profiler csv with percentage of peak and reordering the results as such.")
    parser.add_argument("--input_csv_dir",
                        type=Path,
                        help="Path to input csvs",
                        required=True)
    parser.add_argument("--output_csv_dir",
                        type=Path,
                        help="Path to output csvs",
                        required=True)
    parser.add_argument("--top_csv_path",
                        type=Path,
                        help="Path to output the combined top results",
                        required=False)
    parser.add_argument("--top_count",
                        type=int,
                        help="Path to output the combined top results",
                        required=False,
                        default=10)
    return parser.parse_args()

def main(args: argparse.ArgumentParser):
    print(f"Percentage Peak {args.top_count}")
    input_dir = args.input_csv_dir
    output_dir = args.output_csv_dir
    csv_paths = [Path(x) for x in glob.glob(str(input_dir.joinpath("*.csv")))]

    result_readers : List[ProfilerResultsReader] = []
    for csv_input_path in csv_paths:
        csv_output_path = output_dir.joinpath(Path(csv_input_path).name)
        # print(csv_output_path)

        result_reader = ProfilerResultsReader(csv_input_path)
        result_readers.append(result_reader)
        result_reader.write_updated_results(csv_output_path)
    
    if args.top_csv_path:
        # Sort profile results by dispatch size
        result_readers.sort(key=lambda result_reader: math.prod(result_reader.get_dispatch_shape()))

        writer = ProfilerResultsWriter(args.top_csv_path)
        writer.initialize_output_csv(force=True)
        for result_reader in result_readers:        
            top_results = [result_reader.profiler_result_control] + result_reader.profiler_results_success[0:args.top_count]
            print(f"Top results length {len(top_results)} -- {result_reader.get_dispatch_shape()}")

            for result in top_results:
                writer.write_csv_result(result)
            writer.write_empty_line()


if __name__ == "__main__":
    main(parse_arguments())
