
from pathlib import Path
import argparse
import glob
from typing import Optional, List, Tuple
import math

from results.results import ProfilerResultsReader, ProfilerResultsWriter, PROFILER_RESULT_KEYS
from results.shark_annotate import assemble_shark_config_object

###################################################################################################
# Script for post processing completed profile CSVs. Use for analysis of results. 
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
    parser.add_argument("--shark_config_path",
                        type=Path,
                        help="Path to output a shark config with the top results",
                        required=False)
    return parser.parse_args()

def main(args: argparse.ArgumentParser):
    print(f"Percentage Peak Count: {args.top_count}")
    input_dir = args.input_csv_dir
    output_dir = args.output_csv_dir
    csv_paths = [Path(x) for x in glob.glob(str(input_dir.joinpath("*.csv")))]

    print(f"Found {len(csv_paths)} CSVs to post process in path: {input_dir}")

    result_readers : List[ProfilerResultsReader] = []
    for csv_input_path in csv_paths:
        csv_output_path = output_dir.joinpath(Path(csv_input_path).name)
        print(csv_output_path)

        result_reader = ProfilerResultsReader(csv_input_path)
        result_readers.append(result_reader)
        Path(csv_output_path).parent.mkdir(parents=True, exist_ok=True)
        result_reader.write_updated_results(csv_output_path)
    
    # Sort profile results by dispatch size
    result_readers.sort(key=lambda result_reader: math.prod(result_reader.get_dispatch_shape()))
    
    if args.top_csv_path:
        writer = ProfilerResultsWriter(args.top_csv_path)
        writer.initialize_output_csv(force=True)
        for result_reader in result_readers:        
            top_results = [result_reader.profiler_result_control] + result_reader.profiler_results_success[0:args.top_count]
            print(f"Top results length {len(top_results)} -- {result_reader.get_dispatch_shape()}")

            for result in top_results:
                writer.write_csv_result(result)
            writer.write_empty_line()

    def string_to_int_list(str) -> List[int]:
      return [int(x) for x in str.split(',')]
    
    if args.shark_config_path:
        with open(args.shark_config_path, mode="w", newline="") as shark_config_f:
            for result_reader in result_readers:        
                result_dict = result_reader.profiler_results_success[0].profiler_result_dict
                tile_sizes = result_dict[PROFILER_RESULT_KEYS.TILE_SIZE]
                workgroup_sizes = result_dict[PROFILER_RESULT_KEYS.WORK_GROUP_SIZES]
                shark_config_json = assemble_shark_config_object(
                    string_to_int_list(tile_sizes),
                    string_to_int_list(workgroup_sizes),
                    result_dict[PROFILER_RESULT_KEYS.PIPELINE],
                    int(result_dict[PROFILER_RESULT_KEYS.PIPELINE_DEPTH]),
                    result_dict[PROFILER_RESULT_KEYS.OPERATION],
                    int(result_dict[PROFILER_RESULT_KEYS.B]),
                    int(result_dict[PROFILER_RESULT_KEYS.M]),
                    int(result_dict[PROFILER_RESULT_KEYS.N]),
                    int(result_dict[PROFILER_RESULT_KEYS.K]),
                )
                print(shark_config_json)
                shark_config_f.write(shark_config_json + "\n")

if __name__ == "__main__":
    main(parse_arguments())
