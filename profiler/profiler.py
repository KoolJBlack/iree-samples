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

from model_annotation import model_annotation

from iree.compiler import ir, CompilerToolError
from iree.compiler.transforms import ireec as ireec_trans
from iree.runtime import benchmark_module
from iree.runtime.benchmark import BenchmarkResult
import iree.runtime as ireert
import iree.compiler as ireec

from config_generation import Pipeline, OperationType, DataType, generate_configs, dump_shark_config_json

# ====== IREE Specific ======

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
        # print(f"The resulting IR:")
        # print(annotated_model)
        # print(f"Module type {type(annotated_model)}")

        if annotated_model_output_path:
            with open(annotated_model_output_path, "w") as f:
                f.write(mlir_str)
            print(f"Saved annotated mlir to: {annotated_model_output_path}.")

        return annotated_model


def compile_module_to_flatbuffer(
        module,
        target_device: str,
        frontend: str,
        extra_args: List) -> Optional[bytes]:
    """Compiles mlir module test and returns a flatbuffer blob"""
    args = []
    if target_device == "cuda":
        args.extend(['--iree-llvm-target-cpu-features=host', '--iree-mhlo-demote-i64-to-i32=false', '--iree-flow-demote-i64-to-i32',
                    '--iree-hal-cuda-disable-loop-nounroll-wa', '--iree-stream-resource-index-bits=64', '--iree-vm-target-index-bits=64', '--iree-util-zero-fill-elided-attrs'])
        args.append('--iree-hal-cuda-llvm-target-arch=sm_80')
        args.append('-iree-hal-benchmark-dispatch-repeat-count=1000')
        args.extend(extra_args)
    else:
        raise ValueError(
            "Only `cuda` target device is supported for benchmarking")

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
        pass

    return flatbuffer_blob


def run_benchmark_module(
        flatbuffer_blob: bytes, 
        entry_function: str,
        function_input: List[str] = [],
        device: str = "cuda") -> List[BenchmarkResult]:
    # Create a module
    config = ireert.Config(driver_name="cuda")
    vm_module = ireert.VmModule.from_flatbuffer(
        config.vm_instance, flatbuffer_blob)
    print(f"Did this work? {vm_module}")
    funcs = [a for a in vm_module.function_names if a != "__init"]
    for func in funcs:
        print(f"I found these funcs: {func}")

    benchmark_results = benchmark_module(
        vm_module, device=device, benchmark_repetitions=5, batch_size=1000)
    
    return benchmark_results



def dir_path(string) -> Optional[Path]:
    """Returns path to dir if it exists"""
    if os.path.isdir(string):
        return Path(string)
    else:
        # raise NotADirectoryError(string)
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Profile matmuls in IREE")
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
    return parser.parse_args()


def main(args: argparse.ArgumentParser):
    print("Profiler Start")

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
    generated_config_path = artifacts_dir_path.joinpath("generated_config.json")
    vmfb_path = artifacts_dir_path.joinpath("annotated_flatbuffer.vmfb")
    
    # Load template model
    input_model_path = args.template_mlir_model
    template_model_str =""
    with open(input_model_path, "r") as f:
        template_model_str = f.read()

    if not template_model_str:
        raise ValueError("Unable to read tempalte model.")

    configs = generate_configs(pipeline=Pipeline.GPU_TENSORCORE, operation=OperationType.MATMUL, input_shape=[
                               args.m, args.n, args.k], data_type=args.data_type)
    print(f"Generated configs for model: {len(configs)}")

    profiler_results = []
    # For each config, annotate the model, compile and benchmark
    # Configs need to be saved to file for use. 
    # generated_config_path = artifacts_dir_path.joinpath("generated_config.json")
    for index, config in enumerate(configs):
        print(f"Testing config {index}/{len(configs)}")

        # Save the config to file
        print(f"Saved generated config to: {generated_config_path}.")
        # with open(generated_config_path, "wb") as f:
        #     f.write(flatbuffer_blob)
        bytes_written = dump_shark_config_json(config, generated_config_path)
        annotated_module = annotate_mlir_model(
            input_model_str=template_model_str, config_path=generated_config_path)

        # Compile model
        flatbuffer_blob = compile_module_to_flatbuffer(
            str(annotated_module), "cuda", "mhlo", args.extra_compilation_args)

        # # Was compilation successful?
        # if not flatbuffer_blob:
        #     print("Failed to compile. Exiting!")
        #     sys.exit()
        # print(f"The compiled flatbuffer is {len(flatbuffer_blob)} bytes")

        # print(f"Saved vmfb in {vmfb_path}.")
        # with open(vmfb_path, "wb") as f:
        #     f.write(flatbuffer_blob)

        # benchmark_results = run_benchmark_module(flatbuffer_blob, entry_function="forward")

        # print(
        #     f"I reached the end with these many results: {len(benchmark_results)}!")

    # Cleanup any artifacts
    if temp_dir:
        temp_dir.cleanup()

    print("Run Model Annotation Complete!")


if __name__ == "__main__":
    main(parse_arguments())
