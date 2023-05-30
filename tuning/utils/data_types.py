import enum
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json

###################################################################################################
# This file contains library of enumerations and classes used to build operation descritpions.
###################################################################################################

# Base Types
###################################################################################################


@enum.unique
class DataType(enum.Enum):
    I8 = ("i8", 1)
    I32 = ("i32", 4)
    F32 = ("f32", 4)
    F16 = ("f16", 2)

    def __init__(self, value, bytes_size):
        self.iree_type = value
        self.bytes_size = bytes_size

    @staticmethod
    def from_string(s: str):
        try:
            return DataType[s.capitalize()]
        except KeyError:
            raise ValueError()

# IREE Tools Types
###################################################################################################


@enum.unique
class CompilerFrontend(str, enum.Enum):
    MHLO = "mhlo"
    STABLEHLO = "stablehlo"


@enum.unique
class TargetBackend(str, enum.Enum):
    """ Target backend for IREE Compile
    """
    CUDA = "cuda"

    @staticmethod
    def from_string(s: str):
        try:
            return TargetBackend[s.capitalize()]
        except KeyError:
            raise ValueError()


@enum.unique
class TargetDevice(str, enum.Enum):
    """ Target backend for IREE Run/Benchmark Module/MLIR
    """
    CUDA = "cuda"


@enum.unique
class TargetDriver(str, enum.Enum):
    """ Target backend for IREE Run/Benchmark Module/MLIR
    """
    CUDA = "cuda"

# IREE Model Compilation
###################################################################################################


@dataclass
class CompilationResult:
    config: dict
    flatbuffer_blob: Optional[bytes]
    err: Optional[str]
    compilation_time_s: float

# Profiler Types
###################################################################################################

# Note: this is used in the profiler programs. Needs to match.


@enum.unique
class Pipeline(str, enum.Enum):
    GPU_TENSORCORE_MMASYNC = "LLVMGPUMatmulTensorCoreMmaSync"
    GPU_TENSORCORE = "LLVMGPUMatmulTensorCore"
    GPU_SIMT = "LLVMGPUMatmulSimt"

    def __str__(self):
        return self.name

# Note: this is used in the profiler programs. Needs to match.


@enum.unique
class OperationType(str, enum.Enum):
    MATMUL = "matmul"
    BATCH_MATMUL = "batch_matmul"

    @staticmethod
    def from_string(s: str):
        try:
            return OperationType[s.upper()]
        except KeyError:
            raise ValueError(f"OperationType: {s}")

    def __str__(self):
        return self.name


@dataclass
class Dispatch:
    pipeline_name: Pipeline
    operation: OperationType
    data_type: DataType
    b: Optional[int]
    m: int
    n: int
    k: int


@dataclass
class DispatchConfig:
    tile_size: List[int]
    workgroup_size: List[int]
    pipeline_depth: int

    def __post_init__(self):
        # Ensure workgroup size is in threads not warps. 
        if self.workgroup_size[0] < 32:
            self.workgroup_size[0] *= 32


class DefaultConfig:

    def __init__(self):
        self.tile_size = "default"
        self.workgroup_size = "default"
        self.pipeline_depth = "default"

    def __repr__(self) -> str:
        return "DEFAULT_CONFIG"


"""A blank control config that does not annotate the model."""
DEFAULT_CONFIG = DefaultConfig()


@dataclass
class ProfilerProgram:
    """Defines a run for the profiler and results to collect."""
    name: str
    b: int
    m: int
    n: int
    k: int
    data_type: DataType
    operation_type: OperationType
    target_backend: TargetBackend
    pipeline: Pipeline
    template_mlir_filename: str
    output_csv_filename: str

    def dump_json(self) -> str:
        json_dict = {
            "name": self.name,
            "b": self.b,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "data_type": self.data_type.iree_type,
            "operation_type": self.operation_type.value,
            "target_backend": self.target_backend.value,
            "pipeline": self.pipeline.value,
            "template_mlir_filename": self.template_mlir_filename,
            "output_csv_filename": self.output_csv_filename

        }
        return json.dumps(json_dict)

    @classmethod
    def load_json(cls, json_str: str):
        json_dict = json.loads(json_str)

        data_type = None
        if json_dict["data_type"] == DataType.I8.iree_type:
            data_type = DataType.I8
        elif json_dict["data_type"] == DataType.F16.iree_type:
            data_type = DataType.F16
        elif json_dict["data_type"] == DataType.I32.iree_type:
            data_type = DataType.I32
        elif json_dict["data_type"] == DataType.F32.iree_type:
            data_type = DataType.F32
        else:
            raise RuntimeError("Unknown data type")

        operation_type = None
        if json_dict["operation_type"] == OperationType.MATMUL.value:
            operation_type = OperationType.MATMUL
        elif json_dict["operation_type"] == OperationType.BATCH_MATMUL.value:
            operation_type = OperationType.BATCH_MATMUL
        else:
            raise RuntimeError("Unknown operation type")

        target_backend = TargetBackend.CUDA
        if json_dict["target_backend"] != TargetBackend.CUDA.value:
            raise ValueError("Only target backend CUDA supported")

        pipeline = None
        if json_dict["pipeline"] == Pipeline.GPU_TENSORCORE_MMASYNC.value:
            pipeline = Pipeline.GPU_TENSORCORE_MMASYNC
        elif json_dict["pipeline"] == Pipeline.GPU_TENSORCORE.value:
            pipeline = Pipeline.GPU_TENSORCORE
        elif json_dict["pipeline"] == Pipeline.GPU_SIMT.value:
            pipeline = Pipeline.GPU_SIMT
        else:
            raise RuntimeError("Unknown pipeline")

        return ProfilerProgram(
            name=json_dict["name"],
            b=json_dict["b"],
            m=json_dict["m"],
            n=json_dict["n"],
            k=json_dict["k"],
            data_type=data_type,
            operation_type=operation_type,
            target_backend=target_backend,
            pipeline=pipeline,
            template_mlir_filename=json_dict["template_mlir_filename"],
            output_csv_filename=json_dict["output_csv_filename"],
        )
