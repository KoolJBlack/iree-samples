import enum
from utils.data_types import TargetBackend
from typing import Optional, List, Tuple

###################################################################################################
# This file contains utilities for interfacing with IREE tools.
###################################################################################################


@enum.unique
class CudaFlavors(str, enum.Enum):
    CLI_A2_HIGHGPU = "cli_a2_highgpu"
    SHARK_DEFAULT = "shark_default"
    CUDA_SM_80 = "cuda_sm_80"

    def __str__(self):
        return self.name


def iree_compile_arguments(target_backend: TargetBackend, compile_flavors: List[str]) -> List[str]:
    """Returns the iree_compile argumetns for a given target backend and compile flavor
    """
    cuda_flavors = {
        CudaFlavors.CLI_A2_HIGHGPU: ['--iree-vm-emit-polyglot-zip=true',
                                     '--iree-llvmcpu-debug-symbols=false'],
        CudaFlavors.SHARK_DEFAULT: ['--iree-llvmcpu-target-cpu-features=host',
                                    '--iree-mhlo-demote-i64-to-i32=false',
                                    '--iree-flow-demote-i64-to-i32',
                                    '--iree-stream-resource-index-bits=64',
                                    '--iree-vm-target-index-bits=64',
                                    '--iree-util-zero-fill-elided-attrs'],
        CudaFlavors.CUDA_SM_80: ['--iree-hal-cuda-llvm-target-arch=sm_80']
    }

    args = []
    if target_backend == TargetBackend.CUDA:
        for flavor in compile_flavors:
            args.extend(cuda_flavors[flavor])

    # Return unique arguments list
    return list(set(args))


class BenchmarkTimeout:
    """Class to for enforcing timeouts on benchmarks."""

    def __init__(self,
                 base_time_multiple: float = 1.15,
                 min_benchmark_time_s: float = 10,
                 max_benchmark_time_s: float = 20,
                 max_tolerance: float = 1.05):
        # min time such that we never cutoff below that (for noise)
        # max time such that if the ratio times base time is the max time, we use that or the base time (to prevent excessively long benchmarks)
        self.base_time_s = None
        self.base_time_multiple = base_time_multiple
        self.min_benchmark_time_s = min_benchmark_time_s
        self.max_benchmark_time_s = max_benchmark_time_s

    def set_base_time(self, base_time_s: float):
        """Set base time in seconds. """
        self.base_time_s = base_time_s

    def get_time_limit(self) -> Optional[float]:
        """Returns the benchmark timeout in seconds if possible, or none if there is no limit"""
        if not self.base_time_s:
            return None
        else:
            ratio_time_limit = self.base_time_multiple * self.base_time_s
            if ratio_time_limit < self.min_benchmark_time_s:
                return self.min_benchmark_time_s
            elif ratio_time_limit > self.max_benchmark_time_s:
                return max(self.max_benchmark_time_s, self.base_time_s * self.max_tolerance)
            return ratio_time_limit
