import enum
from utils.data_types import TargetBackend
from typing import Optional, List, Tuple


@enum.unique
class CudaFlavors(str, enum.Enum):
    SHARK_DEFAULT = "cuda"
    CUDA_SM_80 = "cuda_sm_80"

    def __str__(self):
        return self.name

def iree_compile_arguments(target_backend: TargetBackend, compile_flavors: List[str]) -> List[str]:
    """Returns the iree_compile argumetns for a given target backend and compile flavor
    """
    cuda_flavors = {
        CudaFlavors.SHARK_DEFAULT : ['--iree-llvmcpu-target-cpu-features=host',
                          '--iree-mhlo-demote-i64-to-i32=false',
                          '--iree-flow-demote-i64-to-i32',
                          '--iree-stream-resource-index-bits=64',
                          '--iree-vm-target-index-bits=64',
                          '--iree-util-zero-fill-elided-attrs'],
         CudaFlavors.CUDA_SM_80  : ['--iree-hal-cuda-llvm-target-arch=sm_80'] 
    }
    
    args = []
    if target_backend == TargetBackend.CUDA:
        for flavor in compile_flavors:
            args.extend(cuda_flavors[flavor])
    
    # Return unique arguments list
    return list(set(args))


def main():
  args = iree_compile_arguments(TargetBackend.CUDA, [CudaFlavors.CUDA_SM_80])
  print(args)


if __name__ == "__main__":
    main()
