import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CUDAExtension

# python setup.py build_ext --inplace
def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    source_cpp = glob.glob(os.path.join(extensions_dir, "distance_loss_pybind.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "distance_loss.cu"))

    sources = source_cpp + source_cuda
    extension = None
    extra_compile_args = {}

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-O2",
            "-g",
        ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "distance_loss",
            sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

setup(
    name="distance_loss",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
