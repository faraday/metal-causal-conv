"""
Build script for the Metal CausalConvWithState PyTorch extension.

Usage:
    pip install -e .
"""
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Only build for MPS (Apple Silicon)
assert hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(), \
    "MPS (Apple Silicon) is required to build this extension"

# Handle .mm (Objective-C++) files
from distutils.unixccompiler import UnixCCompiler
if '.mm' not in UnixCCompiler.src_extensions:
    UnixCCompiler.src_extensions.append('.mm')
    UnixCCompiler.language_map['.mm'] = 'objc'

# Ensure Ninja is used if available
os.environ['USE_NINJA'] = '1'

# Compile flags for Objective-C++ with Metal framework
extra_compile_args = {
    'cxx': [
        '-std=c++17',
        '-O3',
    ],
}

# Metal/MPS framework flags
os.environ['CFLAGS'] = os.environ.get('CFLAGS', '') + ' -framework Metal -framework Foundation -framework MetalPerformanceShaders -ObjC++'
os.environ['LDFLAGS'] = os.environ.get('LDFLAGS', '') + ' -framework Metal -framework Foundation -framework MetalPerformanceShaders'

ext_modules = [
    CppExtension(
        name='causal_conv_metal_cpp',
        sources=['src/metal_causal_conv/causal_conv_metal.mm'],
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name='metal_causal_conv',
    version='0.1.0',
    description='Fused CausalConvWithState + SiLU Metal kernel for Apple Silicon',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.10',
    package_data={
        '': ['*.metal'],
    },
    include_package_data=True,
    zip_safe=False,
)
