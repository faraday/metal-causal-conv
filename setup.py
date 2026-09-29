# SPDX-License-Identifier: Apache-2.0
"""
Build script for the Metal CausalConvWithState PyTorch extension.

Usage:
    python -m pip install --no-build-isolation -e .
"""
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Building needs the Apple SDK and PyTorch headers. MPS availability is a
# runtime requirement and is checked by the native Metal dispatch.

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
        '-std=c++20',
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
    name='metal-causal-conv',
    version='0.1.0',
    description='Stateful causal depthwise convolution Metal kernel for Apple Silicon with PyTorch/MPS',
    author='Çağatay Çallı',
    author_email='cagataycalli@gmail.com',
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
