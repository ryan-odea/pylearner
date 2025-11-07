import os
import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy as np
import pybind11

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user
    def __str__(self):
        return pybind11.get_include(self.user)

if sys.platform == "darwin":
    extra_compile_args = ['-O3', '-std=c++11', '-Xpreprocessor', '-fopenmp']
    extra_link_args = ['-lomp']
else:
    extra_compile_args = ['-O3', '-std=c++14', '-fopenmp']
    extra_link_args = ['-fopenmp']

vendored_eigen = os.path.join(os.path.dirname(__file__), "vendor", "eigen3")
if os.path.exists(vendored_eigen):
    eigen_include_dir = vendored_eigen
    print(f"Using vendored Eigen directory: {eigen_include_dir}")
else:
    eigen_include_dir = os.getenv("EIGEN3_INCLUDE_DIR")
    if eigen_include_dir and os.path.exists(eigen_include_dir):
        print(f"Using Eigen directory from EIGEN3_INCLUDE_DIR: {eigen_include_dir}")
    else:
        possible_paths = [
            "/opt/homebrew/include/eigen3",  # macOS Homebrew (Apple Silicon)
            "/usr/local/include/eigen3",       # macOS Intel
            "/usr/include/eigen3"              # Linux
        ]
        eigen_include_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                eigen_include_dir = path
                break
        if eigen_include_dir:
            print(f"Using system Eigen directory: {eigen_include_dir}")
        else:
            print("Warning: Could not locate Eigen directory. Please vendor Eigen or ensure it is installed.")

include_dirs = [
    get_pybind_include(),
    get_pybind_include(user=True),
    np.get_include()
]
if eigen_include_dir:
    include_dirs.append(eigen_include_dir)

ext_modules = [
    Extension(
        'pylearner.learner_ext',
        sources=['pylearner/bindings.cpp', 'pylearner/kernels.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='learner-py',
    version='1.0.3',
    author='Sean McGrath, Ryan ODea, Cenhao Zhu, and Rui Duan',
    maintainer='Sean McGrath',
    maintainer_email='ryan.odea@psi.ch',
    description=("""Implements transfer learning methods for low-rank matrix estimation.
                 These methods leverage similarity in the latent row and column spaces between
                 the source and target populations to improve estimation in the target population. 
                 The methods include the LatEnt spAce-based tRaNsfer lEaRning (LEARNER) method and 
                 the direct projection LEARNER (D-LEARNER) method described by McGrath et al. 
                 (2024) <doi:10.48550/arXiv.2412.20605>."""),
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        'numpy>=1.15',
        'pandas>=1.0',
        'pybind11>=2.6.0',
        'screenot>=0.0.2'
    ],
    setup_requires=[
        'numpy>=1.15',
        'pybind11>=2.6.0',
        'screenot>=0.0.2'
    ],
    cmdclass={'build_ext': build_ext},
    zip_safe=False
)
