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
    
eigen_include_dir = os.getenv("EIGEN3_INCLUDE_DIR", "/opt/homebrew/include/eigen3")
if not os.path.exists(eigen_include_dir):
    print(f"Warning: Could not locate Eigen directory at {eigen_include_dir}. Please ensure Eigen is installed.")
    eigen_include_dir = None

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
    name='learner_py',
    version='0.1.0',
    maintainer='Sean McGrath',
    maintainer_email='sean.mcgrath514@gmail.com',
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
        'pybind11>=2.6.0'
    ],
    setup_requires=[
        'numpy>=1.15',
        'pybind11>=2.6.0'
    ],
    cmdclass={'build_ext': build_ext},
    zip_safe=False
)

