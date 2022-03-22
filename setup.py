#!/usr/bin/env python3

# #############################################################################
# setup.py
# ========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Pypeline setup script
"""

from setuptools import find_packages
from skbuild import setup
import os

bluebild_gpu = str(os.getenv('BLUEBILD_GPU', 'CUDA'))

setup(setup_requires=["pbr"], pbr=True,
    packages=(find_packages() + ['bluebild']),
    package_dir={'bluebild': 'src/bluebild/python/bluebild'},
    cmake_source_dir='src/bluebild',
    cmake_args=['-DBLUEBILD_GPU=' + bluebild_gpu],
)
