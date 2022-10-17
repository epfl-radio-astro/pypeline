from skbuild import setup
import os
import pathlib
from setuptools import find_packages
import shlex


current_dir = pathlib.Path(__file__).parent.resolve()
with open(str(current_dir) + '/VERSION') as f:
    version = f.readline().strip()

bluebild_gpu = str(os.getenv('BLUEBILD_GPU', 'CUDA'))
bluebild_cmake_args = str(os.getenv('BLUEBILD_CMAKE_ARGS', ''))
bluebild_cmake_args_list = shlex.split(bluebild_cmake_args) if bluebild_cmake_args else []

setup(
    name="bluebild",
    version=version,
    description="Bluebild imaging algorithm",
    packages=['bluebild'],
    package_dir={"": "python"},
    cmake_install_dir="python", # must match package dir name. Otherwise, installed libraries are seen as independent data
    include_package_data=True,
    python_requires=">=3.6",
    cmake_args=['-DBLUEBILD_GPU=' + bluebild_gpu, '-DBUILD_SHARED_LIBS=ON', '-DBLUEBILD_INSTALL=PYTHON'] + bluebild_cmake_args_list
)
