from skbuild import setup
import os

bluebild_gpu = str(os.getenv('BLUEBILD_GPU', 'CUDA'))

setup(
    name="bluebild",
    version="0.1.0",
    description="Bluebild imaging algorithm",
    packages=['bluebild'],
    package_dir={'': 'python'},
    cmake_args=['-DBLUEBILD_GPU=' + bluebild_gpu, '-DBUILD_SHARED_LIBS=ON'],
)

