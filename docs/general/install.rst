.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


Installation
============

The following describes how `pipeline` and `bluebild` (with its C++ CPU and
GPU ports) can be installed on the EPFL's GPU cluster 
`izar <https://www.epfl.ch/research/facilities/scitas/hardware/izar/>`_ using
the available GCC 9.3.0 software stack.

* Load the following modules::

    ```
    module purge
    module load gcc/9.3.0-cuda
    module load python/3.7.7
    module load cuda/11.0.2
    module load openblas/0.3.10-openmp
    module load mvapich2/2.3.4
    module load fftw/3.3.8-mpi-openmp
    module load cmake

    ```


After installing `Miniconda <https://conda.io/miniconda.html>`_ or `Anaconda
<https://www.anaconda.com/download/#linux>`_, run the following:

* Install C++ performance libraries::

    $ cd <pypeline_dir>/
    $ conda create --name=pypeline       \
                   --channel=defaults    \
                   --channel=conda-forge \
                   --file=conda_requirements.txt
    $ source pypeline.sh --no_shell

* Install `pypeline`::

    $ cd <pypeline_dir>/
    $ pip install -e .
    $ python3 test.py                # Run test suite (optional, recommended)
    $ python3 setup.py build_sphinx  # Generate documentation (optional)


To launch a Python3 shell containing Pypeline, run ``pypeline.sh``.


Remarks
-------

Pypeline is developed and tested on x86_64 systems running Linux.
It should also run correctly on macOS, but we provide no support for this.
