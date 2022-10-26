.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


Installation
============

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
