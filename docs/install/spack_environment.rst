.. ############################################################################
.. spack_environment.rst
.. ===========
.. Author : E. Orliac @EPFL
.. ############################################################################


Spack environment
#################

You can use `Spack <https://spack.io/>`_ to install all the dependencies required
to run ``bluebild++`` setting up an `environment <https://spack.readthedocs.io/en/latest/environments.html>`_.

This documentation is based on Spack v0.19.


Building the Spack environment
==============================

We assumed that Spack is available under ``$SPACK_ROOT``.


Using the Spack environment
===========================
A Spack environment can be activated with::

spack env activate -p my-env

Alternatively, if you generated a Shell script ``activate-my-env.sh`` to load
the Spack environment to share it with other users as presented in the above section,
simply source that file::

source /path/to/activate-my-env.sh

To deactivated 
