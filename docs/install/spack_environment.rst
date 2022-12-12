.. ############################################################################
.. spack_environment.rst
.. ===========
.. Author : E. Orliac @EPFL
.. ############################################################################


Spack environment
#################

You can use `Spack <https://spack.io/>`_ to install all the dependencies required
to run ``bluebild++`` setting up an `environment <https://spack.readthedocs.io/en/latest/environments.html>`_.

This documentation assumes Spack v0.19.


Building the Spack environment
==============================

We assume that Spack is available under ``$SPACK_ROOT``, that Spack is activated 
(otherwise ``source $SPACK_ROOT/share/spack/setup-env.sh``), and that we build a
Spack environment called ``bb-env``.

The environment can be built based on a template ``spack.yaml`` manifest file, available
from here. Simply tune it to fit your environment.

Then, you can create the environment with::

  spack env create bb-env spack.yaml

Once created, the environment can be activated with::

  spack env activate -p bb-env

Then, one needs to build it::

  spack concretize -f
  spack install


To deactivate the environment, execute::

  spack env deactivate

or::

  despacktivate

.. hint::
   if you aim to share the Spack environment with other users, build it
   in a location where others have read access and generate Shell scripts to
   activate and deactivate the Spack environment. That way people can use the 
   environment without the need to have Spack installed::

     spack env activate bb-env --sh -p  > $SPACK_ROOT/var/spack/environments/bb-env/activate_bb-env.sh
     spack env deactivate --sh > $SPACK_ROOT/var/spack/environments/bb-env/deactivate_bb-env.sh


Using the Spack environment
===========================
A Spack environment can be activated with::

  spack env activate -p my-env

Alternatively, if you generated a Shell script ``activate-my-env.sh`` to load
the Spack environment to share it with other users as presented in the above section,
simply source that file::

  source /path/to/activate-my-env.sh

To deactivate
