.. ############################################################################
.. spack_environment.rst
.. ===========
.. Author : E. Orliac @EPFL
.. ############################################################################

.. _ref-spack-environment:

Spack environment
#################

You can use `Spack <https://spack.io/>`_ to install the dependencies required
to run ``bluebild++`` setting up a Spack `environment <https://spack.readthedocs.io/en/latest/environments.html>`_.

.. note::
   The documentation assumes Spack v0.19.


Building the Spack environment
==============================

We assume that Spack is available under ``$SPACK_ROOT``, that Spack is activated 
(otherwise ``source $SPACK_ROOT/share/spack/setup-env.sh``), and that we build a
Spack environment called ``bb-env``. Templates of the Spack environment manifest
files (``spack.yaml`` files) can be found under our
`SKA Spack environments repository <https://github.com/epfl-radio-astro/ska-spack-env>`_.
Simply tune one of those template to fit your working environment.

You can create the environment with::

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

.. _ref-spack-environment-hint:

.. hint::
   if you aim to share the Spack environment with other users, build it
   in a location where others have read access and generate Shell scripts to
   activate and deactivate the Spack environment. That way people can use the 
   environment without the need to have Spack installed::

     spack env activate bb-env --sh -p  > $SPACK_ROOT/var/spack/environments/bb-env/activate_bb-env.sh
     spack env deactivate --sh > $SPACK_ROOT/var/spack/environments/bb-env/deactivate_bb-env.sh


Using/sharing the Spack environment
===================================

If you built the Spack environment yourself, you can either use the ``spack env activate``
and the ``spack env deactivate`` commands to activate and deactivate the environment.
If you followed the hint above, other users can share the environment with you. To
activate it::

  source $SPACK_ROOT/var/spack/environments/bb-env/activate_bb-env.sh

and to deactivate it::

  source $SPACK_ROOT/var/spack/environments/bb-env/deactivate_bb-env.sh
