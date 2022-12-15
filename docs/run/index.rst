.. ############################################################################
.. index.rst
.. =========
.. Author : E. Orliac @EPFL
.. ############################################################################

######################
Running ``bluebild++``
######################

Running from standard installation
==================================
If you either went through a :ref:`ref-manual-installation` or that you use
a :ref:`ref-spack-environment`, you are ready to run the example pipelines
provided with the source code. They are located in the ``examples`` directory
of your ``bluebild++`` installation and are separated between ``examples/real_data``
and ``examples/simulation``::

  python ./examples/simulation/lofar_bootes_nufft3_cpp

or::

  python ../examples/real_data/lofar_bootes_ss.py


Running from containers
=======================


Switching between GPU and CPU
=============================

``bluebild++`` will always use a GPU device for computing, if available. But
you can easily force it to compute on CPU by switching from "AUTO"matic mode
to "CPU" mode with::

  # Create context with selected processing unit.
  # Options are "AUTO", "CPU" and "GPU".
  ctx = bluebild.Context("CPU")

instead of the default::

  ctx = bluebild.Context("AUTO")


