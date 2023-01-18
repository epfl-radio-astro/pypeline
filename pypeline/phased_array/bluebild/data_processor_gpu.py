# #############################################################################
# data_processor.py
# =================
# Author : Arpan Das [arpan.das@epfl.ch]
# #############################################################################

"""
Data processors.
"""

import imot_tools.math.linalg as pylinalg
import scipy.linalg as linalg
import imot_tools.util.argcheck as chk
import numpy as np
import cupy as cp

import pypeline.core as core
import pypeline.phased_array.data_gen.statistics as vis
import pypeline.phased_array.bluebild.gram as gram
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import typing as typ
import scipy.sparse as sparse



class DataProcessorBlock(core.Block):
    """
    Top-level public interface of Bluebild data processors.
    """

    def __init__(self):
        """

        """
        super().__init__()

    def __call__(self, *args, **kwargs):
        """
        fPCA decomposition and data formatting for
        :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.FieldSynthesizerBlock` objects.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.
        """
        raise NotImplementedError
        
class IntensityFieldDataProcessorBlock(DataProcessorBlock):
    """
    Data processor for computing intensity fields.
    """

    def __init__(self, N_eig, cluster_centroids, ctx=None):
        """
        Parameters
        ----------
        N_eig : int
            Number of eigenpairs to output after PCA decomposition.
        cluster_centroids : array-like(float)
            Intensity centroids for energy-level clustering.
        ctx: :py:class:`~bluebild.Context`
            Bluebuild context. If provided, will use bluebild module for computation.

        Notes
        -----
        Both parameters should preferably be set by calling the
        :py:meth:`~pypeline.phased_array.bluebild.parameter_estimator.IntensityFieldParameterEstimator.infer_parameters`
        method from
        :py:class:`~pypeline.phased_array.bluebild.parameter_estimator.IntensityFieldParameterEstimator`.
        """
        if N_eig <= 0:
            raise ValueError("Parameter[N_eig] must be positive.")

        super().__init__()
        self._N_eig = N_eig
        self._cluster_centroids = cp.array(cluster_centroids, copy=False)
        self._ctx = ctx
        
    def __call__(self, S, XYZ, W, wl):
        """
        fPCA decomposition and data formatting for
        :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.FieldSynthesizerBlock` objects.

        .. todo:: How to deal with ill-conditioned G?

        Parameters
        ----------
        S : :py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`
            (N_beam, N_beam) visibility matrix.
        XYZ : :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            (N_antenna, 3) Cartesian antenna coordinates in any reference frame.
        W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
            (N_antenna, N_beam) synthesis beamweights.
        wl : float
            Wavelength [m] at which to compute the Gram.

        Returns
        -------
        D : :py:class:`~numpy.ndarray`
            (N_eig,) positive eigenvalues.

        V : :py:class:`~numpy.ndarray`
            (N_beam, N_eig) complex-valued eigenvectors.

        cluster_idx : :py:class:`~numpy.ndarray`
            (N_eig,) cluster indices of each eigenpair.

        Examples
        --------
        .. testsetup::

           from pypeline.phased_array.data_gen.statistics import VisibilityMatrix
           from pypeline.phased_array.bluebild.gram import GramMatrix
           from pypeline.phased_array.bluebild.data_processor import IntensityFieldDataProcessorBlock
           import numpy as np
           import pandas as pd
           import scipy.linalg as linalg

           def hermitian_array(N: int) -> np.ndarray:
               '''
               Construct a (N, N) Hermitian matrix.
               '''
               D = np.arange(N)
               Rmtx = np.random.randn(N,N) + 1j * np.random.randn(N, N)
               Q, _ = linalg.qr(Rmtx)

               A = (Q * D) @ Q.conj().T
               return A

           np.random.seed(0)

        .. doctest::

           >>> N_beam = 5
           >>> beam_idx = pd.Index(range(N_beam), name='BEAM_ID')

           # Some random visibility matrix
           >>> S = VisibilityMatrix(hermitian_array(N_beam), beam_idx)

           # Some random positive-definite Gram matrix
           >>> G = GramMatrix(hermitian_array(N_beam) + 100*np.eye(N_beam), beam_idx)

           # Get compact energy level descriptors.
           >>> I_dp = IntensityFieldDataProcessorBlock(N_eig=2,
           ...                                         cluster_centroids=[0., 20.])
           >>> D, V, cluster_idx = I_dp(S, G)

           >>> np.around(D, 2)
           array([0.04, 0.03])

           >>> cluster_idx  # useful for aggregation stage.
           array([0, 0])
        """


        N_beam = len(S.data)

        # Remove broken BEAM_IDs
        broken_row_id = cp.flatnonzero(cp.isclose(cp.sum(S.data, axis=0), cp.sum(S.data, axis=1)))
        if broken_row_id.size:
            working_row_id = list(set(cp.arange(N_beam)) - set(broken_row_id))
            idx = cp.ix_(working_row_id, working_row_id)
            S, W = cp.array(S.data[idx]), cp.array(W.data[:, working_row_id])
        else:
            S, W = cp.array(S.data), cp.array(W.data)

        if self._ctx is not None:
            D, V, cluster_idx = self._ctx.intensity_field_data(self._N_eig, cp.array(XYZ.data, order='F'), cp.array(W.data, order='F'),
                    wl, S, self._cluster_centroids)
        else:
            G = gram.GramBlock().compute(XYZ.data, cp.asnumpy(W), cp.asnumpy(wl))
            G = cp.array(G)

            # Functional PCA
            if not cp.allclose(S, 0):
                D, V = pylinalg.eigh(S.get(), G.get(), tau=1, N=self._N_eig)
                D = cp.array(D)
                V = cp.array(V)
            else:  # S is broken beyond use
                D, V = cp.zeros(self._N_eig), 0

            # Determine energy-level clustering
            cluster_dist = cp.absolute(D.reshape(-1, 1) - self._cluster_centroids.reshape(1, -1))
            cluster_idx = cp.argmin(cluster_dist, axis=1)

        # Add broken BEAM_IDs
        if broken_row_id.size:
            V_aligned = cp.zeros((N_beam, self._N_eig), dtype=cp.complex)
            V_aligned[working_row_id] = V
        else:
            V_aligned = V

        return D, V_aligned, cluster_idx