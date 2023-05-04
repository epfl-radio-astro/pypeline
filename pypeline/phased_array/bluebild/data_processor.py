# #############################################################################
# data_processor.py
# =================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Data processors.
"""

import imot_tools.math.linalg as pylinalg
import imot_tools.util.argcheck as chk
import numpy as np

import pypeline.core as core
import pypeline.phased_array.data_gen.statistics as vis
import pypeline.phased_array.bluebild.gram as gram
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import typing as typ
import scipy.linalg as linalg
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

    @chk.check(dict(N_eig=chk.is_integer, cluster_centroids=chk.has_reals))
    def __init__(self, N_eig, cluster_centroids, ctx=None, filter_negative_eigenvalues=True):
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
        self._cluster_centroids = np.array(cluster_centroids, copy=False)
        self._ctx = ctx
        self._filter_negative_eigenvalues = filter_negative_eigenvalues

    @chk.check(
        dict(
            S=chk.is_instance(vis.VisibilityMatrix),
            XYZ=chk.is_instance(instrument.InstrumentGeometry),
            W=chk.is_instance(beamforming.BeamWeights),
            wl=chk.is_real,
        )
    )
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
        broken_row_id = np.flatnonzero(np.isclose(np.sum(S.data, axis=0), np.sum(S.data, axis=1)))
        if broken_row_id.size:
            working_row_id = list(set(np.arange(N_beam)) - set(broken_row_id))
            idx = np.ix_(working_row_id, working_row_id)
            S, W = S.data[idx], W.data[:, working_row_id]
        else:
            S, W = S.data, W.data

        if self._ctx is not None:
            D, V, cluster_idx = self._ctx.intensity_field_data(self._N_eig, np.array(XYZ.data, order='F'), np.array(W.data, order='F'),
                                                               wl, S, self._cluster_centroids)
           
        else:
            G = gram.GramBlock().compute(XYZ.data, W, wl)

            # Functional PCA
            if not np.allclose(S, 0):

                # Default case: use Imot_Tools wrapper for Scipy linalg eigh that filters out
                #               negative eigen values
                if self._filter_negative_eigenvalues:
                    D, V = pylinalg.eigh(S, G, tau=1, N=self._N_eig)

                # Copied from Imot_Tools but here keeping all eigen pairs (hence no padding/selection required)
                else:
                    tau = 1
                    N=self._N_eig
                    try:
                        D, V = linalg.eigh(S, G)
                        idx = np.argsort(D)[::-1]
                        D, V = D[idx], V[:, idx]
                    except linalg.LinAlgError:
                        raise ValueError("Parameter[B] is not PSD.")

            else:  # S is broken beyond use
                D, V = np.zeros(self._N_eig), 0

            # Determine energy-level clustering
            if self._filter_negative_eigenvalues:
                cluster_dist = np.absolute(D.reshape(-1, 1) - self._cluster_centroids.reshape(1, -1))
                cluster_idx = np.argmin(cluster_dist, axis=1)
            else:
                # Cluster positive part as normal, then add an additional level to contain
                # all negative eigenvalues
                cluster_dist = np.absolute(D[D > 0.0].reshape(-1, 1) - self._cluster_centroids.reshape(1, -1))
                cluster_idx_ = np.argmin(cluster_dist, axis=1)
                cluster_idx  = np.zeros(D.shape[0], dtype=cluster_idx_.dtype)
                np.put(cluster_idx, np.arange(cluster_idx_.shape[0]), cluster_idx_)
                np.put(cluster_idx, np.arange(cluster_idx_.shape[0], cluster_idx.shape[0]), cluster_idx_[-1] + 1)

        # Add broken BEAM_IDs
        if broken_row_id.size:
            V_aligned = np.zeros((N_beam, self._N_eig), dtype=V.dtype)
            V_aligned[working_row_id] = V
        else:
            V_aligned = V

        # Scale D by the number of non-zero visibilities
        D /= np.count_nonzero(S.data)
   
        return D, V_aligned, cluster_idx


class SensitivityFieldDataProcessorBlock(DataProcessorBlock):
    """
    Data processor for computing sensitivity fields.
    """

    @chk.check("N_eig", chk.is_integer)
    def __init__(self, N_eig, ctx=None):
        """
        Parameters
        ----------
        N_eig : int
            Number of eigenpairs to output after PCA decomposition.
        ctx: :py:class:`~bluebild.Context`
            Bluebuild context. If provided, will use bluebild module for computation.

        Notes
        -----
        `N_eig` should preferably be set by calling the
        :py:meth:`~pypeline.phased_array.bluebild.parameter_estimator.SensitivityFieldParameterEstimator.infer_parameters`
        method from
        :py:class:`~pypeline.phased_array.bluebild.parameter_estimator.SensitivityFieldParameterEstimator`.
        """
        if N_eig <= 0:
            raise ValueError("Parameter[N_eig] must be positive.")

        super().__init__()
        self._N_eig = N_eig
        self._ctx = ctx

    @chk.check(
        dict(
            XYZ=chk.is_instance(instrument.InstrumentGeometry),
            W=chk.is_instance(beamforming.BeamWeights),
            wl=chk.is_real,
        )
    )
    def __call__(self, XYZ, W, wl):
        """
        fPCA decomposition and data formatting for
        :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.FieldSynthesizerBlock` objects.

        Parameters
        ----------
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

        Examples
        --------
        .. testsetup::

           from pypeline.phased_array.bluebild.gram import GramMatrix
           from pypeline.phased_array.bluebild.data_processor import SensitivityFieldDataProcessorBlock
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

           # Some random positive-definite Gram matrix
           >>> N_beam = 5
           >>> beam_idx = pd.Index(range(N_beam), name='BEAM_ID')
           >>> G = GramMatrix(hermitian_array(N_beam) + 100*np.eye(N_beam), beam_idx)

           # Get compact energy level descriptors.
           >>> S_dp = SensitivityFieldDataProcessorBlock(N_eig=2)
           >>> D, V = S_dp(G)

           >>> np.around(D, 6)
           array([9.2e-05, 9.4e-05])
        """

        if self._ctx is not None:
            return self._ctx.sensitivity_field_data(self._N_eig, np.array(XYZ.data, order='F'), np.array(W.data, order='F'), wl)
        else:
            G = gram.GramBlock()(XYZ, W, wl)
            N_beam = len(G.data)
            D, V = pylinalg.eigh(G.data, np.eye(N_beam), tau=1, N=self._N_eig)
            Dg = 1 / (D ** 2)
            return Dg, V


class VirtualVisibilitiesDataProcessingBlock(DataProcessorBlock):
    r"""
    Data processor for transforming the fPCA decomposition returned by :py:class:`~pypeline.phased_array.bluebild.data_processor.IntensityFieldDataProcessorBlock`
    into a set of virtual visibilities to be fed to the NUFFT synthesizer for imaging.
    """

    @chk.check(dict(N_eig=chk.is_integer))
    def __init__(self, N_eig: int, filters: typ.Tuple[str, ...] = ('lsq', 'std', 'sqrt')):
        r"""

        Parameters
        ----------
        N_eig: int
            Number of eigenpairs in the fPCA decomposition.
        filters: Tuple[str, ...]
            Filters to be applied to the spectrum ``D`` of the fPCA. Possible values are: ``dict(lsq=D, std=np.ones(D.size, np.float), sqrt=np.sqrt(D), inv=1 / D)``.
        """
        if N_eig <= 0:
            raise ValueError("Parameter [N_eig] must be positive.")

        super().__init__()
        self.filters = filters
        self._N_eig = N_eig

    def __call__(self, D: np.ndarray, V: np.ndarray, W: typ.Optional[beamforming.MatchedBeamformerBlock] = None,
                 cluster_idx: typ.Optional[np.ndarray] = None) -> np.ndarray:
        r"""
        Filter the fPCA eigenlevels and transform them into virtual visibilities. If a beamforming matrix is provided the
        filtered eigenlevels are also uncompressed (i.e. beamforming reversed).

        Parameters
        ----------
        D: np.ndarray
            (N_eig,) positive eigenvalues.
        V: np.ndarray
            (N_antenna, N_eig) --or (N_beam, N_eig) with beamforming-- complex-valued eigenvectors.
        W: Optional[pypeline.phased_array.beamforming.MatchedBeamformerBlock]
            (N_antenna, N_beam) optional beamforming matrix.
        cluster_idx: Optional[np.ndarray]
            (N_eig,) cluster indices defining each eigenlevel.

        Returns
        -------
        virtual_vis_stack: np.ndarray
         (N_filter, N_eig, N_antenna, N_antenna) stack of (N_antenna, N_antenna) virtual visibilities.
        """
        Filtered_eigs = dict(lsq=D, std=np.ones(D.size, D.dtype), sqrt=np.sqrt(D), inv=1 / D)
        if W is not None:
            W = sparse.csr_matrix(W.data)
            V_unbeamformed = np.asarray(W * V)
            virtual_vis_stack = np.zeros((len(self.filters), np.unique(cluster_idx).size, W.shape[0], W.shape[0]),
                                         dtype=np.complex128)
        else:
            V_unbeamformed = V
            virtual_vis_stack = np.zeros((len(self.filters), np.unique(cluster_idx).size, V.shape[0], V.shape[0]),
                                         dtype=np.complex128)
        if cluster_idx is None:
            cluster_idx = np.zeros(self._N_eig)

        for k, filter in enumerate(self.filters):
            for i in np.unique(cluster_idx):
                filtered_eig = Filtered_eigs[filter][i == cluster_idx]
                virtual_vis_stack[k, i] = (V_unbeamformed[:, i == cluster_idx] * filtered_eig[None, :]) \
                                          @ V_unbeamformed[:, i == cluster_idx].transpose().conj()
        return virtual_vis_stack
