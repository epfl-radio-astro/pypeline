# #############################################################################
# parameter_estimator.py
# ======================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

r"""
Parameter estimators.

Bluebild field synthesizers output :math:`N_{\text{beam}}` energy levels, with :math:`N_{\text{beam}}`
being the height of the visibility/Gram matrices :math:`\Sigma, G`.
We are often not interested in such fined-grained energy decompositions but would rather have 4-5
well-separated energy levels as output.
This is accomplished by clustering energy levels together during the aggregation stage.

As the energy scale depends on the visibilities, it is preferable to infer the cluster centroids
(and any other parameters of interest) by scanning a portion of the data stream.
Subclasses of :py:class:`~pypeline.phased_array.bluebild.parameter_estimator.ParameterEstimator` are
specifically tailored for such tasks.
"""

#import imot_tools.math.linalg as pylinalg
import imot_tools.util.argcheck as chk
import numpy as np
import sklearn.cluster as skcl
import scipy.linalg as linalg
import pypeline.phased_array.data_gen.statistics as vis
import pypeline.phased_array.bluebild.gram as gr


@chk.check(
    dict(
        A=chk.accept_any(chk.has_reals, chk.has_complex),
        B=chk.allow_None(chk.accept_any(chk.has_reals, chk.has_complex)),
        tau=chk.is_real,
        N=chk.allow_None(chk.is_integer),
    )
)
def pylinalg_eigh(A, B=None, tau=1, N=None, check_hermitian=True):
    """
    Solve a generalized eigenvalue problem.

    Finds :math:`(D, V)`, solution of the generalized eigenvalue problem

    .. math::

       A V = B V D.

    This function is a wrapper around :py:func:`scipy.linalg.eigh` that adds energy truncation and
    extra output formats.

    Parameters
    ----------
    A : :py:class:`~numpy.ndarray`
        (M, M) hermitian matrix.
        If `A` is not positive-semidefinite (PSD), its negative spectrum is discarded.
    B : :py:class:`~numpy.ndarray`, optional
        (M, M) PSD hermitian matrix.
        If unspecified, `B` is assumed to be the identity matrix.
    tau : float, optional
        Normalized energy ratio. (Default: 1)
    N : int, optional
        Number of eigenpairs to output. (Default: K, the minimum number of leading eigenpairs that
        account for `tau` percent of the total energy.)

        * If `N` is smaller than K, then the trailing eigenpairs are dropped.
        * If `N` is greater that K, then the trailing eigenpairs are set to 0.

    Returns
    -------
    D : :py:class:`~numpy.ndarray`
        (N,) positive real-valued eigenvalues.

    V : :py:class:`~numpy.ndarray`
        (M, N) complex-valued eigenvectors.

        The N eigenpairs are sorted in decreasing eigenvalue order.

    """
    A = np.array(A, copy=False)
    M = len(A)
    if check_hermitian:
        if not (chk.has_shape([M, M])(A) and np.allclose(A, A.conj().T)):
            raise ValueError("Parameter[A] must be hermitian symmetric.")

    B = np.eye(M) if (B is None) else np.array(B, copy=False)
    if not (chk.has_shape([M, M])(B) and np.allclose(B, B.conj().T)):
        raise ValueError("Parameter[B] must be hermitian symmetric.")

    if not (0 < tau <= 1):
        raise ValueError("Parameter[tau] must be in [0, 1].")

    if (N is not None) and (N <= 0):
        raise ValueError(f"Parameter[N] must be a non-zero positive integer.")

    # A: drop negative spectrum.
    Ds, Vs = linalg.eigh(A)
    idx = Ds > 0
    Ds, Vs = Ds[idx], Vs[:, idx]
    A = (Vs * Ds) @ Vs.conj().T

    # A, B: generalized eigenvalue-decomposition.
    try:
        D, V = linalg.eigh(A, B)

        # Discard near-zero D due to numerical precision.
        idx = D > 0
        D, V = D[idx], V[:, idx]
        idx = np.argsort(D)[::-1]
        D, V = D[idx], V[:, idx]
    except linalg.LinAlgError:
        raise ValueError("Parameter[B] is not PSD.")

    # Energy selection / padding
    idx = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= tau
    D, V = D[idx], V[:, idx]
    if N is not None:
        M, K = V.shape
        if N - K <= 0:
            D, V = D[:N], V[:, :N]
        else:
            D = np.concatenate((D, np.zeros(N - K)), axis=0)
            V = np.concatenate((V, np.zeros((M, N - K))), axis=1)

    return D, V


class ParameterEstimator:
    """
    Top-level public interface of Bluebild parameter estimators.
    """

    def __init__(self):
        """

        """
        super().__init__()

    def collect(self, *args, **kwargs):
        """
        Ingest data to internal queue for inference.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.
        """
        raise NotImplementedError

    def infer_parameters(self):
        """
        Estimate parameters given ingested data.

        Returns
        -------
        tuple
            Parameters as defined by subclasses.
        """
        raise NotImplementedError


class IntensityFieldParameterEstimator(ParameterEstimator):
    """
    Parameter estimator for computing intensity fields.

    Examples
    --------
    Assume we are imaging a portion of the Bootes field with LOFAR's 24 core stations.
    As 24 energy levels exhibit a smooth spectrum, we decide to aggregate them into 4 well-separated
    energy levels through clustering.

    The short script below shows how to use :py:class:`~pypeline.phased_array.bluebild.parameter_estimator.IntensityFieldParameterEstimator`
    to optimally choose cluster centroids.

    .. testsetup::

       import numpy as np
       import astropy.units as u
       import astropy.time as atime
       import astropy.coordinates as coord
       import scipy.constants as constants
       from pypeline.phased_array.bluebild.parameter_estimator import IntensityFieldParameterEstimator
       from pypeline.phased_array.instrument import LofarBlock
       from pypeline.phased_array.beamforming import MatchedBeamformerBlock
       from pypeline.phased_array.bluebild.gram import GramBlock
       from pypeline.phased_array.data_gen.source import from_tgss_catalog
       from pypeline.phased_array.data_gen.statistics import VisibilityGeneratorBlock

       np.random.seed(0)

    .. doctest::

       ### Experiment setup ================================================
       # Observation
       >>> obs_start = atime.Time(56879.54171302732, scale='utc', format='mjd')
       >>> field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
       >>> field_of_view = np.deg2rad(5)
       >>> frequency = 145e6
       >>> wl = constants.speed_of_light / frequency

       # instrument
       >>> N_station = 24
       >>> dev = LofarBlock(N_station)
       >>> mb = MatchedBeamformerBlock([(_, _, field_center) for _ in range(N_station)])
       >>> gram = GramBlock()

       # Visibility generation
       >>> vis = VisibilityGeneratorBlock(sky_model=from_tgss_catalog(field_center, field_of_view, N_src=10),
       ...                                T=8,
       ...                                fs=196000,
       ...                                SNR=np.inf)

       ### Parameter estimation ============================================
       >>> I_est = IntensityFieldParameterEstimator(N_level=4, sigma=0.95)
       >>> t_est = obs_start + np.arange(20) * 400 * u.s  # sample visibilities throughout the 8h observation.
       >>> for t in t_est:
       ...    XYZ = dev(t)
       ...    W = mb(XYZ, wl)
       ...    S = vis(XYZ, W, wl)
       ...    G = gram(XYZ, W, wl)
       ...
       ...    I_est.collect(S, G)  # Store S, G internally until full 8h interval has been sampled.
       ...
       >>> N_eig, c_centroid = I_est.infer_parameters()  # optimal estimate

       # notice that it is less than N_src=10 because sigma=0.95 made
       # it throw away the trailing eigenpairs that account for 5% of
       # the sky's energy.
       >>> N_eig
       7

       >>> np.around(c_centroid, 3)
       array([124.927,  65.09 ,  38.589,  23.256])
    """

    @chk.check(dict(N_level=chk.is_integer, sigma=chk.is_real))
    def __init__(self, N_level, sigma, filter_negative_eigenvalues=False):
        """
        Parameters
        ----------
        N_level : int
            Number of clustered energy levels to output.
        sigma : float
            Normalized energy ratio for fPCA decomposition.
        """
        super().__init__()

        if N_level <= 0:
            raise ValueError("Parameter[N_level] must be positive.")
        self._N_level = N_level

        if not (0 < sigma <= 1):
            raise ValueError("Parameter[sigma] must lie in (0,1].")
        self._sigma = sigma
        self._filter_negative_eigenvalues = filter_negative_eigenvalues

        # Collected data.
        self._visibilities = []
        self._grams = []

    @chk.check(dict(S=chk.is_instance(vis.VisibilityMatrix), G=chk.is_instance(gr.GramMatrix)))
    def collect(self, S, G):
        """
        Ingest data to internal queue for inference.

        Parameters
        ----------
        S : :py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`
            (N_beam, N_beam) visibility matrix.
        G : :py:class:`~pypeline.phased_array.bluebild.gram.GramMatrix`
            (N_beam, N_beam) gram matrix.
        """
        if not S.is_consistent_with(G, axes=[0, 0]):
            raise ValueError("Parameters[S, G] are inconsistent.")

        self._visibilities.append(S)
        self._grams.append(G)

    def infer_parameters(self, check_hermitian=True):
        """
        Estimate parameters given ingested data.

        Returns
        -------
        N_eig : int
            Number of eigenpairs to use.

        cluster_centroid : :py:class:`~numpy.ndarray`
            (N_level,) intensity field cluster centroids.
        """
        N_data = len(self._visibilities)
        N_beam = N_eig_max = self._visibilities[0].shape[0]

        if self._N_level > N_beam:
            raise ValueError(f"Initialization parameter N_level (set to {self._N_level}) cannot exceed the number of beams ({N_beam}).")

        D_all = np.zeros((N_data, N_eig_max))
        for i, (S, G) in enumerate(zip(self._visibilities, self._grams)):
            # Remove broken BEAM_IDs
            broken_row_id = np.flatnonzero(np.isclose(np.sum(S.data, axis=0), 0))
            working_row_id = list(set(np.arange(N_beam)) - set(broken_row_id))
            idx = np.ix_(working_row_id, working_row_id)
            S, G = S.data[idx], G.data[idx]

            # Functional PCA
            if not np.allclose(S, 0):
                if self._filter_negative_eigenvalues:
                    D, _ = pylinalg_eigh(S, G, tau=self._sigma, check_hermitian=check_hermitian)
                # Copied from Imot_Tools but keeping all eigen pairs (=> no padding/selection required)
                else:
                    try:
                        D = linalg.eigh(S, G, eigvals_only=True)
                    except linalg.LinAlgError:
                        raise ValueError("Parameter[B] is not PSD.")
                D_all[i, : len(D)] = D
            else:
                raise Exception("S, allclose to 0")

        # With fne=0, count all non-zero D
        # Overide N_eig below if fne=1 once D_all = D_all[D_all > 0.0]
        N_eig = max(int(np.ceil(np.count_nonzero(D_all) / N_data)), self._N_level)

        # EO: instead of clustering on non-zero eigenvalues, cluster on strictly
        #    positive eigenvalues to also discard negative eigenvalues if kept in.
        D_all = np.sort(D_all[D_all > 0.0])

        kmeans = skcl.KMeans(n_clusters=self._N_level, random_state=0).fit(np.log(D_all).reshape(-1, 1))

        # For extremely small telescopes or datasets that are mostly 'broken', we can have (N_eig < N_level).
        # In this case we have two options: (N_level = N_eig) or (N_eig = N_level).
        # In the former case, the user's choice of N_level is not respected and subsequent code written by the user
        # could break due to a false assumption. In the latter case, we modify N_eig to match the user's choice.
        # This has the disadvantage of increasing the computational load of Bluebild, but as the N_eig energy levels
        # are clustered together anyway, the trailing energy levels will be (close to) all-0 and can be discarded
        # on inspection.

        # EO: keep cluster centroids for positive part
        cluster_centroid = np.sort(np.exp(kmeans.cluster_centers_)[:, 0])[::-1]

        if self._filter_negative_eigenvalues:
            N_eig = max(int(np.ceil(len(D_all) / N_data)), self._N_level)

        return N_eig, cluster_centroid


class SensitivityFieldParameterEstimator(ParameterEstimator):
    """
    Parameter estimator for computing sensitivity fields.
    """

    @chk.check("sigma", chk.is_real)
    def __init__(self, sigma):
        """
        Parameters
        ----------
        sigma : float
            Normalized energy ratio for fPCA decomposition.
        """
        super().__init__()

        if not (0 < sigma <= 1):
            raise ValueError("Parameter[sigma] must lie in (0,1].")
        self._sigma = sigma
        self._filter_negative_eigenvalues = filter_negative_eigenvalues

        # Collected data.
        self._grams = []

    @chk.check("G", chk.is_instance(gr.GramMatrix))
    def collect(self, G):
        """
        Ingest data to internal queue for inference.

        Parameters
        ----------
        G : :py:class:`~pypeline.phased_array.bluebild.gram.GramMatrix`
            (N_beam, N_beam) gram matrix.
        """
        self._grams.append(G)

    def infer_parameters(self):
        """
        Estimate parameters given ingested data.

        Returns
        -------
        N_eig : int
            Number of eigenpairs to use.
        """
        N_data = len(self._grams)
        N_beam = N_eig_max = self._grams[0].shape[0]

        D_all = np.zeros((N_data, N_eig_max))
        for i, G in enumerate(self._grams):
            # Functional PCA
            if self._filter_negative_eigenvalues:
                D, _ = pylinalg_eigh(G.data, np.eye(N_beam), tau=self._sigma)
            else:
                D = linalg.eigh(G.data, eigvals_only=True)
            D_all[i, : len(D)] = D

        D_all = D_all[D_all.nonzero()]

        if self._filter_negative_eigenvalues:
            N_eig = int(np.ceil(len(D_all) / N_data))
        else:
            N_eig = N_eig_max
        return N_eig
