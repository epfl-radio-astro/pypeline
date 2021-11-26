# #############################################################################
# spatial_domain.py
# =================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Field synthesizers that work in the spatial domain.
"""

import os
import numexpr as ne
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from numpy.ctypeslib import ndpointer
from ctypes import *
import pypeline.phased_array.bluebild.field_synthesizer as synth
import imot_tools.util.argcheck as chk


def _have_matching_shapes(V, XYZ, W):
    if (V.ndim == 2) and (XYZ.ndim == 2) and (W.ndim == 2):
        if V.shape[0] != W.shape[1]:  # N_beam
            return False
        if W.shape[0] != XYZ.shape[0]:  # N_antenna
            return False
        return True

    return False


def print_info(npa, label):
    try:
        print(f'{label:8s} shape={str(npa.shape):18s} dtype={str(npa.dtype):12s} size={npa.nbytes / 1.E9:.3f} GB, type {type(npa)}')
    except:
        print(f'{label:8s} type {type(npa)}')


def c_synthesizer_omp_sp(GRID, V, XYZ, W, wl):

    Nb, Ne     = V.shape
    Na, Nb     = W.shape
    Nc, Nh, Nw = GRID.shape
    GRID = GRID / linalg.norm(GRID, axis=0)
    XYZ  = XYZ - XYZ.mean(axis=0)
    a = 2 * np.pi / wl

    abs_path = os.path.dirname(os.path.abspath(__file__))
    so_file = os.path.join(abs_path, "../../../../", "src/libskabb.so")
    print("so_file = ", so_file)

    custom_functions = CDLL(so_file)
    custom_functions.synthesizer_omp_sp.argtypes=[c_float,                                            # alpha (imag part)
                                                  c_int, c_int, c_int, c_int, c_int, c_int,           # Nb, Ne, Na, Nc, Nh, Nw
                                                  ndpointer(dtype=np.complex64, ndim=2, flags='F'),   # V
                                                  ndpointer(dtype=np.complex64, ndim=2, flags='F'),   # W
                                                  ndpointer(dtype=np.float32,   ndim=2, flags='F'),   # XYZ
                                                  ndpointer(dtype=np.float32,   ndim=3, flags='F'),   # GRID
                                                  ndpointer(dtype=np.float32,   ndim=3, flags='F')]   # I
    V      = np.asfortranarray(V)
    W      = np.asfortranarray(W)
    XYZ    = np.asfortranarray(XYZ)
    GRID   = np.asfortranarray(GRID)

    I      = np.zeros((Ne, Nh, Nw), dtype=np.float32, order='F')

    custom_functions.synthesizer_omp_sp(a, Nb, Ne, Na, Nc, Nh, Nw, V, W, XYZ, GRID, I)

    return I


def c_synthesizer_omp_dp(GRID, V, XYZ, W, wl):

    Nb, Ne     = V.shape
    Na, Nb     = W.shape
    Nc, Nh, Nw = GRID.shape

    GRID = GRID / linalg.norm(GRID, axis=0)
    XYZ  = XYZ - XYZ.mean(axis=0)
    a = 2 * np.pi / wl

    abs_path = os.path.dirname(os.path.abspath(__file__))
    so_file = os.path.join(abs_path, "../../../../", "src/libskabb.so")
    print("so_file = ", so_file)

    custom_functions = CDLL(so_file)
    custom_functions.synthesizer_omp_dp.argtypes=[c_double,                                           # alpha (imag part)
                                                  c_int, c_int, c_int, c_int, c_int, c_int,           # Nb, Ne, Na, Nc, Nh, Nw
                                                  ndpointer(dtype=np.complex128, ndim=2, flags='F'),  # V
                                                  ndpointer(dtype=np.complex128, ndim=2, flags='F'),  # W
                                                  ndpointer(dtype=np.float64,    ndim=2, flags='F'),  # XYZ
                                                  ndpointer(dtype=np.float64,    ndim=3, flags='F'),  # GRID
                                                  ndpointer(dtype=np.float64,    ndim=3, flags='F')]  # I
    V      = np.asfortranarray(V)
    W      = np.asfortranarray(W)
    XYZ    = np.asfortranarray(XYZ)
    GRID   = np.asfortranarray(GRID)
    I      = np.zeros((Ne, Nh, Nw), dtype=np.float64, order='F')

    custom_functions.synthesizer_omp_dp(a, Nb, Ne, Na, Nc, Nh, Nw, V, W, XYZ, GRID, I)

    return I


class SpatialFieldSynthesizerBlock(synth.FieldSynthesizerBlock):
    """
    Field synthesizer based on StandardSynthesis.

    Examples
    --------
    Assume we are imaging a portion of the Bootes field with LOFAR's 24 core stations.

    The short script below shows how to use :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.spatial_domain.SpatialFieldSynthesizerBlock` to form continuous energy level estimates.

    .. testsetup::

       import numpy as np
       import astropy.units as u
       import astropy.time as atime
       import astropy.coordinates as coord
       import scipy.constants as constants
       from tqdm import tqdm as ProgressBar
       from pypeline.phased_array.bluebild.data_processor import IntensityFieldDataProcessorBlock
       from pypeline.phased_array.bluebild.field_synthesizer.spatial_domain import SpatialFieldSynthesizerBlock
       from pypeline.phased_array.instrument import LofarBlock
       from pypeline.phased_array.beamforming import MatchedBeamformerBlock
       from pypeline.phased_array.bluebild.gram import GramBlock
       from pypeline.phased_array.data_gen.source import from_tgss_catalog
       from pypeline.phased_array.data_gen.statistics import VisibilityGeneratorBlock
       from imot_tools.math.sphere.grid import spherical

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
       >>> sky_model=from_tgss_catalog(field_center, field_of_view, N_src=10)
       >>> vis = VisibilityGeneratorBlock(sky_model,
       ...                                T=8,
       ...                                fs=196000,
       ...                                SNR=np.inf)

       ### Energy-level imaging ============================================
       # Pixel grid
       >>> px_grid = spherical(field_center.transform_to('icrs').cartesian.xyz.value,
       ...                     FoV=field_of_view,
       ...                     size=[256, 386]).reshape(3, -1)

       >>> I_dp = IntensityFieldDataProcessorBlock(N_eig=7,  # assumed obtained from IntensityFieldParameterEstimator.infer_parameters()
       ...                                         cluster_centroids=[124.927,  65.09 ,  38.589,  23.256])
       >>> I_fs = SpatialFieldSynthesizerBlock(wl, px_grid)
       >>> t_img = obs_start + np.arange(20) * 400 * u.s  # well-spaced snapshots
       >>> for t in ProgressBar(t_img):
       ...     XYZ = dev(t)
       ...     W = mb(XYZ, wl)
       ...     S = vis(XYZ, W, wl)
       ...     G = gram(XYZ, W, wl)
       ...
       ...     D, V, c_idx = I_dp(S, G)
       ...
       ...     # (N_eig, N_px) energy levels (compact descriptor, not the same thing as [D, V]).
       ...     field_stat = I_fs(V, XYZ.data, W.data)
       ...
       ...     # (N_eig, N_px) energy levels
       ...     # These are the actual field values. Depending on the implementation of FieldSynthesizerBlock, `field_stat` and `field` may differ.
       ...     field = I_fs.synthesize(field_stat)

       # For SpatialFieldSynthesizerBlock(), `field` and `field_stat` are actually identical.
       >>> np.allclose(field_stat, field)
       True

    In the example above, individual snapshots were not added together, hence the final image is just the last field snapshot and can be quite noisy:

    .. doctest::

       from imot_tools.io.s2image import Image
       I_snapshot = Image(data=field, grid=px_grid)

       ax = I_snapshot.draw(index=slice(None),  # Collapse all energy levels
                            catalog=sky_model.xyz.T,
                            data_kwargs=dict(cmap='cubehelix'),
                            catalog_kwargs=dict(s=600))
       ax.get_figure().show()

    .. image:: _img/bluebild_SpatialFieldSynthesizer_snapshot_example.png
    """

    @chk.check(dict(wl=chk.is_real, pix_grid=chk.has_reals, precision=chk.is_integer))
    def __init__(self, wl, pix_grid, precision=64):
        """
        Parameters
        ----------
        wl : float
            Wavelength [m] of observations.
        pix_grid : :py:class:`~numpy.ndarray`
            (3, N_px) pixel vectors.
        precision : int
            Numerical accuracy of floating-point operations.

            Must be 32 or 64.
        """
        super().__init__()
        
        if precision == 32:
            self._fp = np.float32
            self._cp = np.complex64
        elif precision == 64:
            self._fp = np.float64
            self._cp = np.complex128
        else:
            raise ValueError("Parameter[precision] must be 32 or 64.")

        self._precision = precision

        self._wl = wl

        if not ((pix_grid.ndim == 3) and (len(pix_grid) == 3)):
            raise ValueError("Parameter[pix_grid] must have dimensions (3, N_height, N_width).")
        self._grid = pix_grid / linalg.norm(pix_grid, axis=0)

    # needed to remove this check for GPU/CPU flexibility
    # TODO: add back in...
    '''@chk.check(
        dict(
            V=chk.has_complex,
            XYZ=chk.has_reals,
            W=chk.is_instance(np.ndarray, sparse.csr_matrix, sparse.csc_matrix),
        )
    )'''
    
    def __call__(self, V, XYZ, W):
        """
        Compute instantaneous field statistics.
        Parameters
        ----------
        V : :py:class:`~numpy.ndarray`
            (N_beam, N_eig) complex-valued eigenvectors.
        XYZ : :py:class:`~numpy.ndarray`
            (N_antenna, 3) Cartesian instrument geometry.
            `XYZ` must be defined in the same reference frame as `pix_grid` from :py:meth:`~pypeline.phased_array.bluebild.field_synthesizer.spatial_domain.SpatialFieldSynthesizerBlock.__init__`.
        W : :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.csr_matrix` or :py:class:`~scipy.sparse.csc_matrix`
            (N_antenna, N_beam) synthesis beamweights.
        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            (N_eig, N_px) field statistics.
            (Note: StandardSynthesis statistics correspond to the actual field values.)
        """

        # For CPU/GPU agnostic code
        #EO: but not relying on cupy, as it require cuda (not avail on CPU clusters)
        #xp = cp.get_array_module(V)  # not using 'xp' instead of cp or np
        if (type(V) == np.ndarray):
            xp = np
        else:
            import cupy as cp
            if (cp.get_array_module(V) != cp):
                print("Error. V was not recognized correctly as either Cupy or Numpy.")
                sys.exit(1)
            xp = cp
        #print("Using:", xp.__name__)

        if not _have_matching_shapes(V, XYZ, W):
            raise ValueError("Parameters[V, XYZ, W] are inconsistent.")

        # TODO: move precision control outside of the call
        V = V.astype(self._cp, copy=False)
        XYZ = XYZ.astype(self._fp, copy=False)
        W = W.astype(self._cp, copy=False)
        self._grid = self._grid.astype(self._fp, copy=False)
        

        self.mark(self.timer_tag + "Synthesizer call")

        N_antenna, N_beam = W.shape
        N_height, N_width = self._grid.shape[1:]
        N_eig = V.shape[1]

        XYZ = XYZ - XYZ.mean(axis=0)
        #P = xp.zeros((N_antenna, N_height, N_width), dtype=self._cp)
        E = xp.zeros((N_eig, N_height, N_width), dtype=self._cp)

        SKABB_VERBOSE = os.environ.get('SKABB_VERBOSE')

        if SKABB_VERBOSE == "1":
            print_info(self._grid, '._grid')
            print_info(V, 'V')
            print_info(XYZ, 'XYZ')            

        a = 1.0j * 2.0 * np.pi / self._wl

        #EO: check whether C SS was requested
        SKABB_C_SYNTH = os.environ.get('SKABB_C_SYNTH')
        #print("SKABB_C_SYNTH =", SKABB_C_SYNTH)

        self.mark(self.timer_tag + "Synthesizer matmuls")

        # Numpy array (i.e. CPU) + SKABB_C_SYNTH=="1"
        if xp == np and SKABB_C_SYNTH == "1":
            print("@@@ C Standard Synthesizer in action @@@")
            if isinstance(W, sparse.csr.csr_matrix) or isinstance(W, sparse.csc.csc_matrix):
                Wd = W.toarray(order='F')
                if SKABB_VERBOSE == "1":
                    print_info(Wd, 'Wd')
            if self._precision == 32:
                I = c_synthesizer_omp_sp(self._grid, V, XYZ, Wd, self._wl)
            elif self._precision == 64:
                I = c_synthesizer_omp_dp(self._grid, V, XYZ, Wd, self._wl)
            else:
                print(f"Wrong precision {self._precision:d}")
                sys.exit(1)
        else:
            for i in range(N_width):
                pix_gpu = xp.asarray(self._grid[:,:,i])
                b  = xp.matmul(XYZ, pix_gpu)
                P  = xp.exp(a*b)
                if xp == np and (isinstance(W, sparse.csr.csr_matrix) or isinstance(W, sparse.csc.csc_matrix)):
                    PW = W.T @ P
                else:
                    PW = xp.matmul(W.T, P)
                E[:,:,i]  = xp.matmul(V.T, PW)
            I = E.real ** 2 + E.imag ** 2

        self.unmark(self.timer_tag + "Synthesizer matmuls")

        self.unmark(self.timer_tag + "Synthesizer call")

        if SKABB_VERBOSE == "1":
            print_info(I, 'I')
            
        #EO Always return np array
        if xp != np:
            I = I.get()

        return I

    @chk.check("stat", chk.has_reals)
    def synthesize(self, stat):
        """
        Compute field values from statistics.

        Parameters
        ----------
        stat : :py:class:`~numpy.ndarray`
            (N_level, N_px) field statistics.

        Returns
        -------
        field : :py:class:`~numpy.ndarray`
            (N_level, N_px) field values.
        """
        stat = np.array(stat, copy=False)

        if stat.ndim != 2:
            raise ValueError("Parameter[stat] is incorrectly shaped.")

        N_level = len(stat)
        N_px = self._grid.shape[1]

        if not chk.has_shape([N_level, N_px])(stat):
            raise ValueError("Parameter[stat] does not match the grid's dimensions.")

        field = stat
        return field
