# #############################################################################
# fourier_domain.py
# =================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Field synthesizers that work in Fourier Series domain.
"""
import sys
import imot_tools.math.func as func
import imot_tools.math.linalg as pylinalg
import imot_tools.math.sphere.transform as transform
import imot_tools.util.argcheck as chk
import numexpr as ne
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cpfft
import pyffs
#import scipy.fftpack as fftpack # considered legacy
import scipy.fft as fft
import scipy.linalg as linalg
import scipy.sparse as sparse
import nvtx

import pypeline.phased_array.bluebild.field_synthesizer as synth
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as fsd
import pypeline.util.frame as frame
import typing as typ
import warnings
import finufft
import astropy.coordinates as aspy


def print_info(x, name):
    try:
        print(f"info on {name:12s}: {type(x)}, {x.dtype}, {x.shape}") #mean = {np.mean(x):.12f}")
    except:
        print(f"info on {name:12s}: {type(x)}")

def are_close(a, b, label):
    try:
        print(f"are all_close {label}?", np.allclose(a, b, atol=1e-12))
    except:
        print(f"are all_close {label}? WENT WRONG - Check input")

def check_allclose(a_, b_, label):
    a = a_
    b = b_
    if cp.get_array_module(a) == cp:
        a = a_.get()
    if cp.get_array_module(b) == cp:
        b = b_.get()
    mean_diff = np.mean(np.abs(a - b))
    print(f"INFO: {label} allclose? mean abs diff = {mean_diff:.12f}")

    if not np.allclose(a, b, atol=1e-12):
        mean_diff = np.mean(np.abs(a - b))
        print(f"FATAL: {label} not allclose! mean abs diff = {mean_diff}")
        sys.exit(1)

class FourierFieldSynthesizerBlock(synth.FieldSynthesizerBlock):
    """
    Field synthesizer based on PeriodicSynthesis.

    Examples
    --------
    Assume we are imaging a portion of the Bootes field with LOFAR's 24 core stations.

    The short script below shows how to use :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.fourier_domain.FourierFieldSynthesizerBlock` to form continuous energy level estimates.

    .. testsetup::

       import numpy as np
       import astropy.units as u
       import astropy.time as atime
       import astropy.coordinates as coord
       import scipy.constants as constants
       from tqdm import tqdm as ProgressBar
       from pypeline.phased_array.bluebild.data_processor import IntensityFieldDataProcessorBlock
       from pypeline.phased_array.bluebild.field_synthesizer.fourier_domain import FourierFieldSynthesizerBlock
       from pypeline.phased_array.instrument import LofarBlock
       from pypeline.phased_array.beamforming import MatchedBeamformerBlock
       from pypeline.phased_array.bluebild.gram import GramBlock
       from pypeline.phased_array.data_gen.source import from_tgss_catalog
       from pypeline.phased_array.data_gen.statistics import VisibilityGeneratorBlock
       from imot_tools.math.sphere.grid import equal_angle
       from imot_tools.math.sphere.transform import pol2cart

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
       # Kernel parameters
       >>> t_img = obs_start + np.arange(200) * 8 * u.s  # fine-grained snapshots
       >>> obs_start, obs_end = t_img[0], t_img[-1]
       >>> R = dev.icrs2bfsf_rot(obs_start, obs_end)
       >>> N_FS = dev.bfsf_kernel_bandwidth(wl, obs_start, obs_end)
       >>> T_kernel = np.deg2rad(10)

       # Pixel grid: make sure to generate it in BFSF coordinates by applying R.
       >>> _, _, px_colat, px_lon = equal_angle(N=dev.nyquist_rate(wl),
       ...                                      direction=np.dot(R, field_center.transform_to('icrs').cartesian.xyz.value),
       ...                                      FoV=field_of_view)

       >>> I_dp = IntensityFieldDataProcessorBlock(N_eig=7,  # assumed obtained from IntensityFieldParameterEstimator.infer_parameters()
       ...                                         cluster_centroids=[124.927,  65.09 ,  38.589,  23.256])
       >>> I_fs = FourierFieldSynthesizerBlock(wl, px_colat, px_lon, N_FS, T_kernel, R)
       >>> for t in ProgressBar(t_img):
       ...     XYZ = dev(t)
       ...     W = mb(XYZ, wl)
       ...     S = vis(XYZ, W, wl)
       ...     G = gram(XYZ, W, wl)
       ...
       ...     D, V, c_idx = I_dp(S, G)
       ...
       ...     # (N_eig, N_height, N_FS+Q) energy levels (compact descriptor, not the same thing as [D, V]).
       ...     field_stat = I_fs(V, XYZ.data, W.data)

       # (N_eig, N_height, N_width) energy levels
       # These are the actual field values. Depending on the implementation of FieldSynthesizerBlock, `field_stat` and `field` may differ.
       >>> field = I_fs.synthesize(field_stat)

    In the example above, individual snapshots were not added together, hence the final image is just the last field snapshot and can be quite noisy:

    .. doctest::

       from imot_tools.io.s2image import Image
       # Transform grid to ICRS coordinates before plotting.
       px_grid = np.tensordot(R.T, pol2cart(1, px_colat, px_lon), axes=1)
       I_snapshot = Image(data=field, grid=px_grid)

       ax = I_snapshot.draw(index=slice(None),  # Collapse all energy levels
                            catalog=sky_model.xyz.T,
                            data_kwargs=dict(cmap='cubehelix'),
                            catalog_kwargs=dict(s=600))
       ax.get_figure().show()

    .. image:: _img/bluebild_FourierFieldSynthesizer_snapshot_example.png
    """

    @chk.check(
        dict(
            wl=chk.is_real,
            grid_colat=chk.has_reals,
            grid_lon=chk.has_reals,
            N_FS=chk.is_odd,
            T=chk.is_real,
            R=chk.require_all(chk.has_shape([3, 3]), chk.has_reals),
            precision=chk.is_integer,
        )
    )
    def __init__(self, wl, grid_colat, grid_lon, N_FS, T, R, precision=64):
        r"""
        Parameters
        ----------
        wl : float
            Wavelength [m] of observations.
        grid_colat : :py:class:`~numpy.ndarray`
            (N_height, 1) BFSF polar angles [rad].
        grid_lon : :py:class:`~numpy.ndarray`
            (1, N_width) equi-spaced BFSF azimuthal angles [rad].
        N_FS : int
            :math:`2\pi`-periodic kernel bandwidth. (odd-valued)
        T : float
            Kernel periodicity [rad] to use for imaging.
        R : array-like(float)
            (3, 3) ICRS -> BFSF rotation matrix.
        precision : int
            Numerical accuracy of floating-point operations.

            Must be 32 or 64.

        Notes
        -----
        * `grid_colat` and `grid_lon` should be generated using :py:func:`~imot_tools.math.sphere.grid.equal_angle`.
        * `N_FS` can be optimally chosen by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.bfsf_kernel_bandwidth`.
        * `R` can be obtained by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.icrs2bfsf_rot`.
        """
        super().__init__()

        print("???????? precision = ", precision)

        if precision == 32:
            self._fp = np.float32
            self._cp = np.complex64
        elif precision == 64:
            self._fp = np.float64
            self._cp = np.complex128
        else:
            raise ValueError("Parameter[precision] must be 32 or 64.")

        self._wl = wl

        if N_FS <= 0:
            raise ValueError("Parameter[N_FS] must be positive.")

        if not (0 < T <= 2 * np.pi):
            raise ValueError(f"Parameter[T] is out of bounds.")

        if not np.isclose(T, 2 * np.pi):  # PeriodicSynthesis
            self._alpha_window = 0.1
            T_min = (1 + self._alpha_window) * grid_lon.ptp()
            if T < T_min:
                raise ValueError(f"Parameter[T] must be greater that {T_min}.")
            self._T = T

            aw = self._alpha_window
            lon_start, lon_end = grid_lon[0, [0, -1]]
            T_start, T_end = lon_end + T * np.r_[0.5 * aw - 1, 0.5 * aw]
            self._Tc = (T_start + T_end) / 2
            self._mps = lon_start - (T_start + 0.5 * T * aw)  # max_phase_shift
            print(f"_mps = {self._mps:.12f}")

            N_FS_trunc = N_FS / (2 * np.pi) * T
            N_FS_trunc = int(np.ceil(N_FS_trunc))
            N_FS_trunc += 1 if chk.is_even(N_FS_trunc) else 0
            self._NFS = N_FS_trunc
            print(f"_NFS = {self._NFS:.12f}")

        else:  # No PeriodicSynthesis, but set params to still work.
            self._alpha_window = 0
            self._T = 2 * np.pi
            self._Tc = np.pi
            self._mps = 2 * np.pi  # max_phase_shift
            self._NFS = N_FS

        self._grid_colat = grid_colat
        self._grid_lon = grid_lon
        self._R = np.array(R)

        # Buffered state
        self._FSk = None  # (N_antenna, N_height, N_FS+Q) FS coefficients
        self._XYZk = None  # (N_antenna, 3) BFSF coordinates

    #EO: I disable this for now as cupy objects fail the check
    """
    @chk.check(
        dict(
            V=chk.has_complex,
            XYZ=chk.has_reals,
            W=chk.is_instance(np.ndarray, sparse.csr_matrix, sparse.csc_matrix),
        )
    )
    """
    def __call__(self, V, XYZ, W):
        """
        Compute instantaneous field statistics.

        Parameters
        ----------
        V : :py:class:`~numpy.ndarray`
            (N_beam, N_eig) complex-valued eigenvectors.
        XYZ : :py:class:`~numpy.ndarray`
            (N_antenna, 3) Cartesian instrument geometry.

            `XYZ` must be given in ICRS.
        W : :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.csr_matrix` or :py:class:`~scipy.sparse.csc_matrix`
            (N_antenna, N_beam) synthesis beamweights.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            (N_eig, N_height, N_FS + Q) field statistics.
        """

        # for CPU/GPU agnostic code
        with nvtx.annotate(message="f_d/(cu|num)py", color="lime"):
            xp = cp.get_array_module(V)
        print("Using:", xp.__name__)

        self.mark(self.timer_tag + "Synthesizer call")
        if not fsd._have_matching_shapes(V, XYZ, W):
            raise ValueError("Parameters[V, XYZ, W] are inconsistent.")

        print("checking types: target =", self._cp)

        self.mark(self.timer_tag + "Synthesizer: astype casts")

        # TODO: move precision control outside of the call
        #V = V.astype(self._cp, copy=False)
        #print("#V.dtype =", V.dtype, self._cp)
        #XYZ = XYZ.astype(self._fp, copy=False)
        #W = W.astype(self._cp, copy=False)
        
        # need to convert array type to run on gpu
        #if xp == np and (isinstance(W, sparse.csr.csr_matrix) or isinstance(W, sparse.csc.csc_matrix)):
        #    W = W.toarray()

        self.unmark(self.timer_tag + "Synthesizer: astype casts")

        print_info(V, 'V')
        print_info(XYZ, 'XYZ')
        print_info(self._R, 'self._R')

        self.mark(self.timer_tag + "Synthesizer: matmul 1")
        with nvtx.annotate(message="matmul11", color="grey"):
            if xp == np:
                bfsf_XYZ = XYZ @ self._R.T
            else:
                bfsf_XYZ = xp.matmul(XYZ, cp.asarray(self._R.T))
        self.unmark(self.timer_tag + "Synthesizer: matmul 1")
        print_info(bfsf_XYZ, 'bfsf_XYZ')

        self.mark(self.timer_tag + "Synthesizer: calc phase shift")
        with nvtx.annotate(message="synth: phase_shift", color="green"):
            if self._XYZk is None:
                phase_shift = np.inf
            else:
                phase_shift = self._phase_shift(bfsf_XYZ)
        self.unmark(self.timer_tag + "Synthesizer: calc phase shift")
                
        if self._regen_required(phase_shift):
            self.mark(self.timer_tag + "Synthesizer: regenerate kernel")
            with nvtx.annotate(message="synth: regen_kernel", color="yellow"):
                self._regen_kernel(bfsf_XYZ)
                phase_shift = 0
            self.unmark(self.timer_tag + "Synthesizer: regenerate kernel")

        N_antenna, N_height, _2N1Q = self._FSk.shape
        N = (self._NFS - 1) // 2
        Q = _2N1Q - self._NFS
        N_beam = W.shape[1]

        mod_phase = -1j * 2 * np.pi * phase_shift / self._T

        s = np.r_[-N : N + 1, np.zeros(Q)]

        print_info(V, 'V')
        print_info(W, 'W')
        print_info(mod_phase, 'mod_phase')
        print_info(s, 's')

        if xp == np: ### CPU

            FSk   = self._FSk.reshape(N_antenna, N_height * _2N1Q)

            self.mark(self.timer_tag + "Synthesizer: CPU matmuls 2 & 3")
            with nvtx.annotate(message="synth: CPU matmuls 2 & 3", color="chocolate"):
                PW_FS = W.T @ FSk
                E_FS  = V.T @ PW_FS
                E_FS  = E_FS.reshape(E_FS.shape[0], N_height, _2N1Q)
            self.unmark(self.timer_tag + "Synthesizer: CPU matmuls 2 & 3")

            self.mark(self.timer_tag + "Synthesizer: apply phase shift")
            with nvtx.annotate(message="synth: apply phase shift", color="lavender"):
                mod_phase = mod_phase * s
                E_FS *= np.exp(mod_phase)
            self.unmark(self.timer_tag + "Synthesizer: apply phase shift")

            self.mark(self.timer_tag + "Synthesizer: IFFS")
            with nvtx.annotate(message="synth: IFFS", color="red"):
                E_Ny = pyffs.iffs(E_FS, self._T, self._Tc, self._NFS, axis=2)
                print_info(E_Ny, 'E_Ny')
                I_Ny = E_Ny.real ** 2 + E_Ny.imag ** 2
                print_info(I_Ny, 'I_Ny')
            self.unmark(self.timer_tag + "Synthesizer: IFFS")

            return I_Ny

        else: ### GPU

            self.mark(self.timer_tag + "Synthesizer: GPU array allocation")
            with nvtx.annotate(message="synth: GPU alloc", color="pink"):
                FSk = cp.asarray(self._FSk.reshape(N_antenna, N_height * _2N1Q))
            self.unmark(self.timer_tag + "Synthesizer: GPU array allocation")

            self.mark(self.timer_tag + "Synthesizer: GPU matmuls 2 & 3")
            with nvtx.annotate(message="synth: CPU matmuls 2 & 3", color="chocolate"):
                PW_FS = cp.matmul(W.T, FSk)
                E_FS  = cp.matmul(V.T, PW_FS)
                E_FS  = E_FS.reshape(E_FS.shape[0], N_height, _2N1Q)
            self.unmark(self.timer_tag + "Synthesizer: GPU matmuls 2 & 3")

            self.mark(self.timer_tag + "Synthesizer: apply phase shift")
            with nvtx.annotate(message="synth: apply phase shift", color="lavender"):
                mod_phase = cp.asarray(mod_phase * s)
                #print_info(mod_phase, 'mod_phase')
                E_FS *= cp.exp(mod_phase)
            self.unmark(self.timer_tag + "Synthesizer: apply phase shift")
            
            self.mark(self.timer_tag + "Synthesizer: IFFS")
            with nvtx.annotate(message="synth: IFFS", color="red"):
                E_Ny = pyffs.iffs(E_FS, self._T, self._Tc, self._NFS, axis=2) # TODO: send on GPU
                print_info(E_Ny, 'E_Ny')
                I_Ny = E_Ny.real ** 2 + E_Ny.imag ** 2
                print_info(I_Ny, 'I_Ny')
                I_Ny = I_Ny.get()
            self.unmark(self.timer_tag + "Synthesizer: IFFS")

            self.unmark(self.timer_tag + "Synthesizer call")

            return I_Ny

    @nvtx.annotate(message="f_d/synthesize", color="blue")
    @chk.check("stat", chk.has_reals)
    def synthesize(self, stat):
        """
        Compute field values from statistics.

        Parameters
        ----------
        stat : :py:class:`~numpy.ndarray`
            (N_level, N_height, N_FS + Q) field statistics.

        Returns
        -------
        field : :py:class:`~numpy.ndarray`
            (N_level, N_height, N_width) field values.
        """
        stat = np.array(stat, copy=False)

        if stat.ndim != 3:
            raise ValueError("Parameter[stat] is incorrectly shaped.")

        N_level = len(stat)
        N_height, _2N1Q = self._FSk.shape[1:]

        if not chk.has_shape([N_level, N_height, _2N1Q])(stat):
            raise ValueError("Parameter[stat] does not match the kernel's dimensions.")

        field_FS = pyffs.ffs(stat, self._T, self._Tc, self._NFS, axis=2)
        field = pyffs.fs_interp(
            field_FS[:, :, : self._NFS],
            T=self._T,
            a=self._grid_lon[0, 0],
            b=self._grid_lon[0, -1],
            M=self._grid_lon.size,
            axis=2,
            real_x=True,
        )
        return field

    def _phase_shift(self, XYZ):
        """
        Angular shift w.r.t kernel antenna coordinates.

        Parameters
        ----------
        XYZ : :py:class:`~numpy.ndarray`
            (N_antenna, 3) Cartesian instrument geometry.

            `XYZ` must be given in BFSF.

        Returns
        -------
        theta : float
            Angular shift (radians) such that ``dot(_XYZk, R(theta).T) == XYZ``.
        """
        print_info(self._XYZk, 'self._XYZk @@')
        print_info(XYZ,  'XYZ @@')
        if cp.get_array_module(XYZ) == cp:
            R_T, *_ = linalg.lstsq(self._XYZk[:, :2].get(), XYZ[:, :2].get())
        else:
            R_T, *_ = linalg.lstsq(self._XYZk[:, :2], XYZ[:, :2])

        R = np.eye(3)
        R[:2, :2] = R_T.T
        theta = pylinalg.z_rot2angle(R)
        return theta

    def _regen_required(self, shift):
        lhs = np.deg2rad(-0.1)  # Slightly below 0 due to numerical rounding
        if lhs <= shift <= self._mps:
            return False
        else:
            return True

    def _regen_kernel(self, XYZ):
        """
        Compute kernel.

        Parameters
        ----------
        XYZ : :py:class:`~numpy.ndarray`
            (N_antenna, 3) Cartesian instrument geometry.

            `XYZ` must be given in BFSF.
        """

        if cp.get_array_module(XYZ) == cp:
            N_samples = cpfft.next_fast_len(self._NFS)
            lon_smpl = pyffs.ffs_sample(self._T, self._NFS, self._Tc, N_samples, mod=cp)[0].get() #EO
        else:
            N_samples = fft.next_fast_len(self._NFS)
            lon_smpl = pyffs.ffs_sample(self._T, self._NFS, self._Tc, N_samples, mod=np)[0]

        print(f"N_samples = {N_samples}")
        print_info(self._T, 'self._T')
        print_info(self._Tc, 'self._Tc')
        print_info(self._NFS, 'self._NFS')
        print_info(lon_smpl, 'lon_smpl')

        pix_smpl = transform.pol2cart(1, self._grid_colat, lon_smpl.reshape(1, -1))
        print_info(pix_smpl, 'pix_smpl')

        N_antenna = len(XYZ)
        N_height  = len(self._grid_colat)
        print(f"N_antenna = {N_antenna}, N_height = {N_height}")

        # TODO: fix memory problems with DASK array => but can DASK work with gpu?
        #       or break up by blocks of antenna stations?
        
        # `self._NFS` assumes imaging is performed with `XYZ` centered at the origin.
        print_info(XYZ, 'XYZ')
        XYZ_c = XYZ - XYZ.mean(axis=0)
        """
        if cp.get_array_module(XYZ) == cp:
            XYZ_c = XYZ.get() - XYZ.get().mean(axis=0)
        else:
            XYZ_c = XYZ - XYZ.mean(axis=0)
        print_info(XYZ_c, 'XYZ_c')
        #return XYZ_c
        """

        window = func.Tukey(self._T, self._Tc, self._alpha_window)
        
        a = 1j * 2 * np.pi / self._wl

        w = window(lon_smpl)
        print_info(w, 'w')

        if cp.get_array_module(XYZ) == cp:
            b = cp.tensordot(cp.asarray(XYZ_c), cp.asarray(pix_smpl), axes=1)
            print_info(b, 'b')
            print_info(a, 'a')
            k_smpl = cp.dot(cp.asarray(a), b)
            print_info(k_smpl, 'k_smpl ab')
            k_smpl = cp.exp(k_smpl)
            print_info(k_smpl, 'k_smpl exp(ab)')

            #k_smpl = cp.matmul(k_smpl, cp.asarray(window(lon_smpl)), axes=2,)
            #print_info(k_smpl, 'k_smpl')
            
            for i in range(len(lon_smpl)):
                k_smpl[:,:,i] *= w[i] 

            self._FSk = pyffs.ffs(k_smpl.get(), self._T, self._Tc, self._NFS, axis=2) #TODO:  convert to run on GPU
 
        else:
            k_smpl = np.zeros((N_antenna, N_height, N_samples), dtype=self._cp) # allocate on gpu
            ne.evaluate(
                "exp(A * B) * C",
                dict(
                    A=a,
                    B=np.tensordot(XYZ_c, pix_smpl, axes=1),
                    C=window(lon_smpl),
                ),
                out=k_smpl,
                casting="same_kind",
            )  # Due to limitations of NumExpr2
            self._FSk = pyffs.ffs(k_smpl, self._T, self._Tc, self._NFS, axis=2) #TODO:  convert to run on GPU
                
        print_info(self._FSk, 'self._FSk')

        self._XYZk = XYZ


class NUFFTFieldSynthesizerBlock(synth.FieldSynthesizerBlock):
    r"""
    Field synthesizer based on the NUFFT.
    """
    _precision_mappings = dict(single=dict(complex=np.complex64, real=np.float32, dtype='float32'),
                               double=dict(complex=np.complex128, real=np.float64, dtype='float64'))

    def __init__(self, wl: float, UVW: np.ndarray, grid_size: int, FoV: float, field_center: aspy.SkyCoord,
                 eps: float = 1e-6, w_term: bool = True, n_trans: int = 1, precision: str = 'double'):
        r"""

        Parameters
        ----------
        wl: float
            Observation wavelength.
        UVW: np.ndarray
            (3, N_uvw) UVW coordinates expressed in the local UVW frame.
        grid_size: int
            Size of the output imaging grid across each dimension.
        FoV: float
            Size of the FoV in radians.
        field_center: astropy.coordinates.SkyCoord
            Center of the field of view for defining the local UVW frame.
        eps: float
            Relative tolerance of the NUFFT.
        w_term: bool
            Neglects the ``w_term`` (do not use for large FoV!).
        n_trans: int
            Number of simultaneous NUFFT transforms.
        precision: str
            Whether to use ``'single'`` or ``'double'`` precision.
        """
        self._precision = precision
        UVW = np.array(UVW, copy=False)
        self._UVW = (2 * np.pi * UVW.reshape(3, -1) / wl).astype(self._precision_mappings[self._precision]['real'])
        self._grid_size = grid_size
        self._FoV = FoV
        self._field_center = field_center
        self.lmn_grid, self.xyz_grid = self._make_grids()
        self._lmn_grid = self.lmn_grid.reshape(3, -1).astype(self._precision_mappings[self._precision]['real'])
        self._n_trans = n_trans
        if w_term:
            grid_center = self._lmn_grid.mean(axis=-1)
            self._lmn_grid -= grid_center[:, None]
            self._prephasing = np.exp(1j * np.sum(grid_center[:, None] * self._UVW, axis=0)).squeeze().astype(
                self._precision_mappings[self._precision]['complex'])
            self._plan = finufft.Plan(nufft_type=3, n_modes_or_dim=3, eps=eps, isign=1, n_trans=n_trans,
                                      dtype=self._precision_mappings[self._precision]['dtype'])
            self._plan.setpts(x=self._UVW[0], y=self._UVW[1], z=self._UVW[-1],
                              s=self._lmn_grid[0], t=self._lmn_grid[1], u=self._lmn_grid[-1])
            self._inner_fft_sizes = np.floor(4 * np.linalg.norm(self._lmn_grid, ord=np.infty, axis=-1) * \
                                             np.linalg.norm(self._UVW, ord=np.infty, axis=-1) / np.pi +
                                             np.log(1 / eps) + 1)
        else:
            warnings.warn('Setting the parameter w_term to False can result in a loss of accuracy for large FoVs!',
                          UserWarning)
            scaling = (2 * np.sin(self._FoV / 2) / self._grid_size).astype(
                self._precision_mappings[self._precision]['real'])
            self._prephasing = np.exp(1j * self._UVW[-1]).squeeze().astype(
                self._precision_mappings[self._precision]['complex'])
            self._plan = finufft.Plan(nufft_type=1, n_modes_or_dim=(self._grid_size, self._grid_size),
                                      eps=eps, isign=1, n_trans=n_trans,
                                      dtype=self._precision_mappings[self._precision]['dtype'])
            self._plan.setpts(x=scaling * self._UVW[1], y=scaling * self._UVW[0])
        super(NUFFTFieldSynthesizerBlock, self).__init__()

    def _make_grids(self) -> typ.Tuple[np.ndarray, np.ndarray]:
        r"""
        Imaging grid.

        Returns
        -------
        lmn_grid, xyz_grid: Tuple[np.ndarray, np.ndarray]
            (3, grid_size, grid_size) grid coordinates in the local UVW frame and ICRS respectively.
        """
        lim = np.sin(self._FoV / 2)
        grid_slice = np.linspace(-lim, lim, self._grid_size)
        l_grid, m_grid = np.meshgrid(grid_slice, grid_slice)
        n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
        lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
        uvw_frame = frame.uvw_basis(self._field_center)
        xyz_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
        return lmn_grid, xyz_grid

    def __call__(self, V: np.ndarray) -> np.ndarray:
        r"""
        Synthesize a set of virtual visibilities.

        Parameters
        ----------
        V: np.ndarray
            (M, N_uvw) stack of virtual visibilities to synthesize. If the ``n_trans`` parameter of the NUFFT plan is
             different from zero, then one must have ``M==n_trans``. In which case, the ``M`` NUFFTs are computed in parallel
             using OpenMP multi-threading. Otherwise, the ``M`` NUFFTs are computed sequentially.
        Returns
        -------
        field: np.ndarray
            (M, N_pix) field values.
        """
        V = np.array(V, copy=False).squeeze().astype(self._precision_mappings[self._precision]['complex'])
        if V.ndim > 1:
            V = V.reshape(-1, self._UVW.shape[-1])
            self._prephasing = self._prephasing[None, :]
            V *= self._prephasing
            if self._n_trans == 1:  # NUFFT are evaluated sequentially
                out = []
                for n in range(V.shape[0]):
                    out.append(np.real(self._plan.execute(V[n])))
                out = np.stack(out, axis=0)
            else:
                out = np.real(self._plan.execute(
                    V))  # NUFFT are evaluated in parallel (not clear if multi-threaded or multi-processed?)
        else:
            out = np.real(self._plan.execute(V * self._prephasing))
        return out

    def synthesize(self, V: np.ndarray) -> np.ndarray:
        r"""
        Alias for :py:meth:`~pypeline.phased_array.field_synthesizer.fourier_domain.NUFFTFieldSynthesizerBlock.__call__`.
        """
        return self.__call__(V)
