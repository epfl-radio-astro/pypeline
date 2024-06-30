# #############################################################################
# statistics.py
# =============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Visibility generation utilities.

Due to the high data-rates emanating from antennas, raw antenna time-series are rarely archived.
Instead, signals from different antennas are correlated together to form *visibility* matrices.
"""

import imot_tools.math.stat as stat
import imot_tools.util.argcheck as chk
import numpy as np

import pypeline.core as core
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.data_gen.source as sky
import pypeline.util.array as array


class VisibilityMatrix(array.LabeledMatrix):
    """
    Visibility coefficients.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import pandas as pd
       from pypeline.phased_array.data_gen.statistics import VisibilityMatrix

    .. doctest::

       >>> N_beam = 5
       >>> beam_idx = pd.Index(range(N_beam), name='BEAM_ID')
       >>> S = VisibilityMatrix(np.eye(N_beam), beam_idx)

       >>> S.data
       array([[1., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0.],
              [0., 0., 1., 0., 0.],
              [0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 1.]])
    """

    @chk.check(
        dict(
            data=chk.accept_any(chk.has_reals, chk.has_complex), beam_idx=beamforming.is_beam_index
        )
    )

    def __init__(self, data, beam_idx, check_hermitian=True, weight_spectrum=None):

        data = np.array(data, copy=False)

        if weight_spectrum is not None:
            weight_spectrum = np.array(weight_spectrum, copy=False)

        N_beam = len(beam_idx)

        if not chk.has_shape((N_beam, N_beam))(data):
            raise ValueError("Parameters[data, beam_idx] are not consistent.")

        if weight_spectrum is not None:
            if not chk.has_shape((N_beam, N_beam))(weight_spectrum):
                raise ValueError("Parameters[weight_spectrum , beam_idx] are not consistent.")

        if check_hermitian:
            if not np.allclose(data, data.conj().T):
                raise ValueError("Parameter[data] must be hermitian symmetric.")

        # Always flag autocorrelation visibilities
        np.fill_diagonal(data, 0)
        if weight_spectrum is not None:
            np.fill_diagonal(weight_spectrum, 0)

        # Normalize and apply spectrum weights if provided
        nz_vis = np.count_nonzero(data)
        if weight_spectrum is not None:
            data *= weight_spectrum / np.sum(weight_spectrum) * nz_vis

        super().__init__(data, beam_idx, beam_idx)


class VisibilityGeneratorBlock(core.Block):
    """
    Generate synthetic visibility matrices.
    """

    @chk.check(
        dict(
            sky_model=chk.is_instance(sky.SkyEmission),
            T=chk.is_real,
            fs=chk.is_integer,
            SNR=chk.is_real,
        )
    )
    def __init__(self, sky_model, T, fs, SNR):
        """
        Parameters
        ----------
        sky_model : :py:class:`~pypeline.phased_array.data_gen.source.SkyEmission`
            Source model from which to generate data.
        T : float
            Integration time [s].
        fs : int
            Sampling rate [samples/s].
        SNR : float
            Signal-to-Noise-Ratio (dB).
        """
        super().__init__()
        self._N_sample = int(T * fs) + 1
        self._SNR = 10 ** (SNR / 10)
        self._sky_model = sky_model

    @chk.check(
        dict(
            XYZ=chk.is_instance(instrument.InstrumentGeometry),
            W=chk.is_instance(beamforming.BeamWeights),
            wl=chk.is_real,
        )
    )
    def __call__(self, XYZ, W, wl):
        """
        Compute visibility matrix.

        Parameters
        ----------
        XYZ : :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            (N_antenna, 3) ICRS instrument geometry.
        W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
            (N_antenna, N_beam) synthesis beamweights.
        wl : float
            Wavelength [m] at which to generate visibilities.

        Returns
        -------
        :py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`
            (N_beam, N_beam) visibility matrix.

        Examples
        --------
        .. testsetup::

           from pypeline.phased_array.instrument import LofarBlock
           from pypeline.phased_array.beamforming import MatchedBeamformerBlock
           import scipy.constants as constants
           import astropy.units as u
           import astropy.time as atime
           import astropy.coordinates as coord
           from pypeline.phased_array.data_gen.statistics import VisibilityGeneratorBlock
           from pypeline.phased_array.data_gen.source import from_tgss_catalog

        .. doctest::

           # Configure instrument and beamformer
           >>> instr = LofarBlock(N_station=24)
           >>> station_id = instr._layout.index.get_level_values('STATION_ID')

           >>> mb_cfg = [(_, _, coord.SkyCoord(0 * u.deg, 90 * u.deg))
           ...           for _ in station_id.drop_duplicates()]
           >>> mb = MatchedBeamformerBlock(mb_cfg)

           # Configure visibility generator
           >>> sky_model = from_tgss_catalog(coord.SkyCoord(0 * u.deg, 90 * u.deg),
           ...                               FoV=np.deg2rad(5),
           ...                               N_src=10)
           >>> S_gen = VisibilityGeneratorBlock(sky_model,
           ...                                  T=8,
           ...                                  fs=196000,
           ...                                  SNR=np.inf)

           # Generate data
           >>> XYZ = instr(atime.Time('J2000'))
           >>> wl = constants.speed_of_light / 145e6
           >>> W = mb(XYZ, wl)
           >>> S = S_gen(XYZ, W, wl)

           # Only 10 sources & no noise, so rank(S) <= 10
           >>> np.linalg.matrix_rank(S.data) <= 10
           True
        """
        if not XYZ.is_consistent_with(W, axes=[0, 0]):
            raise ValueError("Parameters[XYZ, W] are inconsistent.")

        A = np.exp((1j * 2 * np.pi / wl) * (self._sky_model.xyz @ XYZ.data.T))
        S_sky = (W.data.conj().T @ (A.conj().T * self._sky_model.intensity)) @ (A @ W.data)

        noise_var = np.sum(self._sky_model.intensity) / (2 * self._SNR)
        S_noise = W.data.conj().T @ (noise_var * W.data)

        wishart = stat.Wishart(V=S_sky + S_noise, n=self._N_sample)
        S = wishart()[0] / self._N_sample
        return VisibilityMatrix(data=S, beam_idx=W.index[1])
