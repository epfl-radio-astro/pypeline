# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bluebild (NUFFT).
"""

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
import numpy as np
import scipy.constants as constants
import bluebild
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as bb_synth
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import imot_tools.io.plot as implt
import time as tt

import matplotlib
import matplotlib.pyplot as plt

def convert_filter(filters):
    filter_match = dict(lsq=0, std=1, sqrt=2, inv=3)
    filter_enums = []
    for f in filters:
        filter_enums.append(filter_match[f])
    return np.array(filter_enums, dtype=np.uint32)


def make_lmn_grid(grid_size, FoV):
    r"""
    Imaging grid.

    Returns
    -------
    lmn_grid, xyz_grid: Tuple[np.ndarray, np.ndarray]
        (3, grid_size, grid_size) grid coordinates in the local UVW frame and ICRS respectively.
    """
    lim = np.sin(FoV / 2)
    grid_slice = np.linspace(-lim, lim, grid_size)
    l_grid, m_grid = np.meshgrid(grid_slice, grid_slice)
    n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
    lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
    lmn_grid = lmn_grid.reshape(3, -1).astype(np.float32)
    grid_center = lmn_grid.mean(axis=-1)
    lmn_grid -= grid_center[:, None]
    return lmn_grid, grid_center

def convert_uvw_prephase(wl, grid_center, UVW):
    UVW = np.array(UVW, copy=False)
    UVW = (2 * np.pi * UVW.reshape(-1,3).T.reshape(3, -1) / wl).astype(np.float32)
    prephasing = np.exp(1j * np.sum(grid_center[:, None] * UVW, axis=0)).squeeze().astype(np.complex64)
    prephasing = prephasing.reshape(-1)
    return UVW, prephasing




ctx = bluebild.Context(bluebild.ProcessingUnit.AUTO)


# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV, frequency = np.deg2rad(10), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock(ctx)

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=40)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

# Imaging
N_pix = 512
eps = 1e-3
precision = 'single'

t1 = tt.time()
N_level = 3
time_slice = 25


img = None

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(S, G)

N_eig, c_centroid = I_est.infer_parameters()
intervals = bb_dp.centroid_to_intervals(c_centroid)

# Imaging
bluebild_filter = convert_filter(['lsq', 'sqrt'])
for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)

    if img is None:
        lmn, grid_center = make_lmn_grid(N_pix, FoV)
        img = bluebild.PeriodicSynthesis(ctx, W.data.shape[0], W.data.shape[1], intervals.shape[0], bluebild_filter, np.array(lmn[0], dtype=np.float64), np.array(lmn[1], dtype=np.float64), np.array(lmn[2], dtype=np.float64), eps)
    S = vis(XYZ, W, wl)

    uvw, prephase = convert_uvw_prephase(wl, grid_center, UVW_baselines_t)
    img.collect(N_eig, wl, np.array(intervals, dtype=np.float64), np.array(W.data, dtype=np.complex128), np.array(prephase, dtype=np.complex128), np.array(XYZ.data, dtype=np.float64), np.array(uvw[0], dtype=np.float64), np.array(uvw[1], dtype=np.float64), np.array(uvw[2], dtype=np.float64), np.array(S.data, dtype=np.complex128))

lsq_image = img.get(bluebild.Filter.LSQ).reshape((intervals.shape[0],N_pix,N_pix))
sqrt_image = img.get(bluebild.Filter.SQRT).reshape((intervals.shape[0],N_pix,N_pix))

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
img = None
bluebild_filter = convert_filter(['lsq'])
for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    if img is None:
        lmn, grid_center = make_lmn_grid(N_pix, FoV)
        img = bluebild.PeriodicSynthesis(ctx, W.data.shape[0], W.data.shape[1], intervals.shape[0], bluebild_filter, np.array(lmn[0], dtype=np.float64), np.array(lmn[1], dtype=np.float64), np.array(lmn[2], dtype=np.float64), eps)

    uvw, prephase = convert_uvw_prephase(wl, grid_center, UVW_baselines_t)
    img.collect(N_eig, wl, np.array(intervals, dtype=np.float64), np.array(W.data, dtype=np.complex128), np.array(prephase, dtype=np.complex128), np.array(XYZ.data, dtype=np.float64), np.array(uvw[0], dtype=np.float64), np.array(uvw[1], dtype=np.float64), np.array(uvw[2], dtype=np.float64), None)


sensitivity_image = img.get(bluebild.Filter.LSQ).reshape((intervals.shape[0],N_pix,N_pix))

# TODO remove after xyz grid available
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)

I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')

plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild sqrt, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')

plt.figure()
titles = ['Strong sources', 'Mild sources', 'Faint Sources']
for i in range(lsq_image.shape[0]):
    plt.subplot(1, N_level, i + 1)
    ax = plt.gca()
    plt.title(titles[i])
    I_lsq_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5), show_gridlines=False)

plt.suptitle(f'Bluebild Eigenmaps')
plt.show()
#  plt.savefig('final_bb.png')
