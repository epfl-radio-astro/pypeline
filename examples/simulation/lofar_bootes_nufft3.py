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
import finufft
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as bb_synth
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
matplotlib.use('agg')
import matplotlib.pyplot as plt


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
gram = bb_gr.GramBlock()

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

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))

nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    S_corrected = IV_dp(D, V, W, c_idx)
    nufft_imager.collect(UVW_baselines_t, S_corrected)


# NUFFT Synthesis
lsq_image, sqrt_image = nufft_imager.get_statistic()

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
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)

sensitivity_image = nufft_imager.get_statistic()[0]

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

plt.savefig('final.png')
#  plt.show()
