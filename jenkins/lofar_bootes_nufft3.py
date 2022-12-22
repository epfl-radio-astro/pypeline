# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bluebild (NUFFT).
"""

import os
import sys
import time
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import finufft
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
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
from pypeline.util import frame
import bipptb


args = bipptb.check_args(sys.argv)

# For reproducible results
np.random.seed(0)

jkt0_s = time.time()

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV_deg = 8.0
FoV, frequency = np.deg2rad(FoV_deg), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()

# Data generation
T_integration = 8
N_src     = 40
fs        = 196000
SNR       = 30
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=N_src)
vis       = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=fs, SNR=SNR)
times     = obs_start + (T_integration * u.s) * np.arange(3595)
N_antenna = dev(times[0]).data.shape[0]

# Imaging parameters
N_pix      = 512
N_levels   = 3
precision  = args.precision
eps        = 1e-3
time_slice = 200
times = times[::time_slice]


print(f"-I- precision = {precision}")
print(f"-I- N_station =", N_station)
print(f"-I- N_antenna = {N_antenna:d}")
print(f"-I- T_integration =", T_integration)
print(f"-I- Field center  =", field_center)
print(f"-I- Field of view =", FoV_deg, "deg")
print(f"-I- frequency =", frequency)
print(f"-I- SNR =", SNR)
print(f"-I- fs =", fs)
print(f"-I- N_pix =", N_pix)
print(f"-I- N_levels =", N_levels)
print(f"-I- eps =", eps)
print(f"-I- OMP_NUM_THREADS =", os.getenv('OMP_NUM_THREADS'))


### Intensity Field ===========================================================
# Parameter Estimation
ifpe_s = time.time()
I_est = bb_pe.IntensityFieldParameterEstimator(N_levels, sigma=0.95)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(S, G)
N_eig, intervals = I_est.infer_parameters()
ifpe_e = time.time()
print(f"#@#IFPE {ifpe_e-ifpe_s:.3f} sec")

# Imaging
ifim_s = time.time()
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)

for t in times:
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    D, V = I_dp(S, XYZ, W, wl)
    S_corrected = IV_dp(D, V, W, intervals)
    nufft_imager.collect(UVW_baselines_t, S_corrected)

# NUFFT Synthesis
lsq_image, sqrt_image = nufft_imager.get_statistic()
ifim_e = time.time()
print(f"#@#IFIM {ifim_e-ifim_s:.3f} sec")


### Sensitivity Field =========================================================
# Parameter Estimation
sfpe_s = time.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
sfpe_e = time.time()
print(f"#@#SFPE {sfpe_e-sfpe_s:.3f} sec")

# Imaging
sfim_s = time.time()
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
for t in times:
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)
    S_sensitivity = SV_dp(D, V, W)
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)

sensitivity_image = nufft_imager.get_statistic()[0]

I_lsq_eq  = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)

sfim_e = time.time()
print(f"#@#SFIM {sfim_e-sfim_s:.3f} sec")

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e-jkt0_s:.3f} sec\n")

#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)

print(I_lsq_eq.data)

bipptb.dump_data(I_lsq_eq.data, 'I_lsq_eq_data', args.outdir)
bipptb.dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid', args.outdir)


### Plot results
plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {FoV_deg} degrees.')
fp = "I_lsq_nufft3.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild sqrt, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {FoV_deg} degrees.')
fp = "I_sqrt_nufft3.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

plt.figure()
titles = ['Strong sources', 'Mild sources', 'Faint Sources']
for i in range(lsq_image.shape[0]):
    plt.subplot(1, N_levels, i + 1)
    ax = plt.gca()
    plt.title(titles[i])
    I_lsq_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5), show_gridlines=False)
plt.suptitle(f'Bluebild Eigenmaps')
fp = "final_bb.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)
