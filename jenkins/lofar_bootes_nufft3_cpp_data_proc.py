# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bluebild (NUFFT).
"""

import os, sys, argparse
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import bluebild
from imot_tools.io.plot import cmap
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
import time as tt


np.random.seed(0)

ctx = bluebild.Context(bluebild.ProcessingUnit.AUTO)

# Dump data to args.outdir if defined
def dump_data(stats, filename):
    if args.outdir:
        fp = os.path.join(args.outdir, filename + '.npy')
        with open(fp, 'wb') as f:
            np.save(f, stats)
            print("Wrote ", fp)

jkt0_s = tt.time()


# Check arguments
parser = argparse.ArgumentParser()
parser.add_argument("--outdir",   help="Path to dumping location (no dumps if not set)")
args = parser.parse_args()
if args.outdir:
    if not os.path.exists(args.outdir):
        print('fatal: --outdir ('+args.outdir+') must exists if defined.')
        sys.exit(1)
    print("Dumping directory: ", args.outdir)        
else:
    print("Will not dump anything, --outdir not set.")


# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV, frequency = np.deg2rad(8), 145e6
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

# Imaging parameters
N_pix = 512
N_level = 3
precision = 'single'
time_slice = 200
eps = 1e-3
print("\nImaging parameters")
print(f'N_pix {N_pix}\nN_level {N_level}\nprecision {precision}')
print(f'time_slice {time_slice}\neps {eps}\n')

t1 = tt.time()


### Intensity Field ===========================================================
# Parameter Estimation
ifpe_s = tt.time()
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t in time[::time_slice]:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(S, G)

N_eig, c_centroid = I_est.infer_parameters()
ifpe_e = tt.time()
print(f"#@#IFPE {ifpe_e-ifpe_s:.3f} sec")

# Imaging
ifim_s = tt.time()
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision, ctx=ctx)
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))

for t in time[::time_slice]:
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)

    D, V, c_idx = I_dp(S, XYZ, W, wl)

    S_corrected = IV_dp(D, V, W, c_idx)
    nufft_imager.collect(UVW_baselines_t, S_corrected)


# NUFFT Synthesis
lsq_image, sqrt_image = nufft_imager.get_statistic()
ifim_e = tt.time()
print(f"#@#IFIM {ifim_e-ifim_s:.3f} sec")


### Sensitivity Field =========================================================
# Parameter Estimation
sfpe_s = tt.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in time[::time_slice]:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()
sfpe_e = tt.time()
print(f"#@#SFPE {sfpe_e-sfpe_s:.3f} sec")

# Imaging
sfim_s = tt.time()
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision, ctx=ctx)
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig, ctx)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
for t in time[::time_slice]:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)

    D, V = S_dp(XYZ, W, wl)

    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)

sensitivity_image = nufft_imager.get_statistic()[0]

I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
dump_data(I_lsq_eq.data, 'I_lsq_eq_data')
dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid')

I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)

sfim_e = tt.time()
print(f"#@#SFIM {sfim_e-sfim_s:.3f} sec")

t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

jkt0_e = tt.time()
print(f"#@#TOT {jkt0_e-jkt0_s:.3f} sec\n")


### Plotting section
plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')
fp = "I_lsq_nufft3.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild sqrt, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')
fp = "I_sqrt_nufft3.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

plt.figure()
titles = ['Strong sources', 'Mild sources', 'Faint Sources']
for i in range(lsq_image.shape[0]):
    plt.subplot(1, N_level, i + 1)
    ax = plt.gca()
    plt.title(titles[i])
    I_lsq_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5), show_gridlines=False)
plt.suptitle(f'Bluebild Eigenmaps')
fp = "final_bb.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)
