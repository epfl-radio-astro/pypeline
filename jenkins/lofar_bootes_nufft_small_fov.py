# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bluebild (NUFFT).
"""

import bluebild_tools.cupy_util as bbt_cupy
use_cupy = bbt_cupy.is_cupy_usable()

import os, sys, argparse
#from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import scipy.sparse as sparse
import finufft
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
from   pypeline.phased_array.bluebild.imager import fourier_domain as bb_im
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from imot_tools.math.func import SphericalDirichlet
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import time as tt



np.random.seed(0)


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
gram = bb_gr.GramBlock()

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=40)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

# Field center coordinates
field_center_lon, field_center_lat = field_center.data.lon.rad, field_center.data.lat.rad
field_center_xyz = field_center.cartesian.xyz.value

# UVW reference frame
w_dir = field_center_xyz
u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
v_dir = np.array(
    [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
     np.cos(field_center_lat)])
uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot3D([0, 1], [0, 0], [0, 0], '-ok', linewidth=2)
# ax.text3D(1, 0, 0, 'x', fontsize='large')
# ax.plot3D([0, 0], [0, 1], [0, 0], '-ok', linewidth=2)
# ax.text3D(0, 1, 0, 'y', fontsize='large')
# ax.plot3D([0, 0], [0, 0], [0, 1], '-ok', linewidth=2)
# ax.text3D(0, 0, 1, 'z', fontsize='large')
#
# ax.plot3D([0, u_dir[0]], [0, u_dir[1]], [0, u_dir[-1]], '-sr', linewidth=2)
# ax.text3D(u_dir[0], u_dir[1], u_dir[-1], 'u', fontsize='large')
# ax.plot3D([0, v_dir[0]], [0, v_dir[1]], [0, v_dir[-1]], '-sr', linewidth=2)
# ax.text3D(v_dir[0], v_dir[1], v_dir[-1], 'v', fontsize='large')
# ax.plot3D([0, w_dir[0]], [0, w_dir[1]], [0, w_dir[-1]], '-sr', linewidth=2)
# ax.text3D(w_dir[0], w_dir[1], w_dir[-1], 'w', fontsize='large')

# Imaging Parameters
N_pix = 512
N_level = 3
N_bits = 32
precision = 'single'
time_slice = 200 #36
eps = 1e-3
print("\nImaging Parameters")
print(f'N_pix {N_pix}\nN_level {N_level}\nN_bits {N_bits}')
print(f'time_slice {time_slice}\neps {eps}\n')

t1 = tt.time()

# Imaging grid
ig_s = tt.time()
lim = np.sin(FoV / 2)
pix_slice = np.linspace(-lim, lim, N_pix)
Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
pix_xyz = np.tensordot(uvw_frame, lmn_grid, axes=1)
_, pix_lat, pix_lon = transform.cart2eq(*pix_xyz)
ig_e = tt.time()
print(f"#@#IG {ig_e-ig_s:.3f} sec")

# ax.scatter3D(pix_xyz[0].flatten(), pix_xyz[1].flatten(), pix_xyz[-1].flatten())

# plt.figure()
# plt.scatter(lmn_grid[0], lmn_grid[1], s=2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.figure()
# plt.scatter(pix_lon * 180 / np.pi, pix_lat * 180 / np.pi, s=2)
# plt.scatter(field_center_lon * 180 / np.pi, field_center_lat * 180 / np.pi, c='r', s=10)
# plt.xlabel('RA')
# plt.ylabel('DEC')


### Intensity Field ===========================================================
# Parameter Estimation
ifpe_s = tt.time()
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
#for t in ProgressBar(time[::200]):
for t in time[::200]:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(S, G)

N_eig, c_centroid = I_est.infer_parameters()
intervals = bb_dp.centroid_to_intervals(c_centroid)
ifpe_e = tt.time()
print(f"#@#IFPE {ifpe_e-ifpe_s:.3f} sec")

# Imaging
ifim_s = tt.time()
I_dp   = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp  = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq','sqrt'))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
###UVW_baselines = []
###gram_corrected_visibilities = []
###baseline_rescaling = 2 * np.pi / wl

#for t in ProgressBar(time[0:25]):
for t in time[::time_slice]:
    XYZ = dev(t)
    UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
    UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    S_corrected = IV_dp(D, V, W, intervals)
    nufft_imager.collect(UVW_baselines_t, S_corrected)

# NUFFT Synthesis
lsq_image, sqrt_image = nufft_imager.get_statistic()

ifim_e = tt.time()
print(f"#@#IFIM {ifim_e-ifim_s:.3f} sec")


### Sensitivity Field =========================================================
# Parameter Estimation
sfpe_s = tt.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
#for t in ProgressBar(time[::200]):
for t in time[::200]:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)

N_eig = S_est.infer_parameters()

sfpe_e = tt.time()
print(f"#@#SFPE {sfpe_e-sfpe_s:.3f} sec")

# Imaging
sfim_s = tt.time()
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
sensitivity_coeffs = []
#for t in ProgressBar(time[0:25]):
for t in time[::time_slice]:
    XYZ = dev(t)
    UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
    UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])
    W = mb(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)
    S_sensitivity = SV_dp(D, V, W)
    sensitivity_coeffs.append(S_sensitivity)
    
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)

sensitivity_image = nufft_imager.get_statistic()[0]

I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
dump_data(I_lsq_eq.data, 'I_lsq_eq_data')
dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid')

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
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180/np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')

fp = "test_nufft"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)
