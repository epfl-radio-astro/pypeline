# #############################################################################
# lofar_bootes_ps_small_fov.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com] (modified by Matthieu)
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (Standard, Periodic, and nufft).
"""

'''export OMP_NUM_THREADS=1''' 

import sys
import os
#from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import scipy.constants as constants
import finufft

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
#import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.transform as transform
import time as tt
import pycsou.linop as pyclop
from imot_tools.math.func import SphericalDirichlet
import joblib as job
from timing import Timer


def print_info(npa, label):
    try:
        print(f'{label:8s} shape={str(npa.shape):18s} dtype={str(npa.dtype):12s} size={npa.nbytes / 1.E9:.3f} GB, type {type(npa)}')
    except:
        print(f'{label:8s} type {type(npa)}')


np.random.seed(0)

do_spherical_interpolation = False # BEWARE, if set to true the runtime becomes very slow!!
timeslice = slice(0,1,None)

# Get relevant environment variables
SKABB_C_SYNTH = os.environ.get('SKABB_C_SYNTH')
C_SYNTH = False
if SKABB_C_SYNTH == "1": C_SYNTH = True
OMP_NUM_THREADS = os.environ.get('OMP_NUM_THREADS')

t = Timer()
t1 = tt.time()

t.start_time("Set up data")
tic = tt.time()
# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
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
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=30)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

### Periodic Synthesis Imaging parameters ===========================================================
N_level = 4
N_bits  = 32

'''R = dev.icrs2bfsf_rot(obs_start, obs_end)
_, _, pix_colat, pix_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl),
    direction=R @ field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
    FoV=1.25 * FoV,
)
N_FS, T_kernel = dev.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(16)'''

### Standard Synthesis Imaging parameters ===========================================================
res_fac = 4
_, _, px_colat, px_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl) * res_fac, direction=field_center.cartesian.xyz.value, FoV=1.25*FoV
)

px_grid = transform.pol2cart(1, px_colat, px_lon)

print('''You are running bluebild with the following input parameters:
         T = {0} timesteps
         Ns = {1} stations
         clustering into {2} levels
         The output grid will be {3}x{4} = {5} pixels'''.format(len(time[timeslice]), N_station,N_level, px_grid.shape[1],  px_grid.shape[2],  px_grid.shape[1]* px_grid.shape[2]))

#sys.exit(0)


### NUFFT imaging parameters ===========================================================

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

# Imaging grid
lim = np.sin(FoV / 2)
N_pix = 512
pix_slice = np.linspace(-lim, lim, N_pix)
Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
pix_xyz = np.tensordot(uvw_frame, lmn_grid, axes=1)
_, pix_lat_sq, pix_lon_sq = transform.cart2eq(*pix_xyz)

t.end_time("Set up data")
t_setupdata = tt.time() - tic

### Intensity Field =================================================
# Parameter Estimation
t.start_time("Estimate intensity field parameters")
tic = tt.time()
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for ti in time[::200]: #ProgressBar(time[::200]):
    XYZ = dev(ti)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
t_estifp = tt.time() - tic
t.end_time("Estimate intensity field parameters")

print("decomposing image into Neig = {0} eigenlevels".format(N_eig))

####################################################################
#### Imaging
####################################################################
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs_ss     = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits) # for cpu
I_mfs_ss_gpu = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits) # for gpu

UVW_baselines = []
#ICRS_baselines = []
gram_corrected_visibilities = []
baseline_rescaling = 2 * np.pi / wl
for ti in time[timeslice]: #ProgressBar(time[timeslice]):

    t.start_time("Synthesis: prep input matrices & fPCA")
    XYZ = dev(ti)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)
    D, V, c_idx = I_dp(S, G)
    t.end_time("Synthesis: prep input matrices & fPCA")

    """
    #EO: ideally should all come in the right precision
    print_info(XYZ.data, 'XYZ.data')
    print_info(W.data, 'W.data')
    print_info(G.data, 'G.data')
    print_info(S.data, 'S.data')
    print_info(V, 'V')
    print_info(D, 'D')
    """

    #t.start_time("Periodic Synthesis")
    #_ = I_mfs_ps(D, V, XYZ.data, W.data, c_idx)
    #t.end_time("Periodic Synthesis")

    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.data.toarray())
    V_gpu  = cp.asarray(V)
    
    """
    print_info(XYZ_gpu, 'XYZ_gpu')
    print_info(W_gpu, 'W_gpu')
    print_info(V_gpu, 'V_gpu')
    print("\n")
    """

    t.start_time("Standard Synthesis GPU")
    tic = tt.time()
    I_GPU = I_mfs_ss_gpu(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    t_gpuruntime = tt.time() - tic
    #print_info(I_GPU, 'I_GPU')
    t.end_time("Standard Synthesis GPU")
 
    t.start_time("Standard Synthesis CPU")
    tic = tt.time()
    I_CPU = I_mfs_ss(D, V, XYZ.data, W.data, c_idx)
    t_cpuruntime = tt.time() - tic
    #print_info(I_CPU, 'I_CPU')
    t.end_time("Standard Synthesis CPU")

    #print(f"RMSE CPU - GPU = {np.sqrt(((I_CPU - I_GPU) ** 2).mean()):.10f}")
    #sys.exit(0)

    t.start_time("NUFFT Synthesis 1")
    tic = tt.time()
    print("uvw_frame shape:", uvw_frame.shape)
    print("XYZ shape:", XYZ.data.shape)
    UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
    print("UVW shape:", UVW.shape)
    print("UVW shape a:", UVW[:, None, :].shape)
    print("UVW shape b:", UVW[None, ...].shape)
    UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])
    #ICRS_baselines_t = (XYZ.data[:, None, :] - XYZ.data[None, ...])
    UVW_baselines.append(baseline_rescaling * UVW_baselines_t)
    #ICRS_baselines.append(baseline_rescaling * ICRS_baselines_t)
    print("UVW_baselines_t shape:", UVW_baselines_t.shape)
    W = W.data
    S_corrected  = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    #S_corrected2 = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    gram_corrected_visibilities.append(S_corrected)
    t_nufft1 = tt.time() - tic
    t.end_time("NUFFT Synthesis 1")


#I_std_ps, I_lsq_ps = I_mfs_ps.as_image()
I_std_ss, I_lsq_ss = I_mfs_ss.as_image()

t.start_time("NUFFT Synthesis 2")
tic = tt.time()
UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
#ICRS_baselines = np.stack(ICRS_baselines, axis=0)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=0).reshape(-1)

w_correction = np.exp(1j * UVW_baselines[:, -1])
print("w_correction shape", w_correction.shape)
print("gram_corrected_visibilities shape", gram_corrected_visibilities.shape)
gram_corrected_visibilities_nufft = gram_corrected_visibilities*w_correction

print("test")

scalingx = 2 * lim / N_pix
scalingy = 2 * lim / N_pix
bb_image = finufft.nufft2d1(x=scalingx * UVW_baselines[:, 1],
                            y=scalingy * UVW_baselines[:, 0],
                            c=gram_corrected_visibilities_nufft,
                            n_modes=N_pix, eps=1e-4) # eps is accuracy of output -> TODO: check effect of making this smaller

bb_image = np.real(bb_image)
t_nufft2 = tt.time() - tic
t.end_time("NUFFT Synthesis 2")
print(bb_image.shape,bb_image[0,0])


#====

### Sensitivity Field =========================================================
# Parameter Estimation
t.start_time("Estimate sensitivity field parameters")
tic = tt.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for ti in time[::200]: #ProgressBar(time[::200]):
    XYZ = dev(ti)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
t_estsfp = tt.time() - tic
t.end_time("Estimate sensitivity field parameters")

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
#S_mfs_ps = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
S_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
sensitivity_coeffs = []
for ti in time[timeslice]: #ProgressBar(time[timeslice]): 

    XYZ = dev(ti)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    D, V = S_dp(G)
    W = W.data

    #_ = S_mfs_ps(D, V, XYZ.data, W, cluster_idx=np.zeros(N_eig, dtype=int))

    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.toarray())
    V_gpu  = cp.asarray(V)
    #_ = I_mfs_ss(D, V, XYZ.data, W.data, c_idx)
    #_ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    _ = S_mfs_ss(D, V_gpu, XYZ_gpu, W_gpu, cluster_idx=np.zeros(N_eig, dtype=int))

    S_sensitivity = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    sensitivity_coeffs.append(S_sensitivity)
#_, S_ps = S_mfs_ps.as_image()
_, S_ss = S_mfs_ss.as_image()

#I_lsq_eq_ps = s2image.Image(I_lsq_ps.data / S_ps.data, I_lsq_ps.grid)
I_lsq_eq_ss = s2image.Image(I_lsq_ss.data / S_ss.data, I_lsq_ss.grid)


sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
sensitivity_coeffs *= w_correction
sensitivity_image = finufft.nufft2d1(x=scalingx * UVW_baselines[:, 1],
                                     y=scalingy * UVW_baselines[:, 0],
                                     c=sensitivity_coeffs,
                                     n_modes=N_pix, eps=1e-4)

sensitivity_image = np.real(sensitivity_image)
print(sensitivity_image.shape,sensitivity_image[0,0], pix_xyz[0,0,0])
I_lsq_eq_nufft = s2image.Image(bb_image / sensitivity_image, pix_xyz)

### Spherical reinterpolation Field =========================================================



#============================================================================================

t2 = tt.time()
print(f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180/np.pi)} degrees.\n'
      f'Run time {t2 - t1:.3f} seconds.')

fig, ax = plt.subplots(ncols=2, nrows = 1, figsize=(16, 8))
ax = ax.flatten()
I_lsq_eq_ss.draw(catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[0].set_title('Standard Synthesis')

I_lsq_eq_nufft.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[1].set_title('NUFFT')

plt.savefig("test_bluebild_ss_nufft")

#t.print_summary()

print(f"\nRun times summary + CPU/GPU comparison")
print(f"Time Set up data                            {t_setupdata:.3f} sec")
print(f"Time Estimate intensity field parameters    {t_estifp:.3f} sec")
print(f"Time GPU Standard Synthesis                 {t_gpuruntime:.3f} sec")
print(f"Time CPU Standard Synthesis                 {t_cpuruntime:.3f} sec  C_SYNTH? {C_SYNTH}; OMP_NUM_THREADS = {OMP_NUM_THREADS}")
print(f"RMSE CPU - GPU = {np.sqrt(((I_CPU - I_GPU) ** 2).mean()):.10f}")
print(f"Time NUFFT Synthesis 1                      {t_nufft1:.3f} sec")
print(f"Time NUFFT Synthesis 2                      {t_nufft2:.3f} sec")
print(f"Time Estimate sensitivity field parameters  {t_estsfp:.3f} sec")
t_cum = t_setupdata + t_estifp + t_gpuruntime + t_cpuruntime + t_nufft1 + t_nufft2 + t_estsfp
print(f"Cumulated vs total time {t_cum:.3f} {t2-t1:.3f} sec")
