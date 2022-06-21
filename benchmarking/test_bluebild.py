# #############################################################################
# lofar_bootes_ps_small_fov.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com] (modified by Matthieu)
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (Standard, Periodic, and nufft).
"""

import os
if os.getenv('OMP_NUM_THREADS') == None : os.environ['OMP_NUM_THREADS'] = "1"

import bluebild_tools.cupy_util as bbt_cupy
use_cupy = bbt_cupy.is_cupy_usable()

from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import finufft

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
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


# For CuPy agnostic code
# ----------------------
xp = bbt_cupy.cupy if use_cupy else np


t = Timer()

do_spherical_interpolation = True # BEWARE, if set to true the runtime becomes very slow!!
do_periodic_synthesis = True
timeslice = slice(None, None, 100)

t.start_time("Set up data")


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
t1 = tt.time()
N_level = 4
N_bits = 32
R = dev.icrs2bfsf_rot(obs_start, obs_end)
_, _, pix_colat, pix_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl),
    direction=R @ field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
    FoV=1.25 * FoV,
)
N_FS, T_kernel = dev.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(16)

### Standard Synthesis Imaging parameters ===========================================================
_, _, px_colat, px_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=1.25*FoV
)

px_grid = transform.pol2cart(1, px_colat, px_lon)

print('''You are running bluebild with the following input parameters:
         {0} timesteps
         {1} stations
         clustering into {2} levels
         The output grid will be {3}x{4} = {5} pixels'''.format(len(time[timeslice]), N_station,N_level, px_grid.shape[1],  px_grid.shape[2],  px_grid.shape[1]* px_grid.shape[2]))

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

### Intensity Field =================================================
# Parameter Estimation
t.start_time("Estimate intensity field parameters")
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for ti in ProgressBar(time[::200]):
    XYZ = dev(ti)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
t.end_time("Estimate intensity field parameters")

####################################################################
#### Imaging
####################################################################
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs_ps = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
I_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
UVW_baselines = []
ICRS_baselines = []
gram_corrected_visibilities = []
baseline_rescaling = 2 * np.pi / wl
for ti in ProgressBar(time[timeslice]):
    t.start_time("Synthesis: prep input matrices & fPCA")
    XYZ = dev(ti)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)

    D, V, c_idx = I_dp(S, XYZ, W, wl)
    t.end_time("Synthesis: prep input matrices & fPCA")

    t.start_time("Periodic Synthesis")
    _ = I_mfs_ps(D, V, XYZ.data, W.data, c_idx)
    t.end_time("Periodic Synthesis")

    t.start_time("Standard Synthesis")
    if use_cupy:
        XYZ_gpu = xp.asarray(XYZ.data)
        W_gpu  = xp.asarray(W.data.toarray())
        V_gpu  = xp.asarray(V)
        _ = I_mfs_ss(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    else:
        _ = I_mfs_ss(D, V, XYZ.data, W.data, c_idx)
    t.end_time("Standard Synthesis")

    t.start_time("NUFFT Synthesis 1")
    UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
    UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])
    ICRS_baselines_t = (XYZ.data[:, None, :] - XYZ.data[None, ...])
    UVW_baselines.append(baseline_rescaling * UVW_baselines_t)
    ICRS_baselines.append(baseline_rescaling * ICRS_baselines_t)
    W = W.data
    S_corrected  = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    #S_corrected2 = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    gram_corrected_visibilities.append(S_corrected)
    t.end_time("NUFFT Synthesis 1")


I_std_ps, I_lsq_ps = I_mfs_ps.as_image()
I_std_ss, I_lsq_ss = I_mfs_ss.as_image()

t.start_time("NUFFT Synthesis 2")
UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
ICRS_baselines = np.stack(ICRS_baselines, axis=0)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=0).reshape(-1)

w_correction = np.exp(1j * UVW_baselines[:, -1])
gram_corrected_visibilities_nufft = gram_corrected_visibilities*w_correction

print("test")

scalingx = 2 * lim / N_pix
scalingy = 2 * lim / N_pix
bb_image = finufft.nufft2d1(x=scalingx * UVW_baselines[:, 1],
                            y=scalingy * UVW_baselines[:, 0],
                            c=gram_corrected_visibilities_nufft,
                            n_modes=N_pix, eps=1e-4)

bb_image = np.real(bb_image)
t.end_time("NUFFT Synthesis 2")
print(bb_image.shape,bb_image[0,0])


#====

### Sensitivity Field =========================================================
# Parameter Estimation
t.start_time("Estimate sensitivity field parameters")
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for ti in ProgressBar(time[::200]):
    XYZ = dev(ti)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
t.end_time("Estimate sensitivity field parameters")

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs_ps = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
S_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
sensitivity_coeffs = []
for ti in ProgressBar(time[timeslice]): 

    XYZ = dev(ti)
    W = mb(XYZ, wl)

    D, V = S_dp(XYZ, W, wl)
    W = W.data

    _ = S_mfs_ps(D, V, XYZ.data, W, cluster_idx=np.zeros(N_eig, dtype=int))

    if use_cupy:
        XYZ_gpu = xp.asarray(XYZ.data)
        W_gpu  = xp.asarray(W.toarray())
        V_gpu  = xp.asarray(V)
        #_ = I_mfs_ss(D, V, XYZ.data, W.data, c_idx)
        #_ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
        _ = S_mfs_ss(D, V_gpu, XYZ_gpu, W_gpu, cluster_idx=np.zeros(N_eig, dtype=int))
    else:
        _ = S_mfs_ss(D, V, XYZ.data, W, cluster_idx=np.zeros(N_eig, dtype=int))

    S_sensitivity = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    sensitivity_coeffs.append(S_sensitivity)
_, S_ps = S_mfs_ps.as_image()
_, S_ss = S_mfs_ss.as_image()

I_lsq_eq_ps = s2image.Image(I_lsq_ps.data / S_ps.data, I_lsq_ps.grid)
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

if do_spherical_interpolation:

    #pix_xyz_interp = pix_xyz[:,::5, ::5]  # downsample, too high res!
    pix_xyz_interp = pix_xyz
    dirichlet_kernel = SphericalDirichlet(N=dev.nyquist_rate(wl), approx=True)

    nside = (dev.nyquist_rate(wl) + 1) / 3
    nodal_width = 2.8345 / np.sqrt(12 * nside ** 2)
    interpolator_ss = pyclop.MappedDistanceMatrix(samples1=pix_xyz_interp.reshape(3, -1).transpose(), # output res
                                               samples2=px_grid.reshape(3, -1).transpose(), # input res
                                               function=dirichlet_kernel,
                                               mode='zonal', operator_type='sparse', max_distance=10 * nodal_width,
                                               #eps=1e-1,
                                               )

    with job.Parallel(backend='loky', n_jobs=-1, verbose=True) as parallel:
        interpolated_maps_ss = parallel(job.delayed(interpolator_ss)
                                     (I_lsq_eq_ss.data.reshape(N_level, -1)[n]) for n in range(N_level)
                                     )

    f_interp_ss = np.stack(interpolated_maps_ss, axis=0).reshape((N_level,) + pix_xyz_interp.shape[1:])
    f_interp_ss = f_interp_ss / (dev.nyquist_rate(wl) + 1)
    f_interp_ss = np.clip(f_interp_ss, 0, None)

    #============================================================================================
    if do_periodic_synthesis:
        # 2. ICRS <> BFSF transform.
        # Why are we doing this? The Bluebild image produced by PeriodicSynthesis lies
        # in the BFSF frame. We therefore need to do the interpolation in BFSF
        # coordinates.
        pix_bfsf = np.tensordot(R, pix_xyz_interp, axes=1)
        # TODO/NB: to modify for SS remove above line

        # 3. Interpolation: Part I.
        # Due to the high Nyquist rate in astronomy and large pixel count in the images,
        # it is advantageous to do sparse interpolation. Doing so requires first
        # computing the interpolation kernel's spatial support per output pixel.
        bb_pix_bfsf = transform.pol2cart(1, pix_colat, pix_lon)  # Bluebild critical support points
        # TODO/NB: to modify for SS remove above line

        interpolator_ps = pyclop.MappedDistanceMatrix(samples1=pix_bfsf.reshape(3, -1).transpose(), # output res
                                                   samples2=bb_pix_bfsf.reshape(3, -1).transpose(), # input res
                                                   function=dirichlet_kernel,
                                                   mode='zonal', operator_type='sparse', max_distance=10 * nodal_width,
                                                   #eps=1e-1,
                                                   )

        with job.Parallel(backend='loky', n_jobs=-1, verbose=True) as parallel:
            interpolated_maps_ps = parallel(job.delayed(interpolator_ps)
                                         (I_lsq_eq_ps.data.reshape(N_level, -1)[n])
                                         for n in range(N_level))

        f_interp_ps = np.stack(interpolated_maps_ps, axis=0).reshape((N_level,) + pix_bfsf.shape[1:])
        f_interp_ps = f_interp_ps / (dev.nyquist_rate(wl) + 1)
        f_interp_ps = np.clip(f_interp_ps, 0, None)


#============================================================================================

t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')
print(f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180/np.pi)} degrees.\n'
      f'Run time {np.floor(t2 - t1)} seconds.')
if do_spherical_interpolation:
    fig, ax = plt.subplots(ncols=3, nrows = 2, figsize=(16, 8))
else:
    fig, ax = plt.subplots(ncols=3, nrows = 1, figsize=(16, 8))
ax = ax.flatten()
I_lsq_eq_ss.draw(catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[0].set_title('Standard Synthesis')

I_lsq_eq_ps.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[1].set_title('Periodic Synthesis')

I_lsq_eq_nufft.draw(catalog=sky_model.xyz.T, ax=ax[2], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[2].set_title('NUFFT')


if do_spherical_interpolation:
    I_lsq_eq_ss_interp = s2image.Image(f_interp_ss, pix_xyz_interp)
    I_lsq_eq_ss_interp.draw(catalog=sky_model.xyz.T, ax=ax[3], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
    ax[3].set_title('Interpolated Standard Synthesis')
    if do_periodic_synthesis:
        I_lsq_eq_ps_interp = s2image.Image(f_interp_ps, pix_xyz_interp)
        I_lsq_eq_ps_interp.draw(catalog=sky_model.xyz.T, ax=ax[4], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
        ax[4].set_title('Interpolated Periodic Synthesis')
        np.save("bluebild_ps_img", f_interp_ps)
    plt.savefig("test_bluebild2")


    np.save("bluebild_ss_img", f_interp_ss)
    np.save("bluebild_nufft_img", I_lsq_eq_nufft.data)

else:
    plt.savefig("test_bluebild_planes")
t.print_summary()
np.save("bluebild_np_grid", pix_xyz)

#===
