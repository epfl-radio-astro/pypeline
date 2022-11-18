# #############################################################################
# lofar_toothbrush_ps.py
# ======================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Real-data LOFAR imaging with Bluebild (PeriodicSynthesis).
Compare Bluebild image with WSCLEAN image.
"""

import os
if os.getenv('OMP_NUM_THREADS') == None : os.environ['OMP_NUM_THREADS'] = "1"

import bluebild_tools.cupy_util as bbt_cupy
use_cupy = bbt_cupy.is_cupy_usable()

from tqdm import tqdm as ProgressBar
import astropy.units as u
import imot_tools.io.fits as ifits
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import time
import finufft

import pypeline.phased_array.mscudf as mscudf
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.math.sphere.transform as transform
import pycsou.linop as pyclop
from imot_tools.math.func import SphericalDirichlet
import joblib as job


# For CuPy agnostic code
# ----------------------
xp = bbt_cupy.cupy if use_cupy else np


start_time = time.process_time()

# Instrument
N_station = 37
cudf_file = "/work/ska/cudfoutput/gauss4_t201806301100_SBL180.parquet"
cudfms = mscudf.LofarMeasurementSet(cudf_file, N_station) # stations 1 - N_station 
gram = bb_gr.GramBlock()

print("Reading {0}\nUsing {1} stations".format(cudf_file, N_station))

# Observation
FoV = np.deg2rad(5)
channel_id = 0
frequency = cudfms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
sky_model = source.from_tgss_catalog(cudfms.field_center, FoV, N_src=4)
obs_start, obs_end = cudfms.time[[0, -1]]

# Imaging
N_level = 4
N_bits = 32
#R = ms.instrument.icrs2bfsf_rot(obs_start, obs_end)
#colat_idx, lon_idx, pix_colat, pix_lon = grid.equal_angle(
#    N=ms.instrument.nyquist_rate(wl),
#    direction=R @ ms.field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
#    FoV=FoV,
#)

_, _, px_colat, px_lon = grid.equal_angle(
    N=cudfms.instrument.nyquist_rate(wl), direction=cudfms.field_center.cartesian.xyz.value, FoV=FoV
)

print(cudfms.instrument.nyquist_rate(wl), "Nstation {0}, px_col {1}, px_lon {2}".format(N_station, px_colat.shape, px_lon.shape))

#N_FS, T_kernel = ms.instrument.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(10)
#px_grid = transform.pol2cart(1, px_colat, px_lon).reshape(3, -1)
px_grid = transform.pol2cart(1, px_colat, px_lon)
time_slice = 200
print("Grid size is:", px_colat.shape, px_lon.shape)

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        cudfms.visibilities(
            channel_id=[channel_id], time_id=slice(0, None, 200), column="DATA"
        )
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = cudfms.instrument(t)
    W = cudfms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = mscudf.filter_data(S, W)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

print("N_eig:", N_eig)

# Imaging
print ("centroids = ", c_centroid)
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
#I_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
for t, f, S in ProgressBar(
        cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, 10, 1), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = cudfms.instrument(t)
    W = cudfms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, XYZ, W, wl)
    print(c_idx)
    c_idx = [0,1,2,3]

    #_ = I_mfs(D, V, XYZ.data, W.data, c_idx)

    XYZ = xp.asarray(XYZ.data)
    W   = xp.asarray(W.data)
    V   = xp.asarray(V)

    _ = I_mfs(D, V, XYZ, W, c_idx)
    
I_std, I_lsq = I_mfs.as_image()

end_time = time.process_time()
print("Time elapsed: {0}s".format(end_time - start_time))


### Sensitivity Field =========================================================
# Parameter Estimation
'''S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::200]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

print("Running sensitivity imaging")
# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
#S_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, time_slice), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(G)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()'''

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=N_level, nrows=2)
I_std_eq = s2image.Image(I_std.data, I_std.grid) #  / S.data
I_lsq_eq = s2image.Image(I_lsq.data, I_lsq.grid) # / S.data

for i in range(N_level):
    I_std_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax[0,i])
    ax[0,i].set_title("Standardized Image Level = {0}".format(i))
    I_lsq_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax[1,i])
    ax[1,i].set_title("Least-Squares Image Level = {0}".format(i))
#fig.show()
#plt.show()
#sys.exit()
plt.savefig("4gauss_standard_new")


### Interpolate critical-rate image to any grid resolution ====================
# Example: to compare outputs of WSCLEAN and Bluebild with AstroPy/DS9, we
# interpolate the Bluebild estimate at CLEAN (cl_) sky coordinates.

# 1. Load pixel grid the CLEAN image is defined on.

start_interp_time = time.process_time()

cl_WCS = ifits.wcs("/work/backup/ska/gauss4/gauss4-image-pb.fits")
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

# 2. ICRS <> BFSF transform.
# Why are we doing this? The Bluebild image produced by PeriodicSynthesis lies
# in the BFSF frame. We therefore need to do the interpolation in BFSF
# coordinates.
#cl_pix_bfsf = np.tensordot(R, cl_pix_icrs, axes=1)
# TODO/NB: to modify for SS remove above line

# 3. Interpolation: Part I.
# Due to the high Nyquist rate in astronomy and large pixel count in the images,
# it is advantageous to do sparse interpolation. Doing so requires first
# computing the interpolation kernel's spatial support per output pixel.
#bb_pix_bfsf = transform.pol2cart(1, pix_colat, pix_lon)  # Bluebild critical support points
# TODO/NB: to modify for SS remove above line

dirichlet_kernel = SphericalDirichlet(N=cudfms.instrument.nyquist_rate(wl), approx=True)
nside = (cudfms.instrument.nyquist_rate(wl) + 1) / 3
nodal_width = 2.8345 / np.sqrt(12 * nside ** 2)
interpolator = pyclop.MappedDistanceMatrix(samples1=cl_pix_icrs.reshape(3, -1).transpose(), # output res, replace with icrs for SS
                                           samples2=px_grid.reshape(3, -1).transpose(), # input res, replace with icrs for SS
                                           function=dirichlet_kernel,
                                           mode='zonal', operator_type='sparse', max_distance=10 * nodal_width,
                                           #eps=1e-1,
                                           )

with job.Parallel(backend='loky', n_jobs=-1, verbose=True) as parallel:
    interpolated_maps = parallel(job.delayed(interpolator)
                                 (I_lsq_eq.data.reshape(N_level, -1)[n])
                                 for n in range(N_level))

f_interp = np.stack(interpolated_maps, axis=0).reshape((N_level,) + cl_pix_icrs.shape[1:])
f_interp = f_interp / (cudfms.instrument.nyquist_rate(wl) + 1)
f_interp = np.clip(f_interp, 0, None)
fig, ax = plt.subplots(ncols=N_level, nrows=2)

'''
for i in range(N_level):
    I_lsq_eq_orig = s2image.Image(I_lsq_eq.data[i,], I_lsq_eq.grid)
    I_lsq_eq_orig.draw(catalog=sky_model.xyz.T, ax=ax[0,i])
    ax[0,i].set_title("Critically sampled Bluebild Standard Image Level = {0}".format(i))

    I_lsq_eq_interp = s2image.Image(f_interp[i,], cl_pix_icrs)
    I_lsq_eq_interp.draw(ax=ax[1,i])
    ax[1,i].set_title("Interpolated Bluebild Standard Image Level = {0}".format(i))
plt.show()
plt.savefig("4gauss_interp")'''

# 5. Store the interpolated Bluebild image in standard-compliant FITS for view
# in AstroPy/DS9.
f_interp = (f_interp  # We need to transpose axes due to the FORTRAN
            .reshape(N_level, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
            .transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('bluebild_ss_4gauss_{0}Stations.fits'.format(N_station))

end_interp_time = time.process_time()

print("Time to make BB image: {0}s".format(end_time - start_time))
print("Time to reinterpolate image: {0}s".format(end_interp_time - start_interp_time))
