# #############################################################################
# lofar_toothbrush_ps.py
# ======================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Real-data LOFAR imaging with Bluebild (PeriodicSynthesis).
Compare Bluebild image with WSCLEAN image.
"""

from tqdm import tqdm as ProgressBar
import astropy.units as u
import imot_tools.io.fits as ifits
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import scipy.constants as constants
import sys, time
import finufft

import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
#import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.math.sphere.transform as transform
import pycsou.linop as pyclop
from imot_tools.math.func import SphericalDirichlet
import joblib as job

start_time = time.process_time()

cl_WCS = ifits.wcs("/work/ska/gauss4/gauss4-image-pb.fits")
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

# Instrument
N_station = 37
ms_file = "/work/ska/gauss4/gauss4_t201806301100_SBL180.MS"
ms = measurement_set.LofarMeasurementSet(ms_file, N_station) # stations 1 - N_station 
gram = bb_gr.GramBlock()

print("Reading {0}\nUsing {1} stations".format(ms_file, N_station))

# Observation
FoV = np.deg2rad(5)
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
sky_model = source.from_tgss_catalog(ms.field_center, FoV, N_src=4)
obs_start, obs_end = ms.time["TIME"][[0, -1]]

# Imaging
N_level = 4
N_bits = 32

#px_grid = transform.pol2cart(1, px_colat, px_lon)
px_grid = cl_pix_icrs
time_slice = 200
print("Grid size is:", cl_pix_icrs.shape)

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        ms.visibilities(
            channel_id=[channel_id], time_id=slice(0, None, 200), column="DATA"
        )
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

print("N_eig:", N_eig)

# Imaging
print ("centroids = ", c_centroid)
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
#I_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, 1, 1), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    print(V.shape, D.shape, S.shape, G.shape)
    c_idx = [0,1,2,3]

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    p=ax[0].imshow(np.abs(S.data))

    p2=ax[1].imshow(np.abs(G.data))
    ax[0].set_title('original visibilities')
    ax[1].set_title('Gram matrix')
    plt.colorbar(p, ax=ax[0])
    plt.colorbar(p2, ax = ax[1])
    plt.savefig("visibility_matrix")

    fig, ax = plt.subplots(ncols=len(D), nrows=1, figsize=(10, 3))
    plt.tight_layout(w_pad = 2)
    for i,d in enumerate(D):
        mask = np.zeros(D.shape)
        mask[i] = 1
        D_corr = D*mask
        S_corrected = (V @ np.diag(D_corr)) @ V.transpose().conj()
        p=ax[i].imshow(np.abs(S_corrected))
        ax[i].set_title('eigen visibilities {0}'.format(i))
        plt.colorbar(p, ax=ax[i], fraction=0.046)
        
    plt.savefig("eigen_visibilities")

    #_ = I_mfs(D, V, XYZ.data, W.data, c_idx)

    S_corrected = G.data@ V @ np.diag(D) @ V.transpose().conj()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    p=ax.imshow(np.abs(S_corrected))
    ax.set_title('corrected visibilities')
    plt.colorbar(p)
    plt.savefig("corrected_visibility_matrix")

    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.data.toarray())
    V_gpu  = cp.asarray(V)

    _ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    
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
fig, ax = plt.subplots(ncols=N_level, nrows=1)
I_std_eq = s2image.Image(I_std.data, I_std.grid) #  / S.data
I_lsq_eq = s2image.Image(I_lsq.data, I_lsq.grid) # / S.data

for i in range(N_level):
    #I_std_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax[0,i])
    #ax[0,i].set_title("Standardized Image Level = {0}".format(i))
    I_lsq_eq.draw(index=i, ax=ax[i])
    ax[i].set_title("Level = {0}".format(i))
#fig.show()
#plt.show()
#sys.exit()
plt.savefig("4gauss_standard_new")


### Interpolate critical-rate image to any grid resolution ====================
# Example: to compare outputs of WSCLEAN and Bluebild with AstroPy/DS9, we
# interpolate the Bluebild estimate at CLEAN (cl_) sky coordinates.

# 1. Load pixel grid the CLEAN image is defined on.

start_interp_time = time.process_time()

# 5. Store the interpolated Bluebild image in standard-compliant FITS for view
# in AstroPy/DS9.
f_interp = (I_lsq_eq.data  # We need to transpose axes due to the FORTRAN
            .reshape(N_level, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
            .transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('bluebild_ss_4gauss_{0}Stations.fits'.format(N_station))

end_interp_time = time.process_time()

print("Time to make BB image: {0}s".format(end_time - start_time))
print("Time to reinterpolate image: {0}s".format(end_interp_time - start_interp_time))
