"""
Simulated SKA-LOW imaging with Bluebild (StandardSynthesis).
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
#import imot_tools.math.sphere.interpolate as interpolate
#import imot_tools.math.sphere.transform as transform
#import pycsou.linop as pyclop
#from imot_tools.math.func import SphericalDirichlet
import joblib as job

start_time = time.process_time()


# Instrument
cl_WCS = ifits.wcs("/home/etolley/rascil_ska_sim/results_lofar_time/imaging_dirty.fits")
ms_file = "/home/etolley/rascil_ska_sim/results_lofar_time/lofar-pipeline_simulation.ms"

#cl_WCS = ifits.wcs("/work/ska/lofar30MHz_256/lofar30MHz256-image.fits")
#ms_file = "/work/ska/lofar30MHz_256/lofar30MHz_t201806301100_SBH256.MS"
ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
out_str = "skalow_small"


gram = bb_gr.GramBlock()
print(cl_WCS.to_header())
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, 5), slice(None, None, 5)))  # downsample, too high res!
cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

print("Reading {0}\n".format(ms_file))

# Observation
#FoV = np.deg2rad(5)
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
#sky_model = source.from_tgss_catalog(ms.field_center, FoV, N_src=4)
obs_start, obs_end = ms.time["TIME"][[0, -1]]

print("obs start: {0}, end: {1}".format(obs_start, obs_end))
print(ms.time["TIME"])

# Imaging
N_level = 1
N_bits = 32
#R = ms.instrument.icrs2bfsf_rot(obs_start, obs_end)
#colat_idx, lon_idx, pix_colat, pix_lon = grid.equal_angle(
#    N=ms.instrument.nyquist_rate(wl),
#    direction=R @ ms.field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
#    FoV=FoV,
#)

#_, _, px_colat, px_lon = grid.equal_angle(
#    N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV
#)

#N_FS, T_kernel = ms.instrument.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(10)
#px_grid = transform.pol2cart(1, px_colat, px_lon).reshape(3, -1)
px_grid = cl_pix_icrs
time_slice = 10
print("Grid size is:", px_grid.shape)

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
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, None), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    print(c_idx)
    #c_idx = [0,1,2,3]

    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.data)
    V_gpu  = cp.asarray(V)
    _ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    
I_std, I_lsq = I_mfs.as_image()

end_time = time.process_time()
print("Time elapsed: {0}s".format(end_time - start_time))

'''### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::200]):
    XYZ = ms.instrument(t,field_center = ms.field_center)
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
    XYZ = ms.instrument(t,field_center = ms.field_center)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)
    D, V = S_dp(G)
    #_ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.data.toarray())
    V_gpu  = cp.asarray(V)
    _ = S_mfs(D, V_gpu, XYZ_gpu, W_gpu, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()'''

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=N_level, nrows=2, figsize=(16, 10))
#I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid) 
#I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid) 
I_std_eq = s2image.Image(I_std.data, I_std.grid) 
I_lsq_eq = s2image.Image(I_lsq.data, I_lsq.grid) 

for i in range(N_level):
    top_plot = ax[0] if N_level == 1 else ax[0,i]
    bottom_plot = ax[1] if N_level == 1 else ax[1,i]
    I_std_eq.draw(index=i, ax= top_plot)
    top_plot.set_title("Standardized Image Level = {0}".format(i))
    I_lsq_eq.draw(index=i, ax=bottom_plot)
    bottom_plot.set_title("Least-Squares Image Level = {0}".format(i))

plt.savefig("lofartest_standard_new")

# 5. Store the interpolated Bluebild image in standard-compliant FITS for view
# in AstroPy/DS9.

f_interp = (I_lsq_eq.data  # We need to transpose axes due to the FORTRAN
            .reshape(N_level, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
            .transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(np.sum(f_interp,axis=0), cl_WCS)
I_lsq_eq_interp.to_fits('bluebild_ss_lofartest_combined-test.fits')
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('bluebild_ss_lofartest_levels-test.fits')

end_interp_time = time.process_time()

print("Time to make BB image: {0}s".format(end_time - start_time))