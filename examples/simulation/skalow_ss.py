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
import astropy.coordinates as coord

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
cl_WCS = ifits.wcs("/work/ska/results_rascil_skalow_small/wsclean-image.fits")
ms_file = "/work/ska/results_rascil_skalow_small/ska-pipeline_simulation.ms"
ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
out_str = "skalow_small"

field_center = coord.SkyCoord(ra=15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
FoV = np.deg2rad(5.56111)

field_center_lon, field_center_lat = field_center.data.lon.rad, field_center.data.lat.rad
field_center_xyz = field_center.cartesian.xyz.value
print("field_center_xyz", type(field_center_xyz), field_center_xyz)

# UVW reference frame
w_dir = field_center_xyz
u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
v_dir = np.array(
    [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
    np.cos(field_center_lat)])
uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)
print("uvw_frame\n", uvw_frame)
#continue

gram = bb_gr.GramBlock()
print(cl_WCS.to_header())
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None,10), slice(None, None, 10)))  # downsample, too high res!
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
        ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, time_slice), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)
    D, V, c_idx = I_dp(S, XYZ, W, wl)

    #XYZ_gpu = cp.asarray(XYZ.data)
    #W_gpu  = cp.asarray(W.data)
    #V_gpu  = cp.asarray(V)
    #_ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
    
I_std, I_lsq = I_mfs.as_image()

end_time = time.process_time()
print("Time elapsed: {0}s".format(end_time - start_time))


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

plt.savefig("skalow_standard_new")
plt.show()

# 5. Store the interpolated Bluebild image in standard-compliant FITS for view
# in AstroPy/DS9.

f_interp = (I_lsq_eq.data  # We need to transpose axes due to the FORTRAN
            .reshape(N_level, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
            .transpose(0, 2, 1))
#f_interp = I_lsq_eq.data 
#f_interp = np.rot90(f_interp, 2, axes=(1,2))
#f_interp = np.flip(f_interp, axis=2)
I_lsq_eq_interp = s2image.WCSImage(np.sum(f_interp,axis=0), cl_WCS)
I_lsq_eq_interp.to_fits('bluebild_ss_skalow_combined-test.fits')
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('bluebild_ss_skalow_levels-test.fits')

end_interp_time = time.process_time()

print("Time to make BB image: {0}s".format(end_time - start_time))
