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

#spk edits
import astropy.io.fits as ap_fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

start_time = time.process_time()

# Instrument
N_station = 37
ms_file = "/work/ska/MeerKAT/1569274256_sdp_l0_wtspec_J0159.0-3413.ms"
ms = measurement_set.MwaMeasurementSet(ms_file)#, N_station) # stations 1 - N_station 
gram = bb_gr.GramBlock()

#print("Reading {0}\n".format(ms_file))

# Observation
FoV = np.deg2rad(1.011534)# FOV = 1.02 lambda/D (lambda = c/1283895507.8125 = 0.23366m, D = 13.5m) = 1.011534 degrees OR (from fits file) 250 * sqrt(2) * 0.0002777777 =0.98209 degrees
channel_id = np.arange(0, 1024, 2, dtype=np.int)
#channel_id = 512
frequency = ms.channels["FREQUENCY"][0] + (ms.channels["FREQUENCY"][-1] - ms.channels["FREQUENCY"][0])/2
#frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
#sky_model = source.from_tgss_catalog(ms.field_center, FoV, N_src=4)
obs_start, obs_end = ms.time["TIME"][[0, -1]]

print (frequency, wl)

# Imaging
N_level = 4
N_bits = 32
#R = ms.instrument.icrs2bfsf_rot(obs_start, obs_end)
#colat_idx, lon_idx, pix_colat, pix_lon = grid.equal_angle(
#    N=ms.instrument.nyquist_rate(wl),
#    direction=R @ ms.field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
#    FoV=FoV,
#)
"""
_, _, px_colat, px_lon = grid.equal_angle(
    N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV
)

print("nyquist rate", ms.instrument.nyquist_rate(wl), "px_col {0}, px_lon {1}".format(px_colat.shape, px_lon.shape))

#N_FS, T_kernel = ms.instrument.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(10)
#px_grid = transform.pol2cart(1, px_colat, px_lon).reshape(3, -1)
px_grid = transform.pol2cart(1, px_colat, px_lon)
## replace above px_grid from the WS CLEAN FILE USING SAME CODE AS BELOW WHERE WSCLEAN GRID IS EXTRACTED
"""
cl_WCS = ifits.wcs("/work/backup/ska/wsclean/wsclean-image-J0159.fits")
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, 2), slice(None, None, 2)))  # downsample, too high res!
cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame

px_grid = cl_pix_icrs
#"""

time_slice = 182
print("Grid size is:", px_grid.shape[1], px_grid.shape[2])

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, time_slice), column="CORRECTED_DATA")
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
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits) # replace px_grid 
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, time_slice), column="CORRECTED_DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    print(c_idx)
    c_idx = [0,1,2,3]

    #_ = I_mfs(D, V, XYZ.data, W.data, c_idx)
    """
    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.data.toarray())
    V_gpu  = cp.asarray(V)

    _ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    """
    _ = I_mfs(D, cp.asarray(V), cp.asarray(XYZ.data), cp.asarray(W.data.toarray()), c_idx)

I_std, I_lsq = I_mfs.as_image()


"""
### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
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
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, time_slice), column="CORRECTED_DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(G)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()

I_std_eq = s2image.Image(I_std.data/S.data, I_std.grid) #  / S.data
I_lsq_eq = s2image.Image(I_lsq.data/S.data, I_lsq.grid) # / S.data
"""
I_std_eq = s2image.Image(I_std.data, I_std.grid) #  / S.data
I_lsq_eq = s2image.Image(I_lsq.data, I_lsq.grid) # / S.data
#"""

end_time = time.process_time()
print("Time elapsed: {0}s".format(end_time - start_time))

# Plot Results ================================================================
fig, ax = plt.subplots(figsize=(20,20), ncols=N_level + 1, nrows=2)


for i in np.arange(N_level + 1):
    if (i == 0):
        I_std_eq.draw(ax=ax[0,i])
        ax[0,i].set_title("Standardized Image") 
        I_lsq_eq.draw(ax=ax[1,i])
        ax[1,i].set_title("Least-Squares Image")
    else:
        I_std_eq.draw(index=i-1, ax=ax[0,i]) # catalog=sky_model.xyz.T,
        ax[0,i].set_title("Standardized Image Level = {0}".format(i))
        I_lsq_eq.draw(index=i-1, ax=ax[1,i]) # catalog=sky_model.xyz.T,
        ax[1,i].set_title("Least-Squares Image Level = {0}".format(i))
fig.show()
plt.tight_layout()
#plt.show()
#sys.exit()
plt.savefig("/scratch/izar/krishna/meerkat_bluebild_J0159.png")

fig, ax = plt.subplots(figsize=(20,20), ncols=3, nrows=1)

I_std_eq.draw(ax=ax[0])
ax[0].set_title("Standardized Image") 
I_lsq_eq.draw(ax=ax[1])
ax[1].set_title("Least-Squares Image")

WSClean_image = ap_fits.getdata("/scratch/izar/krishna/ws-clean-image-J0159-500x500-image.fits")
WSClean_scale= ax[2].imshow(WSClean_image[0, 0, :, :], cmap='RdPu')
ax[2].set_title("WSClean Image")
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(WSClean_scale, cax)



fig.show()
plt.tight_layout()
#plt.show()
#sys.exit()
#plt.savefig("/scratch/izar/krishna/meerkat_bluebild_WSClean_Comparison_J0159_chanel_{0}.png".format(channel_id))
plt.savefig("/scratch/izar/krishna/meerkat_bluebild_WSClean_Comparison_J0159.png")


### Interpolate critical-rate image to any grid resolution ====================
# Example: to compare outputs of WSCLEAN and Bluebild with AstroPy/DS9, we
# interpolate the Bluebild estimate at CLEAN (cl_) sky coordinates.

# 1. Load pixel grid the CLEAN image is defined on.

start_interp_time = time.process_time()

cl_WCS = ifits.wcs("/work/backup/ska/wsclean/wsclean-image-J0159.fits")
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, 2), slice(None, None, 2)))  # downsample, too high res!
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

dirichlet_kernel = SphericalDirichlet(N=ms.instrument.nyquist_rate(wl), approx=True)
nside = (ms.instrument.nyquist_rate(wl) + 1) / 3
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
print("test")
f_interp = np.stack(interpolated_maps, axis=0).reshape((N_level,) + cl_pix_icrs.shape[1:])
f_interp = f_interp / (ms.instrument.nyquist_rate(wl) + 1)
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
I_lsq_eq_interp.to_fits('/scratch/izar/krishna/bluebild_ss_meerKAT_J0159.fits')

end_interp_time = time.process_time()

print("Time to make BB image: {0}s".format(end_time - start_time))
print("Time to reinterpolate image: {0}s".format(end_interp_time - start_interp_time))
