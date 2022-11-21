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
import scipy.constants as constants

import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.math.sphere.transform as transform
import pycsou.linop as pyclop
from imot_tools.math.func import SphericalDirichlet
import joblib as job


# Instrument
N_station = 12
ms_file = "/Users/mmjasime/Documents/Datasets/RX42_SB100-109.2ch10s.ms"
ms_file = "/home/etolley/data/toothbrush/RX42_SB100-109.2ch10s.ms"
ms = measurement_set.LofarMeasurementSet(ms_file, N_station)
gram = bb_gr.GramBlock()

# Observation
FoV = np.deg2rad(8)
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
sky_model = source.from_tgss_catalog(ms.field_center, FoV, N_src=20)
obs_start, obs_end = ms.time["TIME"][[0, -1]]

# Imaging
N_level = 4
N_bits = 32
R = ms.instrument.icrs2bfsf_rot(obs_start, obs_end)
colat_idx, lon_idx, pix_colat, pix_lon = grid.equal_angle(
    N=ms.instrument.nyquist_rate(wl),
    direction=R @ ms.field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
    FoV=FoV,
)
N_FS, T_kernel = ms.instrument.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(10)
time_slice = 200

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        ms.visibilities(
            channel_id=[channel_id], time_id=slice(None, None, 200), column="DATA"
        )
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, time_slice), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, XYZ, W, wl)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
I_std, I_lsq = I_mfs.as_image()

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::200]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, time_slice), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(XYZ, W, wl)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
ax[0].set_title("Bluebild Standardized Image")

I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1])
ax[1].set_title("Bluebild Least-Squares Image")
fig.show()
plt.show()
plt.savefig('standard_fig')

### Interpolate critical-rate image to any grid resolution ====================
# Example: to compare outputs of WSCLEAN and Bluebild with AstroPy/DS9, we
# interpolate the Bluebild estimate at CLEAN (cl_) sky coordinates.

# 1. Load pixel grid the CLEAN image is defined on.
cl_WCS = ifits.wcs('/home/etolley//data/toothbrush/toothbrush-image.fits')
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

# 2. ICRS <> BFSF transform.
# Why are we doing this? The Bluebild image produced by PeriodicSynthesis lies
# in the BFSF frame. We therefore need to do the interpolation in BFSF
# coordinates.
cl_pix_bfsf = np.tensordot(R, cl_pix_icrs, axes=1)

# 3. Interpolation: Part I.
# Due to the high Nyquist rate in astronomy and large pixel count in the images,
# it is advantageous to do sparse interpolation. Doing so requires first
# computing the interpolation kernel's spatial support per output pixel.
bb_pix_bfsf = transform.pol2cart(1, pix_colat, pix_lon)  # Bluebild critical support points

dirichlet_kernel = SphericalDirichlet(N=ms.instrument.nyquist_rate(wl), approx=True)
nside = (ms.instrument.nyquist_rate(wl) + 1) / 3
nodal_width = 2.8345 / np.sqrt(12 * nside ** 2)
interpolator = pyclop.MappedDistanceMatrix(samples1=cl_pix_bfsf.reshape(3, -1).transpose(),
                                           samples2=bb_pix_bfsf.reshape(3, -1).transpose(),
                                           function=dirichlet_kernel,
                                           mode='zonal', operator_type='sparse', max_distance=10 * nodal_width,
                                           eps=1e-1)

with job.Parallel(backend='loky', n_jobs=-1, verbose=True) as parallel:
    interpolated_maps = parallel(job.delayed(interpolator)
                                 (I_lsq_eq.data.reshape(N_level, -1)[n])
                                 for n in range(N_level))

f_interp = np.stack(interpolated_maps, axis=0).reshape((N_level,) + cl_pix_bfsf.shape[1:])
f_interp = f_interp / (ms.instrument.nyquist_rate(wl) + 1)
f_interp = np.clip(f_interp, 0, None)
fig, ax = plt.subplots(ncols=2)
I_lsq_eq_interp = s2image.Image(f_interp, cl_pix_bfsf)
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
ax[0].set_title("Critically sampled Bluebild Least-Squares Image")
I_lsq_eq_interp.draw(ax=ax[1])
ax[1].set_title("Interpolated Bluebild Least-Squares Image")
plt.show()
plt.savefig('interp_fig')

# 5. Store the interpolated Bluebild image in standard-compliant FITS for view
# in AstroPy/DS9.
f_interp = (f_interp  # We need to transpose axes due to the FORTRAN
            .reshape(N_level, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
            .transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)

I_lsq_eq_interp.to_fits('toothbrush-image-bb.fits')

