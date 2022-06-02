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
import astropy.coordinates as coord
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
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as bb_synth
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.math.sphere.transform as transform
import pycsou.linop as pyclop
from imot_tools.math.func import SphericalDirichlet
import joblib as job

start_time = time.process_time()

read_coords_from_ms = True

# Instrument
#ms_file = "/home/etolley/rascil_ska_sim/results_test/ska-pipeline_simulation.ms"
ms_file = "/work/ska/results_rascil_skalow_small/ska-pipeline_simulation.ms"
ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
gram = bb_gr.GramBlock()


if read_coords_from_ms:
    cl_WCS = ifits.wcs("/work/ska/results_rascil_skalow_small/wsclean-image.fits")
    cl_WCS = cl_WCS.sub(['celestial'])
    ##cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
    cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
    N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

    width_px, height_px= 2*cl_WCS.wcs.crpix 
    cdelt_x, cdelt_y = cl_WCS.wcs.cdelt 
    FoV = np.deg2rad(abs(cdelt_x*width_px) )
    field_center = ms.field_center
else:
    field_center = coord.SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    #field_center = coord.SkyCoord(ra=90.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    FoV = np.deg2rad(5.5)
    

print("Reading {0}\n".format(ms_file))

channel_id = 0
frequency = 1e8
wl = constants.speed_of_light / frequency
freq_ms = ms.channels["FREQUENCY"][channel_id]
assert freq_ms.to_value(u.Hz) == frequency
obs_start, obs_end = ms.time["TIME"][[0, -1]]
print("obs start: {0}, end: {1}".format(obs_start, obs_end))

# Imaging
N_pix = 512
eps = 1e-5
w_term = True
N_level = 1
precision = 'single'
time_slice = slice(None, None, 10)


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
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))
UVW_baselines = []
gram_corrected_visibilities = []
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=time_slice, column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    #UVW_baselines_t = ms.baselines(t, uvw=True)
    UVW_baselines_t = ms.instrument.baselines(t, field_center = ms.field_center)
    print('baselines shape:',UVW_baselines_t.shape)
    plt.scatter(UVW_baselines_t[:,:,0], UVW_baselines_t[:,:,1])
    plt.savefig("skalow_nufft_new_baselinesUV")
    UVW_baselines.append(UVW_baselines_t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)
    plt.clf()
    plt.imshow(np.absolute(S.data))
    plt.savefig("skalow_nufft_new_visM")
    plt.imshow(np.angle(S.data))
    plt.savefig("skalow_nufft_new_visphi")
    plt.clf()


    D, V, c_idx = I_dp(S, G)
    print(V.shape, D.shape, len(c_idx))
    #print(c_idx)
    S_corrected = IV_dp(D, V, W, c_idx)
    gram_corrected_visibilities.append(S_corrected)

UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3).reshape(*S_corrected.shape[:2], -1)

# NUFFT Synthesis
print("Running NUFFT on the CPU")
t = time.process_time()


if read_coords_from_ms:
    nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                          field_center=field_center, eps=eps, w_term=w_term,
                                          n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
else:
    nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T,  grid_size=N_pix, FoV=FoV,
                                          field_center=field_center, eps=eps, w_term=w_term,
                                          n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
#print(nufft_imager._synthesizer._inner_fft_sizes)
lsq_image, sqrt_image = nufft_imager(gram_corrected_visibilities)

end_time = time.process_time()
print("Time elapsed: {0}s".format(end_time - start_time))

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::200]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()
'''
print("Running sensitivity imaging")
# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=time_slice, column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(G)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    sensitivity_coeffs.append(S_sensitivity)

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
print("Running NUFFT on the CPU")
t = time.process_time()
if read_coords_from_ms:
    nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                          field_center=field_center, eps=eps, w_term=w_term,
                                          n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
else:
    nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T,  grid_size=N_pix, FoV=FoV,
                                          field_center=field_center, eps=eps, w_term=w_term,
                                          n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
sensitivity_image = nufft_imager(sensitivity_coeffs)
print("time elapsed: {0}".format(time.process_time() - t))'''

# Plot Results ================================================================
fig, ax    = plt.subplots(ncols=N_level, nrows=2, figsize=(16, 10))
I_lsq_eq   = s2image.Image(lsq_image, nufft_imager._synthesizer.xyz_grid)
I_std_eq   = s2image.Image(sqrt_image, nufft_imager._synthesizer.xyz_grid)
print(lsq_image.shape)
for i in range(N_level):
    top_plot = ax[0] if N_level == 1 else ax[0,i]
    bottom_plot = ax[1] if N_level == 1 else ax[1,i]
    I_std_eq.draw(index=i, ax= top_plot)
    top_plot.set_title("Standardized Image Level = {0}".format(i))
    I_lsq_eq.draw(index=i, ax=bottom_plot)
    bottom_plot.set_title("Least-Squares Image Level = {0}".format(i))
plt.savefig("skalow_nufft_mscoords{0}".format(read_coords_from_ms))

sys.exit()
if read_coords_from_ms:
    # 5. Store the interpolated Bluebild image in standard-compliant FITS for view
    # in AstroPy/DS9.
    N_cl_lon, N_cl_lat = nufft_imager._synthesizer.xyz_grid.shape[-2:]
    f_interp = (I_lsq_eq.data  # We need to transpose axes due to the FORTRAN
                .reshape(N_level, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
                .transpose(0, 2, 1))
    I_lsq_eq_interp = s2image.WCSImage(np.sum(f_interp,axis=0), cl_WCS)
    I_lsq_eq_interp.to_fits('bluebild_nufft3_skalow_combined-test.fits')
    I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
    I_lsq_eq_interp.to_fits('bluebild_nufft3_skalow_levels-test.fits')