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

#cl_WCS = ifits.wcs("/home/etolley/rascil_ska_sim/results_test/imaging_dirty.fits")
#cl_WCS = cl_WCS.sub(['celestial'])
##cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
#cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
#N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

# Instrument
cl_WCS = ifits.wcs("/work/ska/MWA_1086366992/wsclean-mwa-chan100-dirty.fits")
ms_file = "/work/ska/MWA_1086366992/1086366992.ms"
ms = measurement_set.MwaMeasurementSet(ms_file) # stations 1 - N_station 
out_str = "mwa_galaxy_test"

#cl_WCS = ifits.wcs("/work/ska/gauss4/gauss4-image-pb.fits")
#ms_file = '/work/ska/gauss4/gauss4_t201806301100_SBL180.MS'
#ms = measurement_set.LofarMeasurementSet(ms_file) 



gram = bb_gr.GramBlock()
print(cl_WCS.to_header())
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10))) # downsample!
cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

print("Reading {0}\n".format(ms_file))

# Observation
channel_id = 100
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
#sky_model = source.from_tgss_catalog(ms.field_center, FoV, N_src=4)
obs_start, obs_end = ms.time["TIME"][[0, -1]]
field_center = ms.field_center
FoV = np.deg2rad(20)

print("obs start: {0}, end: {1}".format(obs_start, obs_end))
print(ms.time["TIME"])

# Imaging

# Imaging
N_pix = 512
eps = 1e-5
w_term = True
N_level = 4
precision = 'single'
time_slice = slice(0, 10, 1)


### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        ms.visibilities(
            channel_id=[channel_id], time_id=slice(0, None, 10), column="DATA"
        )
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    print(G)

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
    UVW_baselines_t = ms.instrument.baselines(t)

    
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)

    try:
        
        S_corrected = IV_dp(D, V, W, c_idx)
        gram_corrected_visibilities.append(S_corrected)
        UVW_baselines.append(UVW_baselines_t)
        print(UVW_baselines_t.shape, S_corrected.shape)
    except:
        continue


print(UVW_baselines[0].shape, len(UVW_baselines))
UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
print(UVW_baselines.shape)
print(gram_corrected_visibilities[0].shape, len(gram_corrected_visibilities))
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3).reshape(*S_corrected.shape[:2], -1)
print(gram_corrected_visibilities.shape)
print(UVW_baselines.shape, gram_corrected_visibilities.shape)

# NUFFT Synthesis
print("Running NUFFT on the CPU")
t = time.process_time()
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
#print(nufft_imager._synthesizer._inner_fft_sizes)
lsq_image, sqrt_image = nufft_imager(gram_corrected_visibilities)

end_time = time.process_time()
print("Time elapsed: {0}s".format(end_time - start_time))

### Sensitivity Field =========================================================
# Parameter Estimation
'''S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::10]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

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
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
sensitivity_image = nufft_imager(sensitivity_coeffs)
print("time elapsed: {0}".format(time.process_time() - t))'''

# Plot Results ================================================================
fig, ax    = plt.subplots(ncols=N_level, nrows=2, figsize=(16, 10))
#I_lsq_eq   = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
#I_std_eq   = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_lsq_eq   = s2image.Image(lsq_image, nufft_imager._synthesizer.xyz_grid)
I_std_eq   = s2image.Image(sqrt_image, nufft_imager._synthesizer.xyz_grid)

for i in range(N_level):
    I_std_eq.draw(index=i, ax=ax[0,i])
    ax[0,i].set_title("Standardized Image Level = {0}".format(i))
    I_lsq_eq.draw(index=i, ax=ax[1,i])
    ax[1,i].set_title("Least-Squares Image Level = {0}".format(i))
#fig.show()
#plt.show()
#sys.exit()
plt.savefig("skalow_nufft_new")

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