# #############################################################################
# lofar_nufft.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com] (modified by Michele)
# Simulated LOFAR imaging with Bluebild (NUFFT).
# #############################################################################

import bluebild_tools.cupy_util as bbt_cupy
use_cupy = bbt_cupy.is_cupy_usable()

from tqdm import tqdm as ProgressBar
import astropy.units as u
from imot_tools.io import fits as ifits, s2image
import numpy as np
import scipy.constants as constants

from pypeline.phased_array.bluebild import data_processor as bb_dp, gram as bb_gr
from pypeline.phased_array.bluebild.imager import spatial_domain as bb_sd, fourier_domain as bb_im
from timing import Timer

from pypeline.phased_array import measurement_set

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=AstropyWarning)

t = Timer()

time_slice = 100
N_station = 24
N_level = 4

fname_prefix = 'lofar30MHz1'
path_out = './'
path_in = '/project/c31/%s/' %fname_prefix
fname = '%s_t201806301100_SBL153.MS' %(path_in+fname_prefix)
data_column="MODEL_DATA"

t.start_time("Set up data")
# Measurement Set
ms = measurement_set.LofarMeasurementSet(fname, N_station)
channel_id = 1
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)

# Observation
FoV = np.deg2rad((2000*2.*u.arcsec).to(u.deg).value)
field_center = ms.field_center
time = ms.time['TIME'][:time_slice]

# Instrument
gram = bb_gr.GramBlock()

# Imaging
eps = 1e-3
precision = 'single'
N_bits = 32

### NUFFT imaging parameters ===========================================================
cl_WCS = ifits.wcs('%s%s-psf.fits' %(path_in, fname_prefix))
cl_WCS = cl_WCS.sub(['celestial']) 
#cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
px_grid = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
N_cl_lon, N_cl_lat = px_grid.shape[-2:]
assert N_cl_lon == N_cl_lat
N_pix = N_cl_lon

t.end_time("Set up data")
print('''You are running bluebild on file: %s
         with the following input parameters:
         %d timesteps
         %d stations
         clustering into %d levels
         The output grid will be %dx%d = %d pixels''' %(fname, len(time), N_station, N_level, px_grid.shape[1],  px_grid.shape[2],  px_grid.shape[1]* px_grid.shape[2]))

### Intensity Field =================================================
# Parameter Estimation

t.start_time("Estimate intensity field parameters")
"""
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for i_t, ti in enumerate(ProgressBar(time)):
    tobs, f, S = next(data.ms.visibilities(channel_id=[data.channel_id], time_id=slice(i_t, i_t+1, None), column=data_column))
    wl = constants.speed_of_light / f.to_value(u.Hz) #self.wl
    XYZ = ms.instrument(tobs)
    W = ms.beamformer(XYZ, wl)
    S, _ = measurement_set.filter_data(S, W)
    
    G = gram(XYZ, W, wl)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
print(N_eig, c_centroid)
"""
N_eig, c_centroid = N_level, list(range(N_level))        # bypass centroids
t.end_time("Estimate intensity field parameters")

####################################################################
#### Imaging
####################################################################
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq','sqrt'))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)

I_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)

for i_t, ti in enumerate(ProgressBar(time)):
    t.start_time("Synthesis: prep input matrices & fPCA")
    tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column=data_column))
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(tobs)
    W = ms.beamformer(XYZ, wl)
    S, _ = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, XYZ, W, wl)
    c_idx = list(range(N_level))        # bypass c_idx
    t.end_time("Synthesis: prep input matrices & fPCA")
    
    t.start_time("NUFFT Synthesis")
    UVW_baselines_t = ms.instrument.baselines(ti, uvw=True, field_center=field_center)
    S_corrected = IV_dp(D, V, W, c_idx)
    nufft_imager.collect(UVW_baselines_t, S_corrected)
    t.end_time("NUFFT Synthesis")

# NUFFT Synthesis
lsq_image, sqrt_image = nufft_imager.get_statistic()

### Sensitivity Field =========================================================
t.start_time("Estimate sensitivity field parameters")
"""
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for ti in ProgressBar(time):
    XYZ = ms.instrument(ti)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig, c_centroid = I_est.infer_parameters()
"""
N_eig, c_centroid = N_level, list(range(N_level))
t.end_time("Estimate sensitivity field parameters")

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)

for i_t, ti in enumerate(ProgressBar(time)):
    tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column=data_column))
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(tobs)

    W = ms.beamformer(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)

    UVW_baselines_t = ms.instrument.baselines(ti, uvw=True, field_center=field_center)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))  # (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)

sensitivity_image = nufft_imager.get_statistic()[0]
np.save('%sD_%s' %(path_out, fname_prefix), D.reshape(-1, 1, 1))

#I_sqrt_eq_nufft = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_lsq_eq_nufft = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)

# Save eigen-vectors for NUFFT
#np.save('%sI_lsq_eq_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_nufft.data)
np.save('%sI_nufft_%s' %(path_out, fname_prefix), I_lsq_eq_nufft.data)

# Interpolate image to MS grid-frame for NUFFT
f_interp = (I_lsq_eq_nufft.data.reshape(N_level, N_cl_lon, N_cl_lat).transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('%sI_nufft_%s.fits' %(path_out, fname_prefix))
