# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
TODO:
Compare I/p to IV_dp and SV_dp 

"""

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import finufft
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as bb_synth
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.measurement_set as measurement_set
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import imot_tools.io.plot as implt
import time as tt

#spk edits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.metrics
import imot_tools.math.sphere.grid as grid
import imot_tools.io.fits as ifits

# for final figure text sizes
plt.rcParams.update({'font.size': 22})

start_time = tt.process_time()

###############################################################
# Path Variables
ms_file = ms_file = "/work/ska/MWA/1133149192-187-188_Sun_10s_cal.ms/" # MWA
        
WSClean_image_path = "/scratch/izar/krishna/MWA/WSClean/"
WSClean_image_name ="1133149192-187-188_Sun_10s_cal1024_Pixels_4_5_channels-image.fits" # 1024 pixels, 50"/pixel, 5th channel

WSClean_image_path += WSClean_image_name

output_dir = "/scratch/izar/krishna/MWA/"
###############################################################
## Control Variables

# Field of View in degrees
FOV = np.deg2rad(5)

#Number of levels in output image
N_level = 1

#clustering: If true will cluster log(eigenvalues) based on KMeans
clustering = True

# IF USING WSCLEAN IMAGE GRID: sampling wrt WSClean grid
# 1 means the output will have same number of pixels as WSClean image
# N means the output will have WSClean Image/N pixels
sampling = 1

# Column Name: Column in MS file to be imaged (DATA is usually uncalibrated, CORRECTED_DATA is calibration and MODEL_DATA contains WSClean model output)
column_name = "DATA"

# WSClean Grid: Use Coordinate grid from WSClean image if True
WSClean_grid = True

#ms_fieldcenter: Use field center from MS file if True; only invoked if WSClean_grid is False
ms_fieldcenter = False

#user_fieldcenter: Invoked if WSClean_grid and ms_fieldcenter are False - gives allows custom field center for imaging of specific region
user_fieldcenter = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")

#Time
time_start = 0
time_end = None
time_slice = 100

# channel
channel_id = 4


###############################################################
###############################################################
# Observation

ms = measurement_set.MwaMeasurementSet(ms_file)

gram = bb_gr.GramBlock()

# Observation
try:
    if (channel_id.shape[0] > 1):
        frequency = ms.channels["FREQUENCY"][0] + (ms.channels["FREQUENCY"][-1] - ms.channels["FREQUENCY"][0])/2
        print ("Multi-channel mode with ", channel_id.shape[0], "channels.")
    else: 
        frequency = ms.channels["FREQUENCY"][channel_id]
        print ("Single channel mode.")
except:
    frequency = ms.channels["FREQUENCY"][channel_id]
    print ("Single channel mode.")

wl = constants.speed_of_light / frequency.to_value(u.Hz)

N_bits = 32
field_center = ms.field_center
if (WSClean_grid): 
    cl_WCS = ifits.wcs(WSClean_image_path)
    cl_WCS = cl_WCS.sub(['celestial'])
    cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))  # downsample, too high res!
    cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame

    px_grid = cl_pix_icrs
else: 
    if (ms_fieldcenter):
        _, _, px_colat, px_lon = grid.equal_angle(N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FOV)
    else:
        _, _, px_colat, px_lon = grid.equal_angle(N=ms.instrument.nyquist_rate(wl), direction=user_fieldcenter.cartesian.xyz.value, FoV=FOV)
        field_center = user_fieldcenter
    print("nyquist rate", ms.instrument.nyquist_rate(wl), "px_col {0}, px_lon {1}".format(px_colat.shape, px_lon.shape))

    px_grid = transform.pol2cart(1, px_colat, px_lon)

print("Grid size is:", px_grid.shape[1], px_grid.shape[2])

# Imaging
N_pix = px_grid.shape[1]
eps = 1e-3
w_term = True
precision = 'single'

t1 = tt.time()

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name)
):
    wl = constants.speed_of_light /f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ,wl)
    G = gram(XYZ, W, wl)
    S,_ = measurement_set.filter_data(S,W)

    I_est.collect(S, G)

if (clustering):
    N_eig, c_centroid = I_est.infer_parameters()
else:
    N_eig, c_centroid = N_level ,np.arange(N_level) # Each level gets one eigenvalue?

print("N_eig:", N_eig)
print ("centroids = ", c_centroid)

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'std'))
UVW_baselines = []
gram_corrected_visibilities = []

for t, f, S, UVW_baselines_t in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW=True)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    UVW_baselines.append(UVW_baselines_t) # 128, 128, 3; N_antennas, N_antennas , 3
    XYZ = ms.instrument(t)
    
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    S_corrected = IV_dp(D, V, W, c_idx) #2, 1, 128, 128 ; N_filters, N_level, N_antennas, N_antennas
    gram_corrected_visibilities.append(S_corrected)

UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
print(f"UVW baselines shape after: {UVW_baselines.shape[0]},{UVW_baselines.shape[1]}") #    3, N_uvw
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3).reshape(*S_corrected.shape[:2], -1)
print (f"Gram corrected visibilities shape after: {gram_corrected_visibilities.shape[0]},{gram_corrected_visibilities.shape[1]},{gram_corrected_visibilities.shape[2]}") # (M, N_uvw)

# NUFFT Synthesis
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FOV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
lsq_image, std_image = nufft_imager(gram_corrected_visibilities)

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][time_start:time_end:time_slice]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)

if (clustering):
    N_eig = S_est.infer_parameters()
else: 
    N_eig = N_level

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []


for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(G)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    sensitivity_coeffs.append(S_sensitivity) #1, 1, 128, 128 ; N_filters, N_level, N_antennas, N_antennas

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=-3).reshape(*S_sensitivity.shape[:2],-1)

nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FOV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
print(nufft_imager._synthesizer._inner_fft_sizes)
sensitivity_image = nufft_imager(sensitivity_coeffs)

print (lsq_image.shape, sensitivity_image.shape,nufft_imager._synthesizer.xyz_grid)
print ((lsq_image/sensitivity_image).shape)

I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_std_eq = s2image.Image(std_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')