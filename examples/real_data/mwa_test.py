# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Real data (.ms file) script based on lofar_bootes_nufft3 from ci-master

# Doubts: 

#Should nuFFT imager be re-initialised for each image? If not, which wl to feed to the imager?

which wl - add to slides 

#Does sensitivity field imaging and PE correct for Primary beam shape? Or for sensitivity across FoV? 

primary beam correction


# Without 2 pi/wl factor - the MWA image is NOT recreated - checking for LOFAR image - shape mismatch in ms.visibilities thrown during imaging step
# LOFAR ERROR: UVW_baselines[uvw_indices['f0'],uvw_indices['f1']] = uvw
ValueError: shape mismatch: value array of shape (666,3) could not be broadcast to indexing result of shape (276,3)

# when filter data W passed in Sensitivity Imaging divide by zero error (306) multiply error (372)
might not be an issue 
TODO: 
1) parameter estimator uvw check against measurement set uvw reduce tolerance 
  - maybe don't check for cases where baseline is flagged, then use ratio to getr right Order of magntude in image
2) Check for LOFAR image fidelity with new script
3) Install and demonstrate APSYNSIM
4) 
"""
import time as tt
start_time = tt.process_time()

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
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
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import imot_tools.io.plot as implt


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# spk edits

import sys
np.set_printoptions(threshold=sys.maxsize)
import pypeline.phased_array.measurement_set as measurement_set
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.metrics
import imot_tools.math.sphere.grid as grid
import imot_tools.io.fits as ifits
import astropy.io.fits as ap_fits
import pypeline.util.frame as frame
# for final figure text sizes
#plt.rcParams.update({'font.size': 22})

def RX(teta):
    return np.array([[ 1.0,          0.0,           0.0],
                     [ 0.0,  np.cos(teta),  np.sin(teta)],
                     [ 0.0, -np.sin(teta),  np.cos(teta)]])

def RY(teta):
    return np.array([[ np.cos(teta),  0.0, -np.sin(teta)],
                     [          0.0,  1.0,           0.0],
                     [ np.sin(teta),  0.0,  np.cos(teta)]])

def RZ(teta):
    return np.array([[  np.cos(teta),  np.sin(teta), 0.0],
                     [ -np.sin(teta),  np.cos(teta), 0.0],
                     [           0.0,           0.0, 1.0]])

###############################################################
# Path Variables
###############################################################
ms_file = "/work/ska/MWA/1133149192-187-188_Sun_10s_cal.ms/" # MWA
# for testing against lofar ms file
#ms_file = "/home/krishna/bluebild/testData/gauss4_t201806301100_SBL180.MS"
        
WSClean_image_path = "/scratch/izar/krishna/MWA/WSClean/"
WSClean_image_name ="1133149192-187-188_Sun_10s_cal1024_Pixels_4_5_channels-image.fits" # 1024 pixels, 50"/pixel, 5th channel

WSClean_image_path += WSClean_image_name
#WSClean_image_name = "/scratch/izar/krishna/LOFAR/WSClean/gauss4_t201806301100_SBL180.MS_WSClean-image.fits"

output_dir = "/scratch/izar/krishna/Calibration/"
###############################################################
# Control Variables
###############################################################

#Image params
N_pix = 512

# error tolerance for FFT
eps = 1e-3

#precision of calculation
precision = 'single'

# Field of View in degrees - only used when WSClean_grid is false
FoV = np.deg2rad(6)

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
WSClean_grid = False

#ms_fieldcenter: Use field center from MS file if True; only invoked if WSClean_grid is False
ms_fieldcenter = True

#user_fieldcenter: Invoked if WSClean_grid and ms_fieldcenter are False - gives allows custom field center for imaging of specific region
user_fieldcenter = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")

#Time
time_start = 0
time_end = 1
time_slice = 1

# channel
channel_id = int(4)
#channel_id = int(0)


###############################################################
# Observation set up
###############################################################

ms = measurement_set.MwaMeasurementSet(ms_file)
# for testing against lofar.ms file
#ms = measurement_set.LofarMeasurementSet(ms_file, N_station =24)

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
print (f"wl:{wl}; f: {frequency}")

# pixel grid; uses either i/p WSClean image, or ms.field_center & FoV or user.field_center & FoV
if (WSClean_grid): 
    cl_WCS = ifits.wcs(WSClean_image_path)
    cl_WCS = cl_WCS.sub(['celestial'])
    cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))  # downsample, too high res!
    cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame

    px_grid = cl_pix_icrs
    field_center = ms.field_center
else: 
    if (ms_fieldcenter):
        _, _, px_colat, px_lon = grid.equal_angle(N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV)
        field_center = ms.field_center
    else:
        _, _, px_colat, px_lon = grid.equal_angle(N=ms.instrument.nyquist_rate(wl), direction=user_fieldcenter.cartesian.xyz.value, FoV=FoV)
        field_center = user_fieldcenter
    print("nyquist rate", ms.instrument.nyquist_rate(wl), "px_col {0}, px_lon {1}".format(px_colat.shape, px_lon.shape))

    px_grid = transform.pol2cart(1, px_colat, px_lon)

gram = bb_gr.GramBlock()

print (f"Initial set up takes {tt.time() - start_time} s")

###############################################################
# Intensity Field Parameter Estimation
# (Gives clustering i/p)
###############################################################
if (clustering):
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)

    for t, f, S, uvw in ProgressBar(
            ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
    ):
        wl = constants.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)

        """
        uvw_frame = RX(np.pi/2 - field_center.dec.rad) @ RZ(np.pi/2 + field_center.ra.rad)
        #uvw_frame = frame.uvw_basis(field_center)
        #UVW_t = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
        UVW_t = (uvw_frame @ XYZ.data.transpose()).transpose()
        UVW_t =  (UVW_t[:, None, :] - UVW_t[None, ...])
        #"""

        W = ms.beamformer(XYZ, wl)
        G = gram(XYZ, W, wl)
        S, _ = measurement_set.filter_data(S, W)

        I_est.collect(S, G)
    N_eig, c_centroid = I_est.infer_parameters()

else:
    # Set number of eigenvalues to number of eigenimages 
    # and equally divide the data between them 
    N_eig, c_centroid = N_level, np.arange(N_level)


print(f"Clustering: {clustering}")
print (f"Number of Eigenvalues:{N_eig}, Centroids: {c_centroid}")

####################################################################
# Imaging
# Here we feed the inputs from the Intensity Field Parameter Estimation to get images 
# that are consistent with the Eigenvalues and Centroids produced in the previous step.
####################################################################
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'std'))

# should this be different for each block? or a single wl at the 
# lowest resolution?
# ASK EMMA + MATTHEIU
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
for t, f, S, uvw in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)

    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, XYZ, W, wl)

    S_corrected = IV_dp(D, V, W, c_idx)

    ratio = 1
    ratio = np.max(S_corrected) / np.max(S.data)
    #ratio = np.mean(S.data)/ np.mean(S_corrected)

    print(f'max S{np.max(S.data)}, max S_corrected {np.max(S_corrected)}')
    print(f'mean S{np.mean(S.data)}, mean S_corrected {np.mean(S_corrected)}')


    S_corrected *= ratio
    print (f'Visibility multiplication ratio is {ratio}')

    uvw_frame = RX(np.pi/2 - field_center.dec.rad) @ RZ(np.pi/2 + field_center.ra.rad)

    UVW_t = (uvw_frame @ XYZ.data.transpose()).transpose()

    UVW_t =  (UVW_t[:, None, :] - UVW_t[None, ...])

    UVW_t[uvw == 0] = 0

    if (np.any(-uvw - UVW_t>2e-1)):
        print (uvw.shape, UVW_t.shape)
        residual = np.where(np.abs(-uvw-UVW_t) >2e-1, np.abs(-uvw-UVW_t), 0)
        if (np.count_nonzero(residual.ravel) > 0):
            print (np.vstack((np.nonzero(residual)[0], np.nonzero(residual)[1], np.nonzero(residual)[2])).T)
            # first index 0 -128, 42, 51, 125, third index 0-2+ 125 * 3 * 3 (for row 42, 52 and 125)
            # 64 when 42, 51 and 125 antenna no. is flagged

            print (np.count_nonzero(residual), residual.size)
            print (residual[residual != 0])
        raise Exception("UVW Coordinates are inconsistent")


    #uvw baselines passed without 2pi/wl factor!
    #nufft_imager.collect(-1 * uvw  , S_corrected)
    nufft_imager.collect(UVW_t , S_corrected)


# NUFFT Synthesis
lsq_image, std_image = nufft_imager.get_statistic()

###############################################################################
### Sensitivity Field Parameter Estimation
# Determines Eigenvalues and Cluster centroids to feed to sensitivity imaging for 
# correcting for Primary Beam Shape? (ASK EMMA + MATTHIEU)
###############################################################################
if (clustering):
    S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)

    for t, f, S in ProgressBar(ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name)):
        wl = constants.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)
        W = ms.beamformer(XYZ, wl)
        G = gram(XYZ, W, wl)

        S_est.collect(G)


    N_eig = S_est.infer_parameters()
else: 
    N_eig = N_level

# Imaging
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
for t, f, S, uvw in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)

    S, _ = measurement_set.filter_data(S, W)

    D, V = S_dp(XYZ, W, wl)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))

    uvw_frame = RX(np.pi/2 - field_center.dec.rad) @ RZ(np.pi/2 + field_center.ra.rad)

    UVW_t = (uvw_frame @ XYZ.data.transpose()).transpose()
    
    UVW_t =  (UVW_t[:, None, :] - UVW_t[None, ...])

    #uvw baselines passed without 2pi/wl factor!

    #nufft_imager.collect(-1 * uvw , S_sensitivity)
    nufft_imager.collect(UVW_t, S_sensitivity)

sensitivity_image = nufft_imager.get_statistic()[0]
"""
I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')
+
plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')

plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild sqrt, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')

plt.figure()
titles = ['Strong sources', 'Mild sources', 'Faint Sources']
for i in range(lsq_image.shape[0]):
    plt.subplot(1, N_level, i + 1)
    ax = plt.gca()
    plt.title(titles[i])
    I_lsq_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5), show_gridlines=False)


plt.suptitle(f'Bluebild Eigenmaps')

plt.savefig('final.png')
#  plt.show()
"""

print (lsq_image.shape, sensitivity_image.shape,nufft_imager._synthesizer.xyz_grid.shape)

I_lsq_eq = s2image.Image(lsq_image.reshape(lsq_image.shape[-2:]), nufft_imager._synthesizer.xyz_grid)
I_std_eq = s2image.Image(std_image.reshape(std_image.shape[-2:]), nufft_imager._synthesizer.xyz_grid)
t2 = tt.time()

fig, ax = plt.subplots(2, N_level + 2, figsize = (40, 20))

I_std_eq.draw(ax = ax[0,0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True)
ax[0,0].set_title("Std")
I_lsq_eq.draw( ax=ax[1, 0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True)
ax[1, 0].set_title("Lsq")

for i in np.arange(N_level):
    I_std_eq.draw(index=i, ax = ax[0,i+1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True)
    ax[0, i+1].set_title(f"Std lvl{i}")
    I_lsq_eq.draw(index=i, ax = ax[1,i+1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True)
    ax[1, i+1].set_title(f"Lsq lvl{i}")

WSClean_image = ap_fits.getdata(WSClean_image_path)
WSClean_image = WSClean_image.reshape(WSClean_image.shape[-2:])
WSClean_scale= ax[0, -1].imshow((WSClean_image), cmap='cubehelix')
ax[0, -1].set_title("WSClean Image")
divider = make_axes_locatable(ax[0, -1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(WSClean_scale, cax)


plt.savefig("spk_master_max_scaling_wos_op.png")

I_lsq_eq = s2image.Image(lsq_image.reshape(lsq_image.shape[-2:])/ sensitivity_image.reshape(sensitivity_image.shape[-2:]), nufft_imager._synthesizer.xyz_grid)
I_std_eq = s2image.Image(std_image.reshape(std_image.shape[-2:])/ sensitivity_image.reshape(sensitivity_image.shape[-2:]), nufft_imager._synthesizer.xyz_grid)
t2 = tt.time()

fig, ax = plt.subplots(2, N_level + 2, figsize = (40, 20))

I_std_eq.draw(ax = ax[0,0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True)
ax[0,0].set_title("Std")
I_lsq_eq.draw( ax=ax[1, 0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True)
ax[1, 0].set_title("Lsq")

for i in np.arange(N_level):
    I_std_eq.draw(index=i, ax = ax[0,i+1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True)
    ax[0, i+1].set_title(f"Std lvl{i}")
    I_lsq_eq.draw(index=i, ax = ax[1,i+1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True)
    ax[1, i+1].set_title(f"Lsq lvl{i}")

WSClean_image = ap_fits.getdata(WSClean_image_path)
WSClean_image = WSClean_image.reshape(WSClean_image.shape[-2:])
WSClean_scale= ax[0, -1].imshow((WSClean_image), cmap='cubehelix')
ax[0, -1].set_title("WSClean Image")
divider = make_axes_locatable(ax[0, -1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(WSClean_scale, cax)


plt.savefig("spk_master_max_scaling_op.png")
print(f'Elapsed time: {tt.time() - start_time} seconds.')