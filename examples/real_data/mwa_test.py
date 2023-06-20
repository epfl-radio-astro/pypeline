# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Real data (.ms file) script based on lofar_bootes_nufft3 from ci-master

# Doubts: 

#Does sensitivity field imaging and PE correct for Primary beam shape? Or for sensitivity across FoV? 

primary beam correction


# Without 2 pi/wl factor - the MWA image is NOT recreated - checking for LOFAR image - shape mismatch in ms.visibilities thrown during imaging step
# LOFAR ERROR: UVW_baselines[uvw_indices['f0'],uvw_indices['f1']] = uvw
ValueError: shape mismatch: value array of shape (666,3) could not be broadcast to indexing result of shape (276,3)

# when filter data W passed in Sensitivity Imaging divide by zero error (306) multiply error (372)
might not be an issue 
TODO: 

RASCIL FOR WSClean generation
1) Check script and calibration and UVW baseline check for LOFAR 
 - check with old mean code
LOFAR Baseline check shows 2-4 m difference between MS baselines and BB baselines
LOFAR UVW does not need another -ive sign - 1 negative sign in measurement_set method visibilities gives correct-ish coordinates

 - check ratio by varying number of channels included in MWA images
 - simulation check 
 - TCLEAN Benchmarking

2) write code to separate images based on flux bins - test on LOFAR + MWA
3) Read more WL papers
6) Upload code to spk_dev branch or create new branch and upload !?!?\
7) Conference applications : EAS?  (mid Jan?), cargese school (april 9), US Radio school. 

Mock images using bluebild(Robert feldmann)
- diffuse image reconstruction 
- oskar .ms generation and wsclean vs bluebild comparison 
- FIREBOX stuff - detecting Dwarf Galaxies' HI emission - realistic stellar feedback therefore able to make 
predictions regarding dwarf galaxies (low mass end of lambda CDM) 


Solar data using bluebild (need to write add on for flux separation of eigenlevels)
- Calibration (Single Channel and multi channel)
- simulation comparison (v/s wsclean and bluebild)
- psf separation for eigen decomposition 
- separation of coronal features from other features
- Rohit chat separation of coronal features?

Bluebild for imaging in Weak Lensing
 - I.H. Model 
 - 10 SNR vs 10000 SNR sources - separation into different levels using BB 
 - PSF features isolated between images
 - 


8)_
"""
import time as tt
start_time = tt.time()

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
import imot_tools.math.sphere.grid as grid
import imot_tools.io.fits as ifits
import astropy.io.fits as ap_fits
import pypeline.util.frame as frame
from matplotlib.colors import TwoSlopeNorm
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

WSClean_image_path = "/scratch/izar/krishna/MWA/WSClean/"
#WSClean_image_name = "1133149192-187-188_Sun_10s_cal1024_Pixels_0_64_channels-image.fits" # full channel, uniform weighting
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal1024_Pixels_4_5_channels-image.fits" # 4-5 channels, uniform weighting
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal_0_64_channels_weighting_natural-image.fits" # 0-64 channels, natural weighting
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal_0_64_channels_weighting_uniform-image.fits" # 0-64 channels, uniform weighting
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal_0_5_channels_weighting_natural-image.fits" # 0-5 channels, natural weighting
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal_0_5_channels_weighting_uniform-image.fits" # 0-5 channels, uniform weighting
WSClean_image_name ="1133149192-187-188_Sun_10s_cal_4_5_channels_weighting_natural-image.fits" # 4-5 channels, natural weighting
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal_4_5_channels_weighting_uniform-image.fits" # 4-5 channels, uniform weighting

WSClean_image_path += WSClean_image_name

output_dir = "/scratch/izar/krishna/Calibration/"
###############################################################
# Control Variables
###############################################################

#Image params
N_pix = 1024

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
WSClean_grid = True

#ms_fieldcenter: Use field center from MS file if True; only invoked if WSClean_grid is False
ms_fieldcenter = True

#user_fieldcenter: Invoked if WSClean_grid and ms_fieldcenter are False - gives allows custom field center for imaging of specific region
user_fieldcenter = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")

#Time
time_start = 0
time_end = -1
time_slice = 1

# channel
channel_id = np.array([4], dtype = np.int32)
#channel_id = np.arange(64, dtype = np.int)

# list of filters for imaging
filter_tuple = ('lsq', 'std',)

# Sensitivity_boolean
sensitivity_imaging = False

###############################################################
# Observation set up
###############################################################

ms = measurement_set.MwaMeasurementSet(ms_file)

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

wl = constants.speed_of_light / frequency.to_value(u.Hz) [0]
print (f"wl:{wl}; f: {frequency}")

# pixel grid; uses either i/p WSClean image, or ms.field_center & FoV or user.field_center & FoV
if (WSClean_grid): 
    cl_WCS = ifits.wcs(WSClean_image_path)
    cl_WCS = cl_WCS.sub(['celestial'])
    cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))  # downsample, too high res!
    cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame

    px_grid = cl_pix_icrs
    field_center = ms.field_center

    width_px, height_px= 2*cl_WCS.wcs.crpix 
    cdelt_x, cdelt_y = cl_WCS.wcs.cdelt 
    FoV = np.deg2rad(abs(cdelt_x*width_px) )
    print ("WSClean Grid used.")
else: 
    if (ms_fieldcenter):
        _, _, px_colat, px_lon = grid.equal_angle(N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV)
        field_center = ms.field_center
        print ("Self generated grid used based on ms fieldcenter")
    else:
        _, _, px_colat, px_lon = grid.equal_angle(N=ms.instrument.nyquist_rate(wl), direction=user_fieldcenter.cartesian.xyz.value, FoV=FoV)
        field_center = user_fieldcenter
        print ("Self generated grid used based on user fieldcenter")
    print("nyquist rate", ms.instrument.nyquist_rate(wl), "px_col {0}, px_lon {1}".format(px_colat.shape, px_lon.shape))

    px_grid = transform.pol2cart(1, px_colat, px_lon)

gram = bb_gr.GramBlock()

print (f"Initial set up takes {tt.time() - start_time} s")

###############################################################
# Intensity Field Parameter Estimation
# (Gives clustering i/p)
###############################################################
if (clustering):
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=1)

    for t, f, S, uvw in ProgressBar(
            ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
    ):
        wl = constants.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)

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
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=filter_tuple)

# should this be different for each block? or a single wl at the 
# lowest resolution?
# ASK EMMA + MATTHEIU
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
vis_count = 0
bb_vis_count = 0

for t, f, S, uvw in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)

    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, XYZ, W, wl)

    S_corrected = IV_dp(D, V, W, c_idx)

    uvw_frame = RX(np.pi/2 - field_center.dec.rad) @ RZ(np.pi/2 + field_center.ra.rad)

    UVW_t = (uvw_frame @ XYZ.data.transpose()).transpose()

    UVW_t =  (UVW_t[:, None, :] - UVW_t[None, ...])

    UVW_t[uvw == 0] = 0

    vis_count += np.count_nonzero(S.data)
    bb_vis_count += np.count_nonzero(S_corrected[0, :, :, :])

    if (np.any(-uvw - UVW_t>2e-1)):
        print (uvw.shape, UVW_t.shape)
        residual = np.where(np.abs(-uvw-UVW_t) >2e-1, np.abs(-uvw-UVW_t), 0)
        if (np.count_nonzero(residual.ravel) > 0):
            print (np.vstack((np.nonzero(residual)[0], np.nonzero(residual)[1], np.nonzero(residual)[2])).T)

            print (np.count_nonzero(residual), residual.size)
            print (residual[residual != 0])
        raise Exception("UVW Coordinates are inconsistent")
    
    diagonal_indices = np.diag_indices_from(S_corrected[0, 0, :, :].reshape(S_corrected.shape[-2], S_corrected.shape[-1])) # Take indices of auto correlations

    #set autocorrelations to 0 for each level and each filter
    for level in np.arange(N_level): 
        for filter_no, filters in enumerate(filter_tuple):
            S_corrected_filter_level = S_corrected[filter_no, level, :, :].reshape(S_corrected.shape[-2], S_corrected.shape[-1])
            S_corrected_filter_level[diagonal_indices] = 0
            S_corrected[filter_no, level, :, :] = S_corrected_filter_level.reshape(1, 1, S_corrected.shape[-2], S_corrected.shape[-1])

    nufft_imager.collect(UVW_t, S_corrected)

# NUFFT Synthesis
lsq_image, std_image = nufft_imager.get_statistic()
lsq_image = lsq_image/bb_vis_count
std_image = std_image/bb_vis_count 
I_lsq_eq = s2image.Image(lsq_image.reshape(int(N_level), lsq_image.shape[-2], lsq_image.shape[-1]), nufft_imager._synthesizer.xyz_grid)
I_std_eq = s2image.Image(std_image.reshape(int(N_level), std_image.shape[-2], std_image.shape[-1]), nufft_imager._synthesizer.xyz_grid)

if (sensitivity_imaging):

    ###############################################################################
    ### Sensitivity Field Parameter Estimation
    # Determines Eigenvalues and Cluster centroids to feed to sensitivity imaging for 
    # correcting for Primary Beam Shape? (ASK EMMA + MATTHIEU)
    ###############################################################################
    if (clustering):
        S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=1)

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

        #nufft_imager.collect(-1 * uvw , S_sensitivity)
        nufft_imager.collect(UVW_t, S_sensitivity)

    sensitivity_image = nufft_imager.get_statistic()[0]

    I_lsq_eq = s2image.Image((lsq_image/sensitivity_image).reshape(int(N_level), lsq_image.shape[-2], lsq_image.shape[-1]), nufft_imager._synthesizer.xyz_grid)
    I_std_eq = s2image.Image((std_image/sensitivity_image).reshape(int(N_level), std_image.shape[-2], std_image.shape[-1]), nufft_imager._synthesizer.xyz_grid)

    print (lsq_image.shape, sensitivity_image.shape,nufft_imager._synthesizer.xyz_grid.shape)

# Without sensitivity imaging output
# For each filter - total, each level, wsclean, (wsclean comparison diff, wsclean comparison ratio)
fig, ax = plt.subplots(int(len(filter_tuple)), N_level + 3, figsize = (40, 20))

lsq_levels = I_lsq_eq.data # Nlevel, Npix, Npix
std_levels = I_std_eq.data # Nlevel, Npix, Npix

lsq_image = lsq_levels.sum(axis = 0)
std_image = std_levels.sum(axis = 0)

BBScale = ax[0, 0].imshow(lsq_image, cmap = "RdBu_r")
ax[0, 0].set_title(r"LSQ IMG")
divider = make_axes_locatable(ax[0, 0])
cax = divider.append_axes("right", size = "5%", pad = 0.05)
cbar = plt.colorbar(BBScale, cax)

#plot WSClean image
WSClean_image = ap_fits.getdata(WSClean_image_path)
WSClean_image = np.flipud(WSClean_image.reshape(WSClean_image.shape[-2:]))
WSCleanScale = ax[0, N_level + 1].imshow(WSClean_image, cmap='RdBu_r')
ax[0, N_level+1].set_title(f"WSC IMG")
divider = make_axes_locatable(ax[0, N_level+1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(WSCleanScale, cax)

BBScale = ax[1, 0].imshow(std_image, cmap = "RdBu_r")
ax[1, 0].set_title(r"STD IMG")
divider = make_axes_locatable(ax[1, 0])
cax = divider.append_axes("right", size = "5%", pad = 0.05)
cbar = plt.colorbar(BBScale, cax)

WSCleanScale = ax[1, N_level + 1].imshow(WSClean_image, cmap='RdBu_r')
ax[1, N_level+1].set_title(f"WSC IMG")
divider = make_axes_locatable(ax[1, N_level+1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(WSCleanScale, cax)

for i in np.arange(1,N_level + 1):
    lsqScale = ax[0, i].imshow(lsq_levels[i-1, :, :], cmap = 'RdBu_r')
    ax[0, i].set_title(f"Lsq Lvl {i}")
    divider = make_axes_locatable(ax[0, i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(lsqScale, cax)

    stdScale = ax[1, i].imshow(std_levels[i-1, :, :], cmap = 'RdBu_r')
    ax[1, i].set_title(f"Std Lvl {i}")
    divider = make_axes_locatable(ax[1, i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(stdScale, cax)

diff_image = lsq_image - WSClean_image
diff_norm = TwoSlopeNorm(vmin=diff_image.min(), vcenter=0, vmax=diff_image.max())

diffScale = ax[0, -1].imshow(diff_image, cmap = 'RdBu_r', norm=diff_norm)
ax[0, -1].set_title("Diff IMG")
divider = make_axes_locatable(ax[0, -1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(diffScale, cax)

ratio_image = lsq_image/WSClean_image
ratio_image = np.clip(ratio_image, -2.5, 2.5)
ratio_norm = TwoSlopeNorm(vmin=ratio_image.min(), vcenter=1, vmax=ratio_image.max())

ratioScale = ax[1, -1].imshow(ratio_image, cmap = 'RdBu_r', norm=ratio_norm)
ax[1, -1].set_title("Ratio IMG")
divider = make_axes_locatable(ax[1, -1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(ratioScale, cax)

plt.savefig("spk_mwa_op.png")

print(f'Elapsed time: {tt.time() - start_time} seconds.')