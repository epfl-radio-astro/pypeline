# #############################################################################
# realData_test.py
# ======================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Real-data LOFAR imaging with Bluebild (PeriodicSynthesis).
Compare Bluebild image with WSCLEAN image.
"""

from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
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
import scipy.io


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
import skimage.metrics
# for final figure text sizes
plt.rcParams.update({'font.size': 22})

start_time = time.process_time()

# example run:
# python realData_test.py <mode> <output_name> <channel_start> <channel_end> <ms_path> <WSClean_path>

# ALL INPUTS 

custom_output_name = 0
direction_correction =  0 # 0 for mwa obs - 1 for mwa sim - WHY???
clustering = 1
normalization = "none"
WSClean_grid= True
ms_fieldcenter = True
# Set WSClean_levels to True to plot the WSClean image alongside the eigenlevels image
WSClean_levels = False

fidelity_calculation = 0
structural_similarity_calculation = 0

# only used if both fields above are false: 
FoV = np.deg2rad(2) # degrees - Solar sim ~ 14.4 - Solar Obs
# Solar RA Dec
user_fieldcenter = coord.SkyCoord(ra = 248.7713797* u.deg, dec = -22.0545530* u.deg, frame = 'icrs')# true coordinate but BB images flipped
user_fieldcenter = coord.SkyCoord(ra = 251.6385842 * u.deg, dec = -17.3889944 * u.deg, frame = 'icrs') # false coordinate, but until fix is enacted this will image sun not true coordinate
#FoV = np.deg2rad(14.4) # for full solar image


if (len(sys.argv)> 1):
    mode = sys.argv[1]
    object_name = sys.argv[2]

    if ((mode != 'meerkat') & (mode != 'lofar') & (mode != 'mwa')):
        raise Exception("The mode has to be meerkat, or lofar or mwa !!")
        
    

    elif (mode == "meerkat"):
        instrument_name = "MeerKAT"

        #object_name = "J0159"
        
        ms_file = "/work/ska/MeerKAT/1569274256_sdp_l0_wtspec_J0159.0-3413.ms" # MEERKAT
        
        WSClean_image_path = "/scratch/izar/krishna/MeerKAT/WSClean/J0159.500_Pixels_0_1024_channels-image.fits" # all channels
        #WSClean_image_path = "/scratch/izar/krishna/MeerKAT/WSClean/J0159.0500_Pixels_517_518_channels-image.fits"
        
        column_name = "CORRECTED_DATA" # CORRECTED_DATA is fully flagged DATA is also fully flagged MODEL_DATA is empty
        sampling = 2
        N_level = 1
        output_dir = "/scratch/izar/krishna/MeerKAT/"
        time_slice = 200
        time_start_index = 4
        time_end_index = None

        channel_id = np.arange(1024)

        if (len(sys.argv) >=4):
            channel_start, channel_end = int(sys.argv[3]), int(sys.argv[3]) + 1
            if (len(sys.argv) >=5):
                channel_end = int(sys.argv[4])
                if (len(sys.argv)>=6):
                    ms_file = sys.argv[5]

            channel_id = np.arange(channel_start, channel_end)
            if (len(sys.argv) >= 7):
                WSClean_image_path = sys.argv[6]
            elif (channel_id.shape[0] == 1024):
                WSClean_image_path = "/scratch/izar/krishna/MeerKAT/WSClean/J0159.500_Pixels_0_1024_channels-image.fits" # ALL CHANNELS
            else:
                WSClean_image_path = f"/scratch/izar/krishna/MeerKAT/WSClean/J0159.0500_Pixels_{channel_start}_{channel_end}_channels-image.fits"

        ms = measurement_set.MwaMeasurementSet(ms_file)



    elif (mode == "lofar"):
        instrument_name = "LOFAR"

        #object_name = "4gauss"

        ms_file = "/home/krishna/bluebild/testData/gauss4_t201806301100_SBL180.MS" 
        
        WSClean_image_path = "/home/krishna/bluebild/testData/gauss4-image-pb.fits"
        
        column_name = "DATA"
        sampling = 1
        N_level = 4
        output_dir = "/scratch/izar/krishna/LOFAR/"
        time_slice = 100
        time_start_index = 0
        time_end_index = None

        channel_id = 0
        ms = measurement_set.LofarMeasurementSet(ms_file, N_station = 24)

    elif (mode == "mwa"):
        instrument_name = "MWA"

        #"""
        # This code block is for imaging the MWA observation

        ms_file = "/work/ska/MWA/1133149192-187-188_Sun_10s_cal.ms/" # MWA
        
        WSClean_image_path = "/scratch/izar/krishna/MWA/WSClean/" # only channel 4 image
        
        #WSClean_image_name ="1133149192-187-188_Sun_10s_cal1024_Pixels_0_64_channels-image.fits" # 1024 pixels, 50"/pixel, 64 channels
        WSClean_image_name ="1133149192-187-188_Sun_10s_cal1024_Pixels_4_5_channels-image.fits" # 1024 pixels, 50"/pixel, 4th channel
        #WSClean_image_name ="1133149192-187-188_Sun_10s_cal2048_Pixels_0_64_channels-image.fits" # 2048 pixels, 25"/pixel, 64 channel
        #WSClean_image_name ="1133149192-187-188_Sun_10s_cal2048_Pixels_4_5_channels-image.fits" # 2048 pixels, 25"/pixel, 4th channel
        
        WSClean_image_path += WSClean_image_name

        column_name = "DATA"
        sampling = 1
        N_level = 3
        output_dir = "/scratch/izar/krishna/MWA/"
        time_slice = 100
        time_start_index = 0
        time_end_index = None

        #channel_id = np.arange(0, 64, dtype = np.int)
        channel_id = int(4)

        """
        # This code block is for imaging the MWA simulation
        
        ms_file = "/scratch/izar/krishna/MWA/simulation/simulation_MWA_Obsparams.ms/" # MWA
        #ms_file = "/scratch/izar/krishna/MWA/simulation/simulation_MWA_Obsparams_thresholded.ms/" # MWA
                
        WSClean_image_path = "/scratch/izar/krishna/MWA/simulation/" # only channel 4 image
        
        WSClean_image_name = "simulation_MWA_Obsparams.ms_WSClean-image.fits"
        #WSClean_image_name = "simulation_MWA_Obsparams_thresholded.ms_WSClean-image.fits"
        
        WSClean_image_path += WSClean_image_name

        column_name = "DATA"
        sampling = 1
        N_level = 3
        output_dir = "/scratch/izar/krishna/MWA/simulation/"
        time_slice = 100

        channel_id = int(0)
        #"""

        if (len(sys.argv) >=4):
            channel_start, channel_end = int(sys.argv[3]), int(sys.argv[3]) +1
            if (len(sys.argv) >=5):
                channel_end = int(sys.argv[4])
                if (len(sys.argv)>=6):
                    ms_file = sys.argv[5]

            channel_id = np.arange(channel_start, channel_end)
            if (len(sys.argv) >= 7):
                WSClean_image_path = sys.argv[6]
            else:
                if (channel_id.shape[0] == 64):
                    WSClean_image_path = "/".join(ms_file.split("/")[:-2]) + "/WSClean/" + ms_file.split("/")[-1] + "_WSClean-image.fits"  # ALL CHANNELS
                else:
                    # add individual channel images
                    # NOT DONE YET!!!
                    WSClean_image_path = f"/scratch/izar/krishna/MeerKAT/WSClean/J0159.0500_Pixels_{channel_start}_{channel_end}_channels-image.fits"

        ms = measurement_set.MwaMeasurementSet(ms_file)

        

else:
    raise Exception("You must enter atleast the telescope name and object name along with the script!")


print("Running bluebild on ", instrument_name, ".MS file of object:", object_name, "\n channels:", channel_id,"\n.MS file path:", ms_file)

# Instrument
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

print (f"The frequency of Observation is:"+str(frequency/1e9)+" GHz and the wavelength is:"+ str(wl) + " m")

# Imaging

N_bits = 32

if (WSClean_grid): 
    cl_WCS = ifits.wcs(WSClean_image_path)
    cl_WCS = cl_WCS.sub(['celestial'])
    cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))  # downsample, too high res!
    cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame

    px_grid = cl_pix_icrs
else: 
    if (ms_fieldcenter):
        _, _, px_colat, px_lon = grid.equal_angle(N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV)
    else:
        _, _, px_colat, px_lon = grid.equal_angle(N=ms.instrument.nyquist_rate(wl), direction=user_fieldcenter.cartesian.xyz.value, FoV=FoV)
    print("nyquist rate", ms.instrument.nyquist_rate(wl), "px_col {0}, px_lon {1}".format(px_colat.shape, px_lon.shape))

    px_grid = transform.pol2cart(1, px_colat, px_lon)
"""


#"""


print("Grid size is:", px_grid.shape[1], px_grid.shape[2])

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start_index, time_end_index, time_slice), column=column_name)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    I_est.collect(S, G)

if (clustering):
    N_eig, c_centroid = I_est.infer_parameters()
else:
    N_eig, c_centroid = N_level ,np.arange(N_level)

print("N_eig:", N_eig)

# Imaging
print ("centroids = ", c_centroid)
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
#I_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits) # replace px_grid 
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start_index, time_end_index, time_slice), column=column_name)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)

    print (S.shape, D.shape, V.shape, G.shape) #(128, 128) (110,) (128, 110)

    #fig_vis, ax_vis = plt.subplots(1, V.shape[0] + 1)

    #ax_vis[0].imshow(S)
    #ax_vis[0].set_title("Original Visibilities")
    #for ax_num in np.arange(V.shape[0]):
    #    ax_vis[ax_num+1].imshow(V)
    #    ax_vis[ax_num+1].set_title("Decomposed Visibility [%d]"%ax_num)

    #fig_vis.tight_layout()

    #c_idx = [0,1,2,3] 

    #_ = I_mfs(D, V, XYZ.data, W.data, c_idx)
    """
    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.data.toarray())
    V_gpu  = cp.asarray(V)

    _ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    """
    _ = I_mfs(D, cp.asarray(V), cp.asarray(XYZ.data), cp.asarray(W.data.toarray()), c_idx)

I_std, I_lsq = I_mfs.as_image()


#"""
### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::time_slice]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)

if (clustering):
    N_eig = S_est.infer_parameters()
else: 
    N_eig = N_level

print("Running sensitivity imaging")
# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
#S_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start_index, time_end_index, time_slice), column=column_name)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(G)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()

#I_std_eq = s2image.Image(np.flip(np.flip(I_std.data/S.data, 1), 2), I_std.grid) #  / S.data
#I_lsq_eq = s2image.Image(np.flip(np.flip(I_lsq.data/S.data, 1), 2), I_lsq.grid) # / S.data

I_std_eq = s2image.Image(I_std.data/S.data, I_std.grid)
I_lsq_eq = s2image.Image(I_lsq.data/S.data, I_lsq.grid) 
"""
# Unscaled image
I_std_eq = s2image.Image(np.flip(np.flip(I_std.data,1),2), I_std.grid) #  / S.data
I_lsq_eq = s2image.Image(np.flip(np.flip(I_lsq.data,1),2), I_lsq.grid) # / S.data


#"""

end_time = time.process_time()

minVal = None
maxVal = None

if (direction_correction == 1):
    print ("Direction Correction Initiated")
    I_std_eq = s2image.Image(np.flip(np.fliplr((I_std_eq.data)), 2), ((I_std_eq.grid)))
    I_lsq_eq = s2image.Image(np.flip(np.fliplr((I_lsq_eq.data)), 2), ((I_lsq_eq.grid))) 

if (normalization == "max"): 
    print ("Max Normalization Initiated")
    I_std_eq = s2image.Image(I_std_eq.data/np.max(np.sum(I_std_eq.data, axis = 0)), I_std_eq.grid)
    I_lsq_eq = s2image.Image(I_lsq_eq.data/np.max(np.sum(I_lsq_eq.data, axis = 0)), I_lsq_eq.grid) 
    maxVal = 1
    minVal = 0
    cbar_levels = np.linspace(0, 1, 10)
elif ((normalization == "mean") or (normalization == "standard")):
    global_mean_std = np.mean(np.sum(I_std_eq.data, axis = 0))
    global_mean_lsq = np.mean(np.sum(I_lsq_eq.data, axis = 0))

    global_deviation_std = np.std(np.sum(I_std_eq.data, axis = 0))
    global_deviation_lsq = np.std(np.sum(I_lsq_eq.data, axis = 0))
    
    levels_mean_std = np.tile(np.mean(I_std_eq.data, axis = (1,2)), (I_std_eq.data.shape[1], I_std_eq.data.shape[2], 1)).T
    levels_mean_lsq = np.tile(np.mean(I_lsq_eq.data, axis = (1,2)), (I_lsq_eq.data.shape[1], I_lsq_eq.data.shape[2], 1)).T

    levels_deviation_std = np.tile(np.std(I_std_eq.data, axis = (1,2)), (I_std_eq.data.shape[1], I_std_eq.data.shape[2], 1)).T
    levels_deviation_lsq = np.tile(np.std(I_lsq_eq.data, axis = (1,2)), (I_lsq_eq.data.shape[1], I_lsq_eq.data.shape[2], 1)).T

    I_std_levels_eq = s2image.Image((I_std_eq.data - levels_mean_std)/levels_deviation_std, I_std_eq.grid)
    I_lsq_levels_eq = s2image.Image((I_lsq_eq.data - levels_mean_lsq)/levels_deviation_lsq, I_lsq_eq.grid)

    I_std_eq = s2image.Image((np.sum(I_std_eq.data, axis = 0) - global_mean_std)/global_deviation_std, I_std_eq.grid)
    I_lsq_eq = s2image.Image((np.sum(I_lsq_eq.data, axis = 0) - global_mean_lsq)/global_deviation_lsq, I_lsq_eq.grid)
    
    maxVal = 4
    minVal = -4
    cbar_levels= np.linspace(-4,4, 10)

    if (normalization=="standard"):
        clipped_global_std = np.clip(I_std_eq.data, -3, 3)
        clipped_global_lsq = np.clip(I_lsq_eq.data, -3, 3)
        clipped_levels_std = np.clip(I_std_levels_eq.data, -3, 3)
        clipped_levels_lsq = np.clip(I_lsq_levels_eq.data, -3, 3)

        I_std_levels_eq = s2image.Image((clipped_levels_std + np.max(clipped_levels_std))/(np.max(clipped_levels_std) - np.min(clipped_levels_std)), I_std_levels_eq.grid)
        I_lsq_levels_eq = s2image.Image((clipped_levels_lsq + np.max(clipped_levels_lsq))/(np.max(clipped_levels_lsq) - np.min(clipped_levels_lsq)), I_lsq_levels_eq.grid)

        I_std_eq = s2image.Image((clipped_global_std + np.max(clipped_global_std))/(np.max(clipped_global_std) - np.min(clipped_global_std)), I_std_eq.grid)
        I_lsq_eq = s2image.Image((clipped_global_lsq + np.max(clipped_global_lsq))/(np.max(clipped_global_lsq) - np.min(clipped_global_lsq)), I_lsq_eq.grid)

        maxVal = 0
        minVal = 1
        cbar_levels= np.linspace(0, 1, 10)

std_range = np.max(I_std_eq.data) - np.min(I_std_eq.data)
lsq_range = np.max(I_lsq_eq.data) - np.min(I_lsq_eq.data)

print("Time elapsed: {0}s".format(end_time - start_time))



# Plot Results ================================================================
WSClean_image = ap_fits.getdata(WSClean_image_path)[0, :, :]

if (WSClean_image.shape[0] == 1):
    WSClean_image = WSClean_image.reshape((WSClean_image.shape[1], WSClean_image.shape[2]))

if (normalization == "max"): 
        WSClean_image = WSClean_image /np.max(WSClean_image)
elif ((normalization == "mean") or (normalization == "standard")): 
        WSClean_image = (WSClean_image - WSClean_image.mean())/np.std(WSClean_image)

        if (normalization =="standard"):
            WSClean_image = np.clip(WSClean_image, -3, 3)
            WSClean_image = (WSClean_image + WSClean_image.max())/ (WSClean_image.max() - WSClean_image.min())
WSC_range = np.max(WSClean_image) - np.min(WSClean_image)


if ((ms_file.split("/")[-3] =="simulation") and (mode=="mwa")):

    fig, ax = plt.subplots(nrows=4, ncols=N_level+1, figsize=(40,30))

    simulation = scipy.io.readsav("/scratch/izar/krishna/MWA/20151203_240MHz_psimas.sav")

    sim = simulation['quantmap'][0][0]

    if "thresholded" in ms_file:
        sim = (sim > (np.mean(sim) + 2 * np.std(sim))) * sim

    if (normalization == "max"): 
        sim = sim/np.max(sim) 
    elif ((normalization == "mean") or (normalization == "standard")):
        sim = (sim - np.mean(sim))/np.std(sim)
        if (normalization == "standard"):
            sim = np.clip(sim, -3, 3)
            sim = (sim + sim.max())/ (sim.max() - sim.min())
    sim_range = np.max(sim) - np.min(sim)

    # Fidelity calculation 
    if (fidelity_calculation == 1): 
        if ((normalization == "mean") or (normalization == "standard")):
            stdImg_fidelity = (I_std_eq.data / np.abs(I_std_eq.data - sim))
            lsqImg_fidelity = (I_lsq_eq.data / np.abs(I_lsq_eq.data - sim))
        elif ((normalization == "max") or (normalization =="none")): 
            stdImg_fidelity = (np.sum(I_std_eq.data, axis = 0) / np.abs(np.sum(I_std_eq.data, axis = 0) -  sim))
            lsqImg_fidelity = (np.sum(I_lsq_eq.data, axis = 0) / np.abs(np.sum(I_lsq_eq.data, axis = 0) - sim))
        wscImg_fidelity = (WSClean_image / np.abs(WSClean_image - sim))

    if (structural_similarity_calculation == 1):
        if ((normalization == "max") or (normalization == "standard")):
            stdImg_ss = skimage.metrics.structural_similarity(sim, np.sum(I_std_eq.data, axis = 0), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            lsqImg_ss = skimage.metrics.structural_similarity(sim, np.sum(I_lsq_eq.data, axis = 0), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            wscImg_ss = skimage.metrics.structural_similarity(sim, WSClean_image, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            
        elif (normalization == "mean"):
            print (sim.shape, I_std_eq.data.shape)
            stdImg_ss = skimage.metrics.structural_similarity(sim, I_std_eq.data.reshape(sim.shape[0], sim.shape[0]), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=max(std_range, sim_range))
            lsqImg_ss = skimage.metrics.structural_similarity(sim, I_lsq_eq.data.reshape(sim.shape[0], sim.shape[0]), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=max(lsq_range, sim_range))
            wscImg_ss = skimage.metrics.structural_similarity(sim, WSClean_image, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=max(WSC_range, sim_range))

elif (WSClean_levels):
    fig, ax = plt.subplots(figsize=(40,30), ncols=N_level + 1, nrows=3)

else:
    fig, ax = plt.subplots(figsize=(40,30), ncols=N_level + 1, nrows=2)


for i in np.arange(N_level + 1):
    if (i == 0):
        I_std_eq.draw(ax=ax[0,i], data_kwargs=dict(cmap='cubehelix', vmin=minVal, vmax=maxVal), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
        ax[0,i].set_title("Standardized") 
        I_lsq_eq.draw(ax=ax[1,i], data_kwargs=dict(cmap='cubehelix', vmin=minVal, vmax=maxVal), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
        ax[1,i].set_title("Least-Square")
    else:
        if ((normalization=="max") or (normalization =="none")):
            I_std_eq.draw(index=i-1, ax=ax[0,i], data_kwargs=dict(cmap='cubehelix', vmin=minVal, vmax=maxVal), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5)) # catalog=sky_model.xyz.T,
            ax[0,i].set_title("Level = {0}".format(i))
            I_lsq_eq.draw(index=i-1, ax=ax[1,i], data_kwargs=dict(cmap='cubehelix', vmin=minVal, vmax=maxVal), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5)) # catalog=sky_model.xyz.T,
            ax[1,i].set_title("Level = {0}".format(i))
        elif ((normalization == "mean") or (normalization == "standard")):
            I_std_levels_eq.draw(index=i-1, ax=ax[0,i], data_kwargs=dict(cmap='cubehelix', vmin=minVal, vmax=maxVal), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5)) # catalog=sky_model.xyz.T,
            ax[0,i].set_title("Level = {0}".format(i))
            I_lsq_levels_eq.draw(index=i-1, ax=ax[1,i], data_kwargs=dict(cmap='cubehelix', vmin=minVal, vmax=maxVal), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5)) # catalog=sky_model.xyz.T,
            ax[1,i].set_title("Level = {0}".format(i))

    if (WSClean_levels):
        WSClean_scale= ax[2, i].imshow((WSClean_image), cmap='cubehelix', vmin=minVal, vmax=maxVal)
        ax[2, i].set_title("WSClean Image")
        divider = make_axes_locatable(ax[2, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(WSClean_scale, cax)

        ax[2, i].set_title("WSC Image")
        
    if ((ms_file.split("/")[-3] =="simulation") and (mode=="mwa")):
    
        simulation_scale = ax[3, i].imshow(sim, cmap='cubehelix', vmin=minVal, vmax=maxVal)
        divider = make_axes_locatable(ax[3, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(simulation_scale, cax)
        
        ax[3, i].set_title("Simulation")
        #ax[3, 0].text(5, 20, "Fidelity values\nStd:{:7.2f}\nLsq:{:7.2f}\nWsc:{:7.2f}".format(stdImg_fidelity, lsqImg_fidelity, wscImg_fidelity), bbox={'facecolor': 'white', 'pad': 10})
        #ax[3, 0].text(60, 20, "SSIM values\nStd:{:7.2f}\nLsq:{:7.2f}\nWsc:{:7.2f}".format(stdImg_ss, lsqImg_ss, wscImg_ss), bbox={'facecolor': 'white', 'pad': 10})

fig.suptitle(f'Frequency:{frequency/1e6} MHz')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
if (custom_output_name):
    plt.savefig(output_dir + "simulation_thresholded_MWA_Obsparams_bluebildImaged.png")
else: 
    plt.savefig(output_dir + instrument_name + "_bluebild_" + object_name+".png")


if ((ms_file.split("/")[-3] =="simulation") and (mode=="mwa")):
    fig, ax = plt.subplots(2, 2, figsize=(40,30))
    
    simulation_scale = ax.ravel()[3].imshow(sim, cmap='cubehelix', vmin=minVal, vmax=maxVal)
    divider = make_axes_locatable(ax.ravel()[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(simulation_scale, cax)
    if (fidelity_calculation): 
        ax.ravel()[3].text(5, 20, "Fidelity\nStd:{:7.2f}\nLsq:{:7.2f}\nWsc:{:7.2f}".format(np.sum(stdImg_fidelity), np.sum(lsqImg_fidelity), np.sum(wscImg_fidelity)), bbox={'facecolor': 'white', 'pad': 10})
    
    if (structural_similarity_calculation): 
        ax.ravel()[3].text(60, 20, "SSIM values\nStd:{:7.2f}\nLsq:{:7.2f}\nWsc:{:7.2f}".format(stdImg_ss, lsqImg_ss, wscImg_ss), bbox={'facecolor': 'white', 'pad': 10})
    ax.ravel()[3].set_title("Simulation")
else :
    fig, ax = plt.subplots(figsize=(40,30), ncols=3, nrows=1)

I_std_eq.draw(ax=ax.ravel()[0], data_kwargs=dict(cmap='cubehelix', vmin=minVal, vmax=maxVal), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.ravel()[0].set_title("BB Standardized Image")
 
I_lsq_eq.draw(ax=ax.ravel()[1], data_kwargs=dict(cmap='cubehelix', vmin=minVal, vmax=maxVal), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.ravel()[1].set_title("BB Least-Squares Image")

WSClean_scale= ax.ravel()[2].imshow(WSClean_image, cmap='cubehelix', vmin=minVal, vmax=maxVal)
ax.ravel()[2].set_title("WSClean Image")
divider = make_axes_locatable(ax.ravel()[2])
cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(WSClean_scale, cax)

ax.ravel()[2].set_title("WSC Image")



fig.suptitle(f'Frequency:{frequency/1e6} MHz')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

if (custom_output_name):
    plt.savefig(output_dir + "simulation_thresholded_MWA_Obsparams_BB_WSC_Comparison.png")
else: 
    plt.savefig(output_dir + instrument_name + "_bluebild_WSClean_Comparison_" + object_name + ".png")

if (fidelity_calculation == 1): 
    fig, ax = plt.subplots(1, 3, figsize = (20,20))
    
    std_scale = ax[0].imshow(stdImg_fidelity.reshape(sim.shape[0], sim.shape[0]), cmap='cubehelix')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(std_scale, cax)
    ax[0].set_title("Standardized Fidelity")

    lsq_scale = ax[1].imshow(lsqImg_fidelity.reshape(sim.shape[0], sim.shape[0]), cmap='cubehelix')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(lsq_scale, cax)
    ax[1].set_title("Least-Square Fidelity")

    wsc_scale = ax[2].imshow(wscImg_fidelity.reshape(sim.shape[0], sim.shape[0]), cmap='cubehelix')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(wsc_scale, cax)
    ax[2].set_title("WSC Fidelity")

    plt.tight_layout()

    if (custom_output_name):
        plt.savefig(output_dir + "simulation_thresholded_MWA_Obsparams_BB_WSC_Fidelity_Comparison.png")
    else: 
        plt.savefig(output_dir + instrument_name + "_bluebild_WSClean_fidelity_Comparison_" + object_name + ".png")


### Interpolate critical-rate image to any grid resolution ====================
# Example: to compare outputs of WSCLEAN and Bluebild with AstroPy/DS9, we
# interpolate the Bluebild estimate at CLEAN (cl_) sky coordinates.

# 1. Load pixel grid the CLEAN image is defined on.

start_interp_time = time.process_time()

cl_WCS = ifits.wcs(WSClean_image_path)
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))  # downsample, too high res!

"""
# everything below is needed for interpolating bb grid to wsclean grid
# not needed if wsc grid used as input for bb grid 

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

f_interp = np.stack(interpolated_maps, axis=0).reshape((N_level,) + cl_pix_icrs.shape[1:])
f_interp = f_interp / (ms.instrument.nyquist_rate(wl) + 1)
f_interp = np.clip(f_interp, 0, None)
fig, ax = plt.subplots(ncols=N_level, nrows=2)


#for i in range(N_level):
#    I_lsq_eq_orig = s2image.Image(I_lsq_eq.data[i,], I_lsq_eq.grid)
#    I_lsq_eq_orig.draw(catalog=sky_model.xyz.T, ax=ax[0,i])
#    ax[0,i].set_title("Critically sampled Bluebild Standard Image Level = {0}".format(i))
#
#    I_lsq_eq_interp = s2image.Image(f_interp[i,], cl_pix_icrs)
#    I_lsq_eq_interp.draw(ax=ax[1,i])
#    ax[1,i].set_title("Interpolated Bluebild Standard Image Level = {0}".format(i))
#plt.show()
#plt.savefig("4gauss_interp")'''

# 5. Store the interpolated Bluebild image in standard-compliant FITS for view
# in AstroPy/DS9.
f_interp = (f_interp  # We need to transpose axes due to the FORTRAN
            .reshape(N_level, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
            .transpose(0, 2, 1))



"""
f_interp = I_lsq_eq.data.transpose(0, 2, 1)
#"""
I_lsq_eq_interp = s2image.WCSImage(np.vstack((np.sum(f_interp, axis = 0).reshape(1, f_interp.shape[1], f_interp.shape[2]), f_interp)), cl_WCS)

if (custom_output_name):
    I_lsq_eq_interp.to_fits(output_dir + "simulation_MWA_thresholded_Obsparams.ms_Bluebild.fits")
else:
    I_lsq_eq_interp.to_fits(output_dir + instrument_name + "_bluebild_" + object_name + ".fits")
end_interp_time = time.process_time()

print("Time to make BB image: {0}s".format(end_time - start_time))
print("Time to reinterpolate image: {0}s".format(end_interp_time - start_interp_time))
