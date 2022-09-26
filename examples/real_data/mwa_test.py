# #############################################################################
# .py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# Author : Shreyam Parth Krishna [shreyamkrishna@gmail.com]
# #############################################################################

"""
Simulation MWA imaging with Bluebild (NUFFT).
"""
import matplotlib 
matplotlib.use('agg')
from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import astropy.io.fits as ap_fits

import imot_tools.io.fits as ifits
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
from imot_tools.math.func import SphericalDirichlet
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import scipy.constants as constants
import finufft
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as bb_synth
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import imot_tools.io.plot as implt
import pycsou.linop as pyclop
import joblib as job
import time as tt
import scipy.io

import plotly.offline as py
from mpl_toolkits.axes_grid1 import make_axes_locatable


t0 = tt.time()
#Output Directory
output_dir = "/scratch/izar/krishna/MWA/simulation/"
time_slice = 1 # integrate 1 time step in a go (i.e. 0.5 seconds currently, same as paper)
sampling= 1 # ratio of pixels in WSC image v/s bluebild image
WSClean_image_path = "/scratch/izar/krishna/MWA/WSClean/" # only channel 4 image
#WSClean_image_path = "/scratch/izar/krishna/MWA/simulation/" # Path to WSC simulated solar image
synthesizerList = ["ss", "nufft"]
        
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal1024_Pixels_0_64_channels-image.fits" # 1024 pixels, 50"/pixel, 64 channels
WSClean_image_name ="1133149192-187-188_Sun_10s_cal1024_Pixels_4_5_channels-image.fits" # 1024 pixels, 50"/pixel, 4th channel
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal2048_Pixels_0_64_channels-image.fits" # 2048 pixels, 25"/pixel, 64 channel
#WSClean_image_name ="1133149192-187-188_Sun_10s_cal2048_Pixels_4_5_channels-image.fits" # 2048 pixels, 25"/pixel, 4th channel'
#WSClean_image_name ="simulation_MWA_Obsparams.ms_WSClean-image.fits" # 100 pixels,  

WSClean_image_path += WSClean_image_name
#Instrument 
ms_file = "/work/ska/MWA/1133149192-187-188_Sun_10s_cal.ms/"
#ms_file = "/scratch/izar/krishna/MWA/simulation/simulation_MWA_Obsparams.ms"
ms = measurement_set.MwaMeasurementSet(ms_file) # stations 1 - N_station 
gram = bb_gr.GramBlock()



# Observation
FoV = np.deg2rad(50.0 * 1024.0/ 3600.0) # images have vertical and horizontal range of 4000"
#FoV = np.deg2rad(1.388889)
print ("FOV is {0} radians or {1} deg".format(FoV, np.rad2deg(FoV)))
#channel_id = np.arange(0, 4, dtype = np.int) # 64 total channels in MS file
#frequency = (ms.channels["FREQUENCY"][32] + ms.channels["FREQUENCY"][31])/2
channel_id = 4
#channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
obs_start, obs_end = ms.time["TIME"][[0, -1]]

N_level = 3
N_bits = 32
N_station = 128

_, _, px_colat, px_lon = grid.equal_angle(
    N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV
)

print(ms.instrument.nyquist_rate(wl), "px_col {0}, px_lon {1}".format(px_colat.shape, px_lon.shape))

px_grid = transform.pol2cart(1, px_colat, px_lon)

N_pix = 1024
eps = 1e-3
w_term = True
precision="single"


print("Grid size is:", px_colat.shape, px_lon.shape)

print ("Setup Time: {0}s".format(tt.time() - t0))
t1 = tt.time()
### Intensity Field ===========================================================
# Parameter Estimation
"""
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
    ms.visibilities(channel_id=[channel_id], time_id = slice(0, None, time_slice), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    I_est.collect(S, G)

N_eig, c_centroid = I_est.infer_parameters()
"""
N_eig, c_centroid = N_level, list(range(N_level))  

print("N_eig:", N_eig,"centroids = ", c_centroid)

print ("Intensity Field Parameter Estimation Time: {0}s".format(tt.time()-t1))

# Imaging
t2 = tt.time()
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))



UVW_baselines = []
gram_corrected_visibilities = []
gram_corrected_visibilities = np.zeros((2,N_level, 0, N_station, N_station))

I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)

for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, time_slice), column="DATA")
):

    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)

    # for using nuFFT synthesizer; do later
    # new method imported from instrument.py to measurement_set.py
    

    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)


    D, V, c_idx = I_dp(S, G)
    c_idx = list(range(N_eig)) # clustering without kmeans ; is not working when input to IV_dp 

    if ("nufft" in synthesizerList):
        UVW_baselines_t = ms.instrument.baselines(t, True, ms.field_center)
        UVW_baselines.append(UVW_baselines_t)
        S_corrected = IV_dp(D, V, W, c_idx)
        gram_corrected_visibilities = np.append(gram_corrected_visibilities, S_corrected.reshape(2,N_level,1, N_station, N_station), axis = 2)
    
    if ("ss" in synthesizerList):
        XYZ_gpu = cp.asarray(XYZ.data)
        W_gpu  = cp.asarray(W.data.toarray())
        V_gpu  = cp.asarray(V)
        _ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)

if ("ss" in synthesizerList):
    I_std, I_lsq = I_mfs.as_image()

if ("nufft" in synthesizerList):
    UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
    gram_corrected_visibilities = gram_corrected_visibilities.reshape(*S_corrected.shape[:2], -1)

    fig_uvw = plt.figure()
    ax_uvw = Axes3D(fig_uvw)
    ax_uvw.scatter3D(UVW_baselines[::N_station, 0], UVW_baselines[::N_station, 1], UVW_baselines[::N_station, -1], s=.01)
    ax_uvw.set_xlabel('u')
    ax_uvw.set_ylabel('v')
    ax_uvw.set_zlabel('w')
    fig_uvw.savefig(output_dir + "MWA_bluebild_simulated_UVW_baselines")

    fig_uv, ax_uv = plt.subplots(1,1, figsize = (20,20))
    ax_uv.scatter(UVW_baselines[:, 0], UVW_baselines[:, 1], s=.01)
    ax_uv.set_xlabel('u')
    ax_uv.set_ylabel('v')
    fig_uv.savefig(output_dir + "MWA_bluebild_simulated_UV_baselines")


    # NUFFT Synthesis
    nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                        field_center=ms.field_center, eps=eps, w_term=w_term,
                                        n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
    print(nufft_imager._synthesizer._inner_fft_sizes)
    lsq_image, std_image = nufft_imager(gram_corrected_visibilities)

print ("Intensity Field Imaging Time: {0}s".format(tt.time() - t2))

t3 = tt.time()
### Sensitivity Field =========================================================
# Parameter Estimation
"""
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::time_slice]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()
"""
N_eig = 6

print ("Sensitiity Field Parameter Estimation Time: {0}s".format(tt.time() - t3))

t4 = tt.time()
# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []

S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)

for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, time_slice), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    D, V = S_dp(G)

    if ("ss" in synthesizerList):
        XYZ_gpu = cp.asarray(XYZ.data)
        W_gpu  = cp.asarray(W.data.toarray())
        V_gpu  = cp.asarray(V)
        _ = S_mfs(D, V_gpu, XYZ_gpu, W_gpu, cluster_idx=np.zeros(N_eig, dtype=int))

    if ("nufft" in synthesizerList):
        S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
        sensitivity_coeffs.append(S_sensitivity)

if ("ss" in synthesizerList):
    _, S = S_mfs.as_image()
    I_ss_std_eq = s2image.Image(I_std.data/S.data, I_std.grid) #  / S.data
    I_ss_lsq_eq = s2image.Image(I_lsq.data/S.data, I_lsq.grid) # / S.data
    
    # max scaling
    I_ss_std_eq = s2image.Image(I_std.data/(S.data * np.max(I_std.data)), I_std.grid) #  / S.data
    I_ss_lsq_eq = s2image.Image(I_lsq.data/(S.data * np.max(I_lsq.data)), I_lsq.grid) # / S.data

if ("nufft" in synthesizerList):
    sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
    nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=ms.field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
    print(nufft_imager._synthesizer._inner_fft_sizes)
    sensitivity_image = nufft_imager(sensitivity_coeffs)

    I_nufft_lsq_eq = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
    I_nufft_std_eq = s2image.Image(std_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)

    #max scaling
    I_nufft_lsq_eq = s2image.Image(lsq_image / (sensitivity_image * np.max(lsq_image)), nufft_imager._synthesizer.xyz_grid)
    I_nufft_std_eq = s2image.Image(std_image / (sensitivity_image * np.max(std_image)), nufft_imager._synthesizer.xyz_grid)

print ("Sensitivity Field Imaging Time: {0}s".format(tt.time() - t4))

print('Total Calculation Elapsed time: {0} seconds.'.format(tt.time() - t0))

t5 = tt.time()

fig, ax = plt.subplots(3,N_level + 1, figsize = (20,20))
WSClean_image = ap_fits.getdata(WSClean_image_path)

WSClean_MaxScaled_Image = WSClean_image[0, 0, :, :]/np.max(WSClean_image[0, 0, :, :])

for i in np.arange(N_level+1):
    if (i == 0):
        I_nufft_lsq_eq.draw(ax=ax[0,i], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
        ax[0, i].set_title("nuFFT Least-Squares Image (SC)") 
        I_ss_lsq_eq.draw(ax=ax[1,i], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
        ax[1, i].set_title("SS Least-Squares Image (SC)")

    else:
        I_nufft_lsq_eq.draw(index=i-1, ax=ax[0,i], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
        ax[0,i].set_title("nuFFT Image Level = {0}".format(i-1))
        I_ss_lsq_eq.draw(index=i-1, ax=ax[1,i], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
        ax[1,i].set_title("SS Image Level = {0}".format(i-1))

    #WSClean_scale= ax[2, i].imshow((WSClean_image[0, 0, :, :]), cmap='RdPu')
    WSClean_scale= ax[2, i].imshow((WSClean_MaxScaled_Image), cmap='cubehelix')
    ax[2, i].set_title("WSClean Image")
    divider = make_axes_locatable(ax[2, i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(WSClean_scale, cax)

fig.tight_layout()
fig.savefig(output_dir+"MWA_observation_bluebild_levels_WSClean_Comparison.png")

fig, ax = plt.subplots(1,4, figsize = (20,20))


I_nufft_lsq_eq.draw(ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax[0].set_title("nuFFT Least-Squares Image (SC)") 
I_ss_lsq_eq.draw(ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax[1].set_title("SS Least-Squares Image (SC)")

#WSClean_scale= ax[2].imshow((WSClean_image[0, 0, :, :]), cmap='RdPu')
WSClean_scale= ax[2].imshow((WSClean_MaxScaled_Image), cmap='cubehelix')
ax[2].set_title("WSClean Image")
divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = plt.colorbar(WSClean_scale, cax2)

path = "/scratch/izar/krishna/MWA/20151203_240MHz_psimas.sav"
simulation = scipy.io.readsav(path)#, python_dict = True)
ax[3].imshow(simulation['quantmap'][0][0]/np.max(simulation['quantmap'][0][0]), cmap='cubehelix')
ax[3].set_title('Simulation')
divider2 = make_axes_locatable(ax[3])
cax3 = divider2.append_axes("right", size="5%", pad=0.05)
cbar3 = plt.colorbar(WSClean_scale, cax3)

fig.tight_layout()
fig.savefig(output_dir+"MWA_observation_bluebild_WSClean_Comparison.png")


"""
ax_lsq.set_title(f'Bluebild least-squares, sensitivity-corrected image \n'
             f'Real MWA solar data FoV: {np.round(FoV * 180 / np.pi,2)} degrees.\n'
             f'Run time {np.floor(t5 - t1)} seconds.')
fig_lsq.savefig(output_dir+"MWA_Solar_lsq")

fig_std, ax_std = plt.subplots(1,1, figsize = (20,20))
I_std_eq.draw(ax=ax_std, data_kwargs=dict(cmap='cubehelix'), show_gridlines=True, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax_std.set_title(f'Bluebild std, sensitivity-corrected image \n'
             f'Real MWA solar data FoV: {np.round(FoV * 180 / np.pi, 2)} degrees.\n'
             f'Run time {np.floor(t5 - t1)} seconds.')
fig_std.savefig(output_dir+"MWA_Solar_std")

fig, ax = plt.subplots(1, N_level, figsize=(20,20))
titles = ['Level ' + str(n) for n in np.arange(N_level)]
for i in range(I_lsq_eq.shape[0]):
    plt.title(titles[i])
    I_lsq_eq.draw(index=i, ax=ax[i], data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5), show_gridlines=True)

fig.suptitle(f'Bluebild Eigenmaps')
fig.savefig(output_dir+"MWA_Solar_Eigenlevels_lsq")
"""

print ("Plotting Time: {0}s".format(tt.time() - t5))

# interpolation between bb and WSC image
t6 = tt.time()

cl_WCS = ifits.wcs(WSClean_image_path)
cl_WCS = cl_WCS.sub(['celestial'])
cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))  # downsample, too high res!

#"""
# everything below is needed for interpolating bb grid to wsclean grid
# not needed if wsc grid used as input for bb grid 

cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

# 3. Interpolation: Part I.
# Due to the high Nyquist rate in astronomy and large pixel count in the images,
# it is advantageous to do sparse interpolation. Doing so requires first
# computing the interpolation kernel's spatial support per output pixel.
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
                                 (I_nufft_lsq_eq.data.reshape(N_level, -1)[n])
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
f_interp = (np.flip(np.flip(f_interp,axis=2),axis=1)  # Currently image is stored such that WSC image matches to BB image, first undo that
            .reshape(N_level, N_cl_lon, N_cl_lat)  # We need to transpose axes due to the FORTRAN indexing conventions of the FITS standard.
            .transpose(0, 2, 1))

I_nufft_lsq_eq_interp = s2image.WCSImage(np.sum(f_interp, axis = 0), cl_WCS)
I_nufft_lsq_eq_interp.to_fits(output_dir+ "MWA_observation_bluebild_1133149192-Sun_Channel_4.fits")

I_nufft_lsq_levels_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_nufft_lsq_levels_eq_interp.to_fits(output_dir+ "MWA_observation_bluebild_levels.fits")

print("Total interpolation and fits creation time:{0}s".format(tt.time()-t6))
print ("Total Elapsed Time: {0}s".format(tt.time() - t0))
