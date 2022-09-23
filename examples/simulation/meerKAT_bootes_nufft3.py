# #############################################################################
# MeerKAT_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# Author : Shreyam Parth Krishna [shreyamkrishna@gmail.com]
# #############################################################################

"""
Simulation MeerKAT imaging with Bluebild (NUFFT).
"""
import matplotlib 
matplotlib.use('agg')
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
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import imot_tools.io.plot as implt
import time as tt

import plotly.offline as py
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

def density_scatter( x , y, ax = None, fig = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


t0 = tt.time()
#Output Directory
output_dir = "/scratch/izar/krishna/MeerKAT/"

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
ra_field_center = 90#150.119
dec_field_center = -60#2.205
field_center = coord.SkyCoord(ra=ra_field_center * u.deg, dec=dec_field_center * u.deg, frame="icrs") # southern hemisphere COSMOS pointing
#field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs") # LOFAR pointing
frequency = 1283895507.8125 # Hz
wl = constants.speed_of_light / frequency
FoV = 1.02 * wl/15 # 1.02* lambda/D, D - individual antenna diameter

# Instrument
N_station = 64 # max 64
dev = instrument.MeerkatBlock(N_station)

#fig_layout,ax_layout = dev.draw()
#fig_layout.savefig(output_dir + "MeerKAT_Antenna_Layout")

mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()

# Data generation
T_integration = 8 # 8 units integrated at a time. 
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=5)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time_range = 120 # in minutes
time = obs_start + (T_integration * u.s) * np.arange(time_range*60/T_integration) # 7200 = 16 hours 3600 = 8 hours 1800 = 4 hours
obs_end = time[-1]
print ("Total integration time: {0} hours".format((obs_end - obs_start)*24))

# Imaging
N_pix = 512
eps = 1e-3
w_term = True
precision = 'single'


N_level = 4
time_slice = 25
print ("Setup Time: {0}s".format(tt.time() - t0))
t1 = tt.time()
### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    #W = np.ones(W.shape)
    G = gram(XYZ, W, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(S, G)

N_eig, c_centroid = I_est.infer_parameters()

print ("Intensity Field Parameter Estimation Time: {0}s".format(tt.time()-t1))
t2 = tt.time()
# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))
UVW_baselines = []
gram_corrected_visibilities = []
gram_corrected_visibilities = np.zeros((2,N_level, 0, N_station, N_station))

for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    UVW_baselines.append(UVW_baselines_t)

    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S = vis(XYZ, W, wl)

    D, V, c_idx = I_dp(S, G) # replace c_idx with self-input eigenlevels

    c_idx = np.array([0, 1, 2, 3]) # workaround for nonbiased?? eigen levels right now
    
    S_corrected = IV_dp(D, V, W, c_idx )
    #gram_corrected_visibilities.append(S_corrected)
    # shape of S_corrected needs some work, sometimes below snippet works, sometimes above one does. 
    gram_corrected_visibilities = np.append(gram_corrected_visibilities, S_corrected.reshape(2,N_level,1, N_station, N_station), axis = 2)

UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
"""
Remove clustering and continue!!

p1 144 (2, 3, 58, 58)
p2 (2, 3, 144, 58, 58)
p3 (2, 3, 484416)

print ("p1", len(gram_corrected_visibilities), gram_corrected_visibilities[0].shape)
for i in np.arange(len(gram_corrected_visibilities)):
        #if ((gram_corrected_visibilities[i].shape[0]!= 2) & (gram_corrected_visibilities[i].shape[1]!= 3) & (gram_corrected_visibilities[i].shape[2]!= 58) & (gram_corrected_visibilities[i].shape[2]!= 58)):
        print (i, gram_corrected_visibilities[i].shape)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3)
#"""
gram_corrected_visibilities = gram_corrected_visibilities.reshape(*S_corrected.shape[:2], -1)

#fig_uvw, ax_uvw = plt.subplots(1,1, figsize = (20,20))
#ax_uvw.scatter3D(UVW_baselines[::N_station, 0], UVW_baselines[::N_station, 1], UVW_baselines[::N_station, -1], s=.01)
#ax_uvw.set_xlabel('u')
#ax_uvw.set_ylabel('v')
#ax_uvw.set_zlabel('w')
#fig_uvw.savefig(output_dir + "meerKAT_simulated_UVW_baselines")

fig_uv, ax_uv = plt.subplots(1,1, figsize = (20,20))
ax_uv = density_scatter(UVW_baselines[:, 0], UVW_baselines[:, 1], ax = ax_uv, fig = fig_uv)
#ax_uv.scatter(UVW_baselines[:, 0], UVW_baselines[:, 1], s=.01)
ax_uv.set_xlabel('u')
ax_uv.set_ylabel('v')
fig_uv.savefig(output_dir + "meerKAT_simulated_UV_baselines_RA_" + str(ra_field_center) + "_Dec_" + str(dec_field_center))

# NUFFT Synthesis
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
print(nufft_imager._synthesizer._inner_fft_sizes)
lsq_image, sqrt_image = nufft_imager(gram_corrected_visibilities)
print ("Intensity Field Imaging Time: {0}s".format(tt.time() - t2))

t3 = tt.time()
### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    #W = np.ones(W.shape)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

print ("Sensitiity Field Parameter Estimation Time: {0}s".format(tt.time() - t3))

t4 = tt.time()
# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []
for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    #W = np.ones(W.shape)
    G = gram(XYZ, W, wl)
    D, V = S_dp(G)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    sensitivity_coeffs.append(S_sensitivity)

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
print(nufft_imager._synthesizer._inner_fft_sizes)
sensitivity_image = nufft_imager(sensitivity_coeffs)

print ("Sensitivity Field Imaging Time: {0}s".format(tt.time() - t4))

I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)

print('Total Calculation Elapsed time: {0} seconds.'.format(tt.time() - t0))

t5 = tt.time()

fig_lsq, ax_lsq = plt.subplots(1,1, figsize = (20,20))
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax_lsq, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax_lsq.set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), MeerKAT: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')
fig_lsq.savefig(output_dir+"meerKAT_simulated_nufft_lsq_RA_" + str(ra_field_center) + "_Dec_" + str(dec_field_center))

fig_sqrt, ax_sqrt = plt.subplots(1,1, figsize = (20,20))
I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax_sqrt, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax_sqrt.set_title(f'Bluebild sqrt, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), MeerKAT: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')
fig_sqrt.savefig(output_dir+"meerKAT_simulated_nufft_sqrt_RA_" + str(ra_field_center) + "_Dec_" + str(dec_field_center))

fig, ax = plt.subplots(1, N_level, figsize=(20,20))
titles = ['Level ' + str(n) for n in np.arange(N_level)]
for i in range(lsq_image.shape[0]):
    plt.title(titles[i])
    I_lsq_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax[i], data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5), show_gridlines=False)

fig.suptitle(f'Bluebild Eigenmaps')
fig.savefig(output_dir+"meerKAT_simulated_eigenlevels_nufft_lsq_RA_" + str(ra_field_center) + "_Dec_" + str(dec_field_center))

print ("Plotting Time: {0}s".format(tt.time() - t5))
print ("Total Elapsed Time: {0}s".format(tt.time() - t0))
