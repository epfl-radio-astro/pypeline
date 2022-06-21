# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bluebild (NUFFT).
"""

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
import imot_tools.io.fits as ifits
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import scipy.sparse as sparse
import finufft
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.measurement_set as measurement_set
import pypeline.phased_array.data_gen.statistics as statistics
from imot_tools.math.func import SphericalDirichlet
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import time as tt

read_coords_from_ms = True

# Instrument
#ms_file = "/home/etolley/rascil_ska_sim/results_test/ska-pipeline_simulation.ms"
ms_file = "/home/etolley/rascil/examples/pipelines/ska-pipelines/results_rascil_skalow_test/ska-pipeline_simulation.ms"
ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
gram = bb_gr.GramBlock()


if read_coords_from_ms:
    cl_WCS = ifits.wcs("/home/etolley/rascil/examples/pipelines/ska-pipelines/results_rascil_skalow_test/imaging_dirty.fits")
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


field_center_lon, field_center_lat = ms.field_center.data.lon.rad, ms.field_center.data.lat.rad
field_center_xyz = ms.field_center.cartesian.xyz.value

# UVW reference frame
w_dir = field_center_xyz
u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
v_dir = np.array(
    [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
     np.cos(field_center_lat)])
uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)

# Imaging
N_pix = 512
eps = 1e-3
precision = 'single'

t1 = tt.time()
N_level = 1
time_slice = 100

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

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))

nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
for t, f, S, uvw in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, None), column="DATA", return_UVW=True)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
    UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])
    W = ms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    S_corrected = IV_dp(D, V, W, c_idx)
    nufft_imager.collect(UVW_baselines_t, S_corrected)


# NUFFT Synthesis
lsq_image, sqrt_image = nufft_imager.get_statistic()

'''### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=precision)
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)

sensitivity_image = nufft_imager.get_statistic()[0]'''

I_lsq_eq = s2image.Image(lsq_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image, nufft_imager._synthesizer.xyz_grid)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

plt.figure()
ax = plt.gca()
I_lsq_eq.draw(ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)')

plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild sqrt, sensitivity-corrected image (NUFFT)')

plt.figure()
titles = ['Strong sources', 'Mild sources', 'Faint Sources']
for i in range(lsq_image.shape[0]):
    plt.subplot(1, N_level, i + 1)
    ax = plt.gca()
    plt.title(titles[i])
    I_lsq_eq.draw(index=i, ax=ax, data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5), show_gridlines=False)

plt.suptitle(f'Bluebild Eigenmaps')
plt.savefig('skalow_final.png')
#  plt.show()
