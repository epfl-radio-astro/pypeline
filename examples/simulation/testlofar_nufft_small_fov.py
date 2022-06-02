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
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
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
ms_file = "/work/ska/results_rascil_lofar/ska-pipeline_simulation.ms"
ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
gram = bb_gr.GramBlock()


if read_coords_from_ms:
    cl_WCS = ifits.wcs("/work/ska/results_rascil_lofar/wsclean-image.fits")
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
frequency =145e6
wl = constants.speed_of_light / frequency
freq_ms = ms.channels["FREQUENCY"][channel_id]
print(freq_ms)
assert freq_ms.to_value(u.Hz) == frequency
obs_start, obs_end = ms.time["TIME"][[0, -1]]
print("obs start: {0}, end: {1}".format(obs_start, obs_end))

# Field center coordinates

field_center_lon, field_center_lat = ms.field_center.data.lon.rad, ms.field_center.data.lat.rad
field_center_xyz = ms.field_center.cartesian.xyz.value

# UVW reference frame
w_dir = field_center_xyz
u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
v_dir = np.array(
    [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
     np.cos(field_center_lat)])
uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot3D([0, 1], [0, 0], [0, 0], '-ok', linewidth=2)
# ax.text3D(1, 0, 0, 'x', fontsize='large')
# ax.plot3D([0, 0], [0, 1], [0, 0], '-ok', linewidth=2)
# ax.text3D(0, 1, 0, 'y', fontsize='large')
# ax.plot3D([0, 0], [0, 0], [0, 1], '-ok', linewidth=2)
# ax.text3D(0, 0, 1, 'z', fontsize='large')
#
# ax.plot3D([0, u_dir[0]], [0, u_dir[1]], [0, u_dir[-1]], '-sr', linewidth=2)
# ax.text3D(u_dir[0], u_dir[1], u_dir[-1], 'u', fontsize='large')
# ax.plot3D([0, v_dir[0]], [0, v_dir[1]], [0, v_dir[-1]], '-sr', linewidth=2)
# ax.text3D(v_dir[0], v_dir[1], v_dir[-1], 'v', fontsize='large')
# ax.plot3D([0, w_dir[0]], [0, w_dir[1]], [0, w_dir[-1]], '-sr', linewidth=2)
# ax.text3D(w_dir[0], w_dir[1], w_dir[-1], 'w', fontsize='large')

# Imaging grid
lim = np.sin(FoV / 2)
N_pix = 512
pix_slice = np.linspace(-lim, lim, N_pix)
Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
#pix_xyz = lmn_grid
pix_xyz = np.tensordot(uvw_frame, lmn_grid, axes=1)
_, pix_lat, pix_lon = transform.cart2eq(*pix_xyz)

# ax.scatter3D(pix_xyz[0].flatten(), pix_xyz[1].flatten(), pix_xyz[-1].flatten())

# plt.figure()
# plt.scatter(lmn_grid[0], lmn_grid[1], s=2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.figure()
# plt.scatter(pix_lon * 180 / np.pi, pix_lat * 180 / np.pi, s=2)
# plt.scatter(field_center_lon * 180 / np.pi, field_center_lat * 180 / np.pi, c='r', s=10)
# plt.xlabel('RA')
# plt.ylabel('DEC')

# Imaging Parameters
t1 = tt.time()
N_level = 4
N_bits = 32
time_slice = 1

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
UVW_baselines = []
ICRS_baselines = []
gram_corrected_visibilities = []
baseline_rescaling = 2 * np.pi / wl

for t, f, S, uvw in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, None), column="DATA", return_UVW=True)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    #uvw = ms.UVW([channel_id], t)
    print(uvw.shape)
    UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
    UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])

    ICRS_baselines_t = (XYZ.data[:, None, :] - XYZ.data[None, ...])
    UVW_baselines.append(baseline_rescaling * UVW_baselines_t)
    ICRS_baselines.append(baseline_rescaling * ICRS_baselines_t)

    print(uvw.shape)
    print(UVW_baselines[-1].shape)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)
    W = W.data
    D, V, _ = I_dp(S, G)
    S_corrected = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    gram_corrected_visibilities.append(S_corrected)

UVW_baselines = np.stack(UVW_baselines, axis=0)
ICRS_baselines = np.stack(ICRS_baselines, axis=0).reshape(-1, 3)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=0).reshape(-1)

print('baselines', UVW_baselines.shape,UVW_baselines)

# UVW_baselines = UVW_baselines.reshape((UVW_baselines.shape[0], -1, 3))
#
# plt.figure()
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# for i in range(0, UVW_baselines.shape[1], 10):
#     plt.plot(UVW_baselines[:,i, 0] * 2 * lim / N_pix, UVW_baselines[:,i, 1] * 2 * lim / N_pix, color=colors[0], linewidth=0.01)
# plt.xlim(-np.pi, np.pi)
# plt.ylim(-np.pi, np.pi)


# fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.scatter3D(UVW_baselines[::N_station, 0], UVW_baselines[::N_station, 1], UVW_baselines[::N_station, -1], s=.01)
# # plt.xlabel('u')
# # plt.ylabel('v')
# # ax.set_zlabel('w')
# plt.figure()
# plt.scatter(UVW_baselines[:, 0], UVW_baselines[:, 1], s=0.01)
# plt.xlabel('u')
# plt.ylabel('v')

UVW_baselines=UVW_baselines.reshape(-1,3)

plt.scatter(UVW_baselines_t[:,:,0], UVW_baselines_t[:,:,1])
plt.savefig("lofartest_nufft_new_baselinesUV")

w_correction = np.exp(1j * UVW_baselines[:, -1])
gram_corrected_visibilities *= w_correction
scalingx = 2 * lim / N_pix
scalingy = 2 * lim / N_pix
bb_image = finufft.nufft2d1(x=scalingx * UVW_baselines[:, 1],
                            y=scalingy * UVW_baselines[:, 0],
                            c=gram_corrected_visibilities,
                            n_modes=N_pix, eps=1e-4)

bb_image = np.real(bb_image)

print(bb_image.shape,bb_image[0,0])

### Sensitivity Field =========================================================
# Parameter Estimation
'''
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
sensitivity_coeffs = []
for t in ProgressBar(time[0:25]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    W = W.data
    D, V = S_dp(G)
    S_sensitivity = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    sensitivity_coeffs.append(S_sensitivity)

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
sensitivity_coeffs *= w_correction
sensitivity_image = finufft.nufft2d1(x=scalingx * UVW_baselines[:, 1],
                                     y=scalingy * UVW_baselines[:, 0],
                                     c=sensitivity_coeffs,
                                     n_modes=N_pix, eps=1e-4)

sensitivity_image = np.real(sensitivity_image)

print(sensitivity_image.shape,sensitivity_image[0,0])
'''
I_lsq_eq = s2image.Image(bb_image, pix_xyz)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

plt.figure()
ax = plt.gca()
I_lsq_eq.draw( ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax.set_title(f'Bluebild Least-squares, sensitivity-corrected image (NUFFT)\n')

plt.savefig("lofartest_test_nufft")

print(bb_image.data.shape)
'''
gaussian=np.exp(-(Lpix ** 2 + Mpix ** 2)/(4*lim))
gridded_visibilities=np.sqrt(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian*bb_image)))))
gridded_visibilities[int(gridded_visibilities.shape[0]/2)-2:int(gridded_visibilities.shape[0]/2)+2, int(gridded_visibilities.shape[1]/2)-2:int(gridded_visibilities.shape[1]/2)+2]=0
plt.figure()
plt.imshow(np.flipud(gridded_visibilities), cmap='cubehelix')
'''

if read_coords_from_ms:
    # 5. Store the interpolated Bluebild image in standard-compliant FITS for view
    # in AstroPy/DS9.

    f_interp = (I_lsq_eq.data  # We need to transpose axes due to the FORTRAN
                .reshape(1, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
                .transpose(0, 2, 1))
    I_lsq_eq_interp = s2image.WCSImage(np.sum(f_interp,axis=0), cl_WCS)
    I_lsq_eq_interp.to_fits('bluebild_nufft2_lofartest_combined-test.fits')
    I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
    I_lsq_eq_interp.to_fits('bluebild_nufft2_lofartest_levels-test.fits')

