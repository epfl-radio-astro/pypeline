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
import pypeline.phased_array.data_gen.statistics as statistics
from imot_tools.math.func import SphericalDirichlet
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import time as tt

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV, frequency = np.deg2rad(8), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=30)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

# Field center coordinates

field_center_lon, field_center_lat = field_center.data.lon.rad, field_center.data.lat.rad
field_center_xyz = field_center.cartesian.xyz.value

# UVW reference frame
w_dir = field_center_xyz
u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
v_dir = np.array(
    [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
     np.cos(field_center_lat)])
uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)

# Imaging grid
lim = np.sin(FoV / 2)
N_pix = 256
pix_slice = np.linspace(-lim, lim, N_pix)
Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
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
time_slice = 100

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(S, G)

N_eig, c_centroid = I_est.infer_parameters()

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
UVW_baselines = []
ICRS_baselines = []
gram_corrected_visibilities = []
baseline_rescaling = 2 * np.pi / wl

for t in ProgressBar(time[0:25]):
    XYZ = dev(t)
    UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
    UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])
    ICRS_baselines_t = (XYZ.data[:, None, :] - XYZ.data[None, ...])
    UVW_baselines.append(baseline_rescaling * UVW_baselines_t)
    ICRS_baselines.append(baseline_rescaling * ICRS_baselines_t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    D, V, _ = I_dp(S, XYZ, W, wl)
    S_corrected = (W.data @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.data.transpose().conj()
    gram_corrected_visibilities.append(S_corrected)

UVW_baselines = np.stack(UVW_baselines, axis=0)
ICRS_baselines = np.stack(ICRS_baselines, axis=0).reshape(-1, 3)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=0).reshape(-1)


UVW_baselines=UVW_baselines.reshape(-1,3)
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
    D, V = S_dp(XYZ, W, wl)
    S_sensitivity = (W.data @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.data.transpose().conj()
    sensitivity_coeffs.append(S_sensitivity)

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
sensitivity_coeffs *= w_correction
sensitivity_image = finufft.nufft2d1(x=scalingx * UVW_baselines[:, 1],
                                     y=scalingy * UVW_baselines[:, 0],
                                     c=sensitivity_coeffs,
                                     n_modes=N_pix, eps=1e-4)

sensitivity_image = np.real(sensitivity_image)

print(sensitivity_image.shape,sensitivity_image[0,0])

I_lsq_eq = s2image.Image(bb_image / sensitivity_image, pix_xyz)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax.set_title(f'Bluebild Least-squares, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180/np.pi)} degrees.\n'
             f'Run time {np.floor(t2 - t1)} seconds.')

plt.savefig("test2_nufft")


gaussian=np.exp(-(Lpix ** 2 + Mpix ** 2)/(4*lim))
gridded_visibilities=np.sqrt(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian*bb_image)))))
gridded_visibilities[int(gridded_visibilities.shape[0]/2)-2:int(gridded_visibilities.shape[0]/2)+2, int(gridded_visibilities.shape[1]/2)-2:int(gridded_visibilities.shape[1]/2)+2]=0
plt.figure()
plt.imshow(np.flipud(gridded_visibilities), cmap='cubehelix')

