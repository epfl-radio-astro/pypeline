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

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV, frequency = np.deg2rad(20), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=40)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

# Imaging
N_pix = 512
eps = 1e-5
w_term = True
precision = 'single'

t1 = tt.process_time()
N_level = 3
time_slice = 25

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
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))
UVW_baselines = []
gram_corrected_visibilities = []

for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    UVW_baselines.append(UVW_baselines_t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)
    D, V, c_idx = I_dp(S, G)
    S_corrected = IV_dp(D, V, W, c_idx)
    gram_corrected_visibilities.append(S_corrected)

UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3).reshape(*S_corrected.shape[:2], -1)

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

# NUFFT Synthesis
print("Running NUFFT on the CPU")
t = tt.process_time()
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
#print(nufft_imager._synthesizer._inner_fft_sizes)
lsq_image, sqrt_image = nufft_imager(gram_corrected_visibilities)
print("time elapsed: {0}".format(tt.process_time() - t))
print("Running NUFFT on the GPU")
t = tt.process_time()
cunufft_imager = bb_im.CUNUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
#print(cunufft_imager._synthesizer._inner_fft_sizes)
lsq_image_cu, sqrt_image_cu = cunufft_imager(gram_corrected_visibilities)
print("time elapsed: {0}".format(tt.process_time() - t))
print("Avg diff of lsq_img:", np.mean(lsq_image_cu-lsq_image))
print("Avg diff of sqrt_img:", np.mean(sqrt_image_cu-sqrt_image))

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
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []
for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    D, V = S_dp(G)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    sensitivity_coeffs.append(S_sensitivity)

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
print("Running NUFFT on the CPU")
t = tt.process_time()
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
sensitivity_image = nufft_imager(sensitivity_coeffs)
print("time elapsed: {0}".format(tt.process_time() - t))
print("Running NUFFT on the GPU")
t = tt.process_time()
cunufft_imager = bb_im.CUNUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
sensitivity_image_cu = cunufft_imager(sensitivity_coeffs)
print("time elapsed: {0}".format(tt.process_time() - t))
print("Avg diff of sens_img:", np.mean(sensitivity_image_cu-sensitivity_image))

I_lsq_eq     = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq    = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_lsq_eq_cu  = s2image.Image(lsq_image_cu / sensitivity_image_cu, cunufft_imager._synthesizer.xyz_grid)

I_sqrt_eq_cu      = s2image.Image(sqrt_image_cu / sensitivity_image_cu, cunufft_imager._synthesizer.xyz_grid)
I_lsq_diff        = s2image.Image(I_sqrt_eq_cu.data-I_sqrt_eq.data, nufft_imager._synthesizer.xyz_grid)
#I_lsq_pctdiff     = s2image.Image( np.divide(I_sqrt_eq_cu.data-I_sqrt_eq.data, I_sqrt_eq_cu.data, out=np.zeros_like(I_sqrt_eq.data), where=I_sqrt_eq.data!=0), nufft_imager._synthesizer.xyz_grid)
t2 = tt.process_time()
print(f'Total elapsed time for all processes: {t2 - t1} seconds.')

print(I_lsq_diff.data.shape)
print( max(np.max( np.sum(I_sqrt_eq_cu.data, axis = 0)), np.max( np.sum(I_sqrt_eq.data, axis = 0))))

fig, ax = plt.subplots(ncols = 3)
fig.suptitle("FoV = {0:0.2f}, eps = {1}".format(FoV,eps))
ax = ax.flatten()
zmin = min(np.min(I_sqrt_eq_cu.data), np.min(I_sqrt_eq.data))
zmax = 50000#

I_lsq_eq.draw(   catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix', vmin = 0, vmax = 4e4 ), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
I_lsq_eq_cu.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix',  vmin = 0, vmax = 4e4  ), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
I_lsq_diff.draw( catalog=sky_model.xyz.T, ax=ax[2], data_kwargs=dict(cmap='coolwarm', vmin = -400, vmax = 400), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
#I_lsq_pctdiff.draw( catalog=sky_model.xyz.T, ax=ax[3], data_kwargs=dict(cmap='coolwarm', vmin = -300, vmax = 300), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax[0].set_title('NUFFT CPU')
ax[1].set_title('NUFFT GPU')
ax[2].set_title('GPU - CPU')
#ax[3].set_title('GPU - CPU / GPU')
fig.savefig("test_cunufft.png")
fig.show()
plt.show()
