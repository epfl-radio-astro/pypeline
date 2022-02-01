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
FoV, frequency = np.deg2rad(12), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 38
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=60)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

# Imaging
N_pix = 256 ** 2
eps = 1e-3
w_term = True
precision = 'single'
hermitian = True
grid_type = 'healpix'
visu = 'healpy'

t1 = tt.time()
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
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision,
                                      hermitian=hermitian, grid_type=grid_type)
if w_term:
    print(nufft_imager._synthesizer._inner_fft_sizes)
lsq_image, sqrt_image = nufft_imager(gram_corrected_visibilities)

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
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision, hermitian=hermitian, grid_type=grid_type)
if w_term:
    print(nufft_imager._synthesizer._inner_fft_sizes)
sensitivity_image = nufft_imager(sensitivity_coeffs)

I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

if grid_type == 'dircosines' or visu=="imot_tools":
    plt.figure()
    ax = plt.gca()
    I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=True,
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5))
    ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n'
                 f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
                 f'Run time {np.floor(t2 - t1)} seconds.')

    plt.figure()
    ax = plt.gca()
    I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False,
                   catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5))
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
                      catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5), show_gridlines=False)

    plt.suptitle(f'Bluebild Eigenmaps')

else:
    import healpy as hp
    import healpy.pixelfunc as hpix
    import healpy.visufunc as hvisu
    import healpy.fitsfunc as hfits

    lsq_eq_hp = np.zeros((N_level, hpix.nside2npix(nufft_imager._synthesizer._nside)), dtype=lsq_image.dtype)
    lsq_eq_hp[lsq_eq_hp == 0] = np.NaN
    lsq_eq_hp[:, nufft_imager._synthesizer._ipix] = lsq_image / sensitivity_image

    sqrt_eq_hp = np.zeros((N_level, hpix.nside2npix(nufft_imager._synthesizer._nside)), dtype=lsq_image.dtype)
    sqrt_eq_hp[sqrt_eq_hp == 0] = np.NaN
    sqrt_eq_hp[:, nufft_imager._synthesizer._ipix] = sqrt_image / sensitivity_image
    fig = plt.figure()
    hvisu.gnomview(map=np.sum(lsq_eq_hp, axis=0), rot=(field_center.data.lon.deg, field_center.data.lat.deg, 0), xsize=np.ceil(np.sqrt(lsq_image.size)), fig=fig.number,
                   title=f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n'
                 f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
                 f'Run time {np.floor(t2 - t1)} seconds.', cmap='cubehelix', bgcolor='w', reso=60 * np.rad2deg(FoV)/np.ceil(np.sqrt(lsq_image.size)))
    hvisu.projscatter(hp.vec2dir(sky_model.xyz.T, lonlat=True), s=30, lonlat=True, marker='o', edgecolor='w', facecolor='None', linewidths=0.5)
    hp.graticule(alpha=0.3)
    lsq_eq_hp[np.isnan(lsq_eq_hp)] = hp.UNSEEN
    hfits.write_map('bb_lsq.fits', np.sum(lsq_eq_hp, axis=0), coord='C', partial=False, overwrite=True)

    fig = plt.figure()
    hvisu.gnomview(map=np.sum(sqrt_eq_hp, axis=0), rot=(field_center.data.lon.deg, field_center.data.lat.deg, 0), xsize=np.ceil(np.sqrt(lsq_image.size)), fig=fig.number,
                   title=f'Bluebild sqrt, sensitivity-corrected image (NUFFT)\n'
                 f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n'
                 f'Run time {np.floor(t2 - t1)} seconds.', cmap='cubehelix', bgcolor='w', reso=60 * np.rad2deg(FoV)/np.ceil(np.sqrt(lsq_image.size)))
    hvisu.projscatter(hp.vec2dir(sky_model.xyz.T, lonlat=True), s=30, lonlat=True, marker='o', edgecolor='w', facecolor='None', linewidths=0.5)
    hp.graticule(alpha=0.3)
    sqrt_eq_hp[np.isnan(sqrt_eq_hp)] = hp.UNSEEN
    hfits.write_map('bb_sqrt.fits', np.sum(sqrt_eq_hp, axis=0), coord='C', partial=False, overwrite=True)

    fig = plt.figure()
    k=0
    titles = ['Strong sources', 'Mild sources', 'Faint Sources']
    for i in range(lsq_image.shape[0]):
        hvisu.gnomview(map=lsq_eq_hp[k], rot=(field_center.data.lon.deg, field_center.data.lat.deg, 0),
                       xsize=np.ceil(np.sqrt(lsq_image.size)), fig=fig.number, sub=(1,3, k+1),
                       title=titles[k], cmap='cubehelix', bgcolor='w',
                       reso=60 * np.rad2deg(FoV) / np.ceil(np.sqrt(lsq_image.size)))
        k+=1
    hvisu.projscatter(hp.vec2dir(sky_model.xyz.T, lonlat=True), s=30, lonlat=True, marker='o', edgecolor='w', facecolor='None',
                      linewidths=0.5)
    hp.graticule(alpha=0.3)
    plt.title('Bluebild Eigenmaps')
    hfits.write_map('bb_eigenmaps.fits', lsq_eq_hp, coord='C', partial=False, overwrite=True, column_names=titles)
