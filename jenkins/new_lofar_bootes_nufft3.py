import os
import sys
import time
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import bluebild
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.data_gen.statistics as statistics
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import imot_tools.io.plot as implt
import bb_tb


args = bb_tb.check_args(sys.argv)

np.random.seed(0)


jkt0_s = time.time()

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

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=40)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
times = obs_start + (T_integration * u.s) * np.arange(3595)

# Imaging parameters
N_pix    = 512
N_levels = 3
eps      = 1e-3
time_slice = 200
times = times[::time_slice]

print("\nImaging parameters")
print(f'N_pix {N_pix}\nN_levels {N_levels}\nprecision {args.precision}')
print(f'time_slice {time_slice}\neps {eps}\n')

print("-I- processing unit:", args.processing_unit)
ctx = None if args.processing_unit == None else bluebild.Context(args.processing_unit)

gram = bb_gr.GramBlock(ctx)


### Intensity Field ===========================================================

# Parameter Estimation
ifpe_s = time.time()
I_est = bb_pe.IntensityFieldParameterEstimator(N_levels, sigma=0.95)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
ifpe_e = time.time()
print(f"#@#IFPE {ifpe_e - ifpe_s:.3f} sec")

# Imaging
ifim_s = time.time()
I_dp  = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=args.precision, ctx=ctx)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    S_corrected = IV_dp(D, V, W, c_idx)
    nufft_imager.collect(UVW_baselines_t, S_corrected)
lsq_image, sqrt_image = nufft_imager.get_statistic()
ifim_e = time.time()
print(f"#@#IFIM {ifim_e - ifim_s:.3f} sec")


### Sensitivity Field =========================================================
# Parameter Estimation
sfpe_s = time.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
sfpe_e = time.time()
print(f"#@#SFPE {sfpe_e - sfpe_s:.3f} sec")

# Imaging
sfim_s = time.time()
S_dp  = bb_dp.SensitivityFieldDataProcessorBlock(N_eig, ctx)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps,
                                      n_trans=1, precision=args.precision, ctx=ctx)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)
sensitivity_image = nufft_imager.get_statistic()[0]
I_lsq_eq  = s2image.Image(lsq_image  / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
sfim_e = time.time()
print(f"#@#SFIM {sfim_e - sfim_s:.3f} sec")

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e - jkt0_s:.3f} sec\n")


#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)


bb_tb.dump_data(I_lsq_eq.data, 'I_lsq_eq_data', args.outdir)
bb_tb.dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid', args.outdir)


### Plotting section
plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.')
fp = "I_lsq.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)


plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild sqrt, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.')
fp = "I_sqrt.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

