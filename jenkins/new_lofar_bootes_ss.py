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
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from pypeline.util import frame
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
N_bits   = 32 if args.precision == 'single' else 64
dtype_f  = np.float32 if N_bits == 32 else np.float64
time_slice = 200
times = times[::time_slice]

# Grid
lim = np.sin(FoV / 2)
grid_slice = np.linspace(-lim, lim, N_pix)
l_grid, m_grid = np.meshgrid(grid_slice, grid_slice)
n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
uvw_frame = frame.uvw_basis(field_center)
px_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)

print("\nImaging parameters")
print(f'N_pix {N_pix}\nN_levels {N_levels}\nN_bits {N_bits}')
print(f'time_slice {time_slice}')

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
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx) #EO: bug in C++ version???
#I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx=None)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_levels, N_bits, ctx)
for t in times:
    d2h = True if t == times[-1] else False
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    I_mfs(D, V, XYZ.data, W.data, c_idx, d2h)
I_std, I_lsq = I_mfs.as_image()
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
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits, ctx)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S_ss = S_mfs.as_image()
I_std_eq = s2image.Image(I_std.data / S_ss.data, I_lsq.grid)
I_lsq_eq = s2image.Image(I_lsq.data / S_ss.data, I_lsq.grid)
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
ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (SS)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.')
fp = "I_lsq.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)


plt.figure()
ax = plt.gca()
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild std, sensitivity-corrected image (SS)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.')
fp = "I_std.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

