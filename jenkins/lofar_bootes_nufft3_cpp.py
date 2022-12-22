# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bluebild (NUFFT).
"""

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
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.instrument as instrument
from pypeline.util import frame
import bluebild
import bipptb


args = bipptb.check_args(sys.argv)
#print("-I- args =", args)

# For reproducible results
np.random.seed(0)

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
print("-I- processing unit:", args.processing_unit)
ctx = bluebild.Context(args.processing_unit)


jkt0_s = time.time()

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV_deg = 8.0
FoV, frequency = np.deg2rad(FoV_deg), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock(ctx)

# Data generation
T_integration = 8
N_src     = 40
fs        = 196000
SNR       = 30
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=N_src)
vis       = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=fs, SNR=SNR)
times     = obs_start + (T_integration * u.s) * np.arange(3595)
N_antenna = dev(times[0]).data.shape[0]

# Imaging parameters
N_pix      = 512
N_levels   = 3
precision  = args.precision
eps        = 1e-3
time_slice = 200
times = times[::time_slice]

print("\nImaging parameters")
print(f'N_pix {N_pix}\nN_levels {N_levels}\nprecision {args.precision}')
print(f'time_slice {time_slice}\neps {eps}\n')

# Grids
lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)
px_w = xyz_grid.shape[1]
px_h = xyz_grid.shape[2]


print(f"-I- precision = {precision}")
print(f"-I- N_station =", N_station)
print(f"-I- N_antenna = {N_antenna:d}")
print(f"-I- T_integration =", T_integration)
print(f"-I- Field center  =", field_center)
print(f"-I- Field of view =", FoV_deg, "deg")
print(f"-I- frequency =", frequency)
print(f"-I- SNR =", SNR)
print(f"-I- fs =", fs)
print(f"-I- N_pix =", N_pix)
print(f"-I- px_w = {px_w:d}, px_h = {px_h:d}")
print(f"-I- N_levels =", N_levels)
print(f"-I- eps =", eps)
print(f"-I- OMP_NUM_THREADS =", os.getenv('OMP_NUM_THREADS'))


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
N_eig, intensity_intervals = I_est.infer_parameters()
ifpe_e = time.time()
print(f"#@#IFPE {ifpe_e - ifpe_s:.3f} sec")

# Imaging
ifim_s = time.time()
imager = bluebild.NufftSynthesis(
    ctx,
    N_antenna,
    N_station,
    intensity_intervals.shape[0],
    ["LSQ", "SQRT"],
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    precision,
    eps)

for t in times:
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, uvw, S.data)

lsq_image  = imager.get("LSQ").reshape((-1, N_pix, N_pix))
sqrt_image = imager.get("SQRT").reshape((-1, N_pix, N_pix))

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
sensitivity_intervals = np.array([[0, np.finfo("f").max]])
imager = None  # release previous imager first to some additional memory
imager = bluebild.NufftSynthesis(
    ctx,
    N_antenna,
    N_station,
    sensitivity_intervals.shape[0],
    ["INV_SQ"],
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    precision,
    eps
)

# Imaging
sfim_s = time.time()
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    imager.collect(N_eig, wl, sensitivity_intervals, W.data, XYZ.data, uvw, None)

sensitivity_image = imager.get("INV_SQ").reshape((-1, N_pix, N_pix))

I_lsq_eq  = s2image.Image(lsq_image  / sensitivity_image, xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, xyz_grid)

print(I_lsq_eq.data)

sfim_e = time.time()
print(f"#@#SFIM {sfim_e - sfim_s:.3f} sec")

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e - jkt0_s:.3f} sec\n")


#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)


bipptb.dump_data(I_lsq_eq.data, 'I_lsq_eq_data', args.outdir)
bipptb.dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid', args.outdir)


### Plotting section
plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'BIPP LSQ, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {FoV_deg} degrees.')
fp = "I_lsq.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)


plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'BIPP SQRT, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {FoV_deg} degrees.')
fp = "I_sqrt.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

