# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (StandardSynthesis).
"""

import os
import sys
from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
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


np.random.seed(0)

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bluebild.Context("AUTO")

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
precision  = "double"
time_slice = 200
times = times[::time_slice]

# Grids
lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)
px_w = xyz_grid.shape[1]
px_h = xyz_grid.shape[2]
xyz_grid = xyz_grid.reshape(3, -1)

print(f"-I- precision =", precision)
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
print(f"-I- OMP_NUM_THREADS =", os.getenv('OMP_NUM_THREADS'))


### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_levels, sigma=0.95)
for t in ProgressBar(times):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)
    I_est.collect(S, G)
N_eig, intensity_intervals = I_est.infer_parameters()

# Imaging
imager = bluebild.StandardSynthesis(
    ctx,
    N_antenna,
    N_station,
    intensity_intervals.shape[0],
    ["LSQ", "STD"],
    xyz_grid[0],
    xyz_grid[1],
    xyz_grid[2],
    precision)

for t in ProgressBar(times):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, S.data)

I_lsq = imager.get("LSQ").reshape((-1, px_w, px_h))
I_std = imager.get("STD").reshape((-1, px_w, px_h))


### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(times):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
sensitivity_intervals = np.array([[0, np.finfo("f").max]])
imager = None  # release previous imager first to some additional memory
imager = bluebild.StandardSynthesis(
    ctx,
    N_antenna,
    N_station,
    sensitivity_intervals.shape[0],
    ["INV_SQ"],
    xyz_grid[0],
    xyz_grid[1],
    xyz_grid[2],
    precision,
)

for t in ProgressBar(times):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    imager.collect(N_eig, wl, sensitivity_intervals, W.data, XYZ.data)

sensitivity_image = imager.get("INV_SQ").reshape((-1, px_w, px_h))


### Plot results
fig, ax = plt.subplots(ncols=2)
I_std_eq = s2image.Image(I_std / sensitivity_image, px_grid.reshape(3, px_w, px_h))
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
ax[0].set_title("BIPP STD Image")

I_lsq_eq = s2image.Image(I_lsq / sensitivity_image, px_grid.reshape(3, px_w, px_h))
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1])
ax[1].set_title("BIPP LSQ Image")
fig.savefig("test.png")
fig.show()
plt.show()
