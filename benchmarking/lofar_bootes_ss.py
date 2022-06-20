# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (StandardSynthesis).
"""
import argparse
import time as stime
import timing

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
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.instrument as instrument

parser = argparse.ArgumentParser(description='Simulated LOFAR imaging with Bluebild (StandardSynthesis).')
parser.add_argument("timestep")
args = parser.parse_args()

timer = timing.Timer()

timer.start_time("Total")
timer.start_time("Setup")
# Observation
timer.start_time("Observation setup")
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
FoV, frequency = np.deg2rad(5), 145e6
wl = constants.speed_of_light / frequency
timer.end_time("Observation setup")

# Instrument
timer.start_time("Instrument setup")
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()
timer.end_time("Instrument setup")

# Data generation
timer.start_time("Data generation")
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=50)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
time = obs_start + (T_integration * u.s) * np.arange(3595)
timer.end_time("Data generation")

# Imaging
timer.start_time("Imaging setup")
N_level = 4
N_bits = 32
_, _, px_colat, px_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=FoV
)
px_grid = transform.pol2cart(1, px_colat, px_lon)

timer.end_time("Imaging setup")
timer.end_time("Setup")

imaging_timesteps = time[::int(args.timestep)]
estimation_timesteps = time[::200]
print( "Processing",len(imaging_timesteps), "time iterations" )

### Intensity Field ===========================================================
# Parameter Estimation
timer.start_time("Intensity field parameter estimation")
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t in ProgressBar(estimation_timesteps):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
timer.end_time("Intensity field parameter estimation")

# Imaging
timer.start_time("Intensity field imaging")
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
I_mfs.set_timer(timer)
for t in ProgressBar(imaging_timesteps):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)

    D, V, c_idx = I_dp(S, XYZ, W, wl)

    timer.start_time("Intensity field imager call")
    __  = I_mfs(D, V, XYZ.data, W.data, c_idx)
    timer.end_time("Intensity field imager call")

I_std, I_lsq = I_mfs.as_image()
timer.end_time("Intensity field imaging")

### Sensitivity Field =========================================================
# Parameter Estimation
timer.start_time("Sensitivity field parameter estimation")
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(estimation_timesteps):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()
timer.end_time("Sensitivity field parameter estimation")

# Imaging
timer.start_time("Sensitivity field imaging")
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
S_mfs.set_timer(timer)
for t in ProgressBar(imaging_timesteps):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    D, V = S_dp(G)

    __ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
    #stats= S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()
timer.end_time("Sensitivity field imaging")

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
ax[0].set_title("Bluebild Standardized Image")

I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1])
ax[1].set_title("Bluebild Least-Squares Image")
fig.savefig("test_ss.png")
fig.show()
plt.show()

timer.end_time("Total")
print(timer.summary())

tsumfile = open("lofar_bootes_ss_timing_timestep{0}.txt".format(args.timestep),'w')
tsumfile.write(timer.summary())

