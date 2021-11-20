# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (StandardSynthesis).
"""

import os, sys, argparse
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
import time as tt


# Dump data to args.outdir if defined
def dump_data(stats, filename):
    if args.outdir:
        fp = os.path.join(args.outdir, filename + '.npy')
        with open(fp, 'wb') as f:
            np.save(f, stats)
            print("Wrote ", fp)

jkt0_s = tt.time()


# Check arguments
parser = argparse.ArgumentParser()
parser.add_argument("--outdir",   help="Path to dumping location (no dumps if not set)")
args = parser.parse_args()
if args.outdir:
    if not os.path.exists(args.outdir):
        print('fatal: --outdir ('+args.outdir+') must exists if defined.')
        sys.exit(1)
    print("Dumping directory: ", args.outdir)        
else:
    print("Will not dump anything, --outdir not set.")


# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV, frequency = np.deg2rad(8), 145e6 #EO was 5
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
print(time.size)

# Imaging
N_level = 3 #4
N_bits = 32 # 32
time_slice = 200 #36 #25
_, _, px_colat, px_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=FoV
)

px_grid = transform.pol2cart(1, px_colat, px_lon)
print("px_grid=",px_grid.shape)

#sys.exit(0)

t1 = tt.time()

### Intensity Field ===========================================================
# Parameter Estimation
ifpe_s = tt.time()
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
idx = 0
for t in time[::200]:  #ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)
    I_est.collect(S, G)
    idx += 1

N_eig, c_centroid = I_est.infer_parameters()
ifpe_e = tt.time()
print(f"#@#IFPE {ifpe_e-ifpe_s:.3f} sec")


# Imaging
ifim_s = tt.time()
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
#for t in ProgressBar(time[::time_slice]):
for t in time[::time_slice]:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)
    D, V, c_idx = I_dp(S, G)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)

I_std, I_lsq = I_mfs.as_image()
ifim_e = tt.time()
print(f"#@#IFIM {ifim_e-ifim_s:.3f} sec")

### Sensitivity Field =========================================================
# Parameter Estimation
sfpe_s = tt.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
#for t in ProgressBar(time[::200]):
for t in time[::200]:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()
sfpe_e = tt.time()
print(f"#@#SFPE {sfpe_e-sfpe_s:.3f} sec")

# Imaging
sfim_s = tt.time()
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
#for t in ProgressBar(time[::time_slice]):
for t in time[::time_slice]:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    D, V = S_dp(G)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()

I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
dump_data(I_lsq_eq.data, 'I_lsq_eq_data')
dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid')

sfim_e = tt.time()
print(f"#@#SFIM {sfim_e-sfim_s:.3f} sec")

t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

jkt0_e = tt.time()
print(f"#@#TOT {jkt0_e-jkt0_s:.3f} sec\n")

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
ax[0].set_title("Bluebild Standardized Image")

#I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1])
ax[1].set_title("Bluebild Least-Squares Image")
fp = "test.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

fig.show()
plt.show()
