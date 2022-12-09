# #############################################################################
# lofar_bootes_ps.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (PeriodicSynthesis).
"""
import argparse
import time as stime

from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.instrument as instrument

parser = argparse.ArgumentParser(description='Simulated LOFAR imaging with Bluebild (PeriodicSynthesis).')
parser.add_argument("timestep")
args = parser.parse_args()

t_start = stime.process_time()

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
FoV, frequency = np.deg2rad(5), 145e6
wl = constants.speed_of_light / frequency

t_obssetup = stime.process_time()

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()

t_instsetup = stime.process_time()

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=20)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

t_datagen = stime.process_time()

# Imaging
N_level = 4
N_bits = 32
R = dev.icrs2bfsf_rot(obs_start, obs_end)
_, _, pix_colat, pix_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl),
    direction=R @ field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
    FoV=FoV,
)
N_FS, T_kernel = dev.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(10)

t_imgsetup = stime.process_time()

t_setup = stime.process_time()
print("Setup time:", t_setup - t_start )
print("  Observation setup time:", t_obssetup - t_start )
print("  Instrument setup time:", t_instsetup -t_obssetup )
print("  Data generation time:", t_datagen -t_obssetup  )
print("  Imaging setup time:", t_imgsetup - t_datagen )

imaging_timesteps = time[::int(args.timestep)]
estimation_timesteps = time[::200]
print( "Using",len(estimation_timesteps), "time iterations to estimate parameters" )
print( "Processing",len(imaging_timesteps), "time iterations for imaging" )

### Intensity Field ===========================================================
t_ifield = stime.process_time()
# Parameter Estimation

I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t in ProgressBar(estimation_timesteps):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

t_ifield_param = stime.process_time()
("Intensity field parameter estimation: ", t_ifield_param - t_ifield)

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
t_ifi_iteration = 0
t_ifi_iteration_data = 0
t_ifi_iteration_dp = 0
t_ifi_iteration_mfs = 0
for t in ProgressBar(imaging_timesteps):
    t_ifi_start = stime.process_time()
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    t_ifi_iteration_data += stime.process_time() - t_ifi_start

    t = stime.process_time()
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    t_ifi_iteration_dp += stime.process_time() - t

    t = stime.process_time()
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
    t_ifi_iteration_mfs += stime.process_time() - t

    t_ifi_end = stime.process_time()
    t_ifi_iteration += t_ifi_end - t_ifi_start
I_std, I_lsq = I_mfs.as_image()
t_ifield_image = stime.process_time()
("Intensity field imaging: ", t_ifield_image  - t_ifield_param )

### Sensitivity Field =========================================================
t_sfield = stime.process_time()
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(estimation_timesteps):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

t_sfield_param = stime.process_time()
("Sensitivity field parameter estimation: ", t_sfield_param - t_sfield)

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
t_sfi_iteration = 0
t_sfi_iteration_data = 0
t_sfi_iteration_dp = 0
t_sfi_iteration_mfs = 0
for t in ProgressBar(imaging_timesteps):
    t_sfi_start = stime.process_time()
    XYZ = dev(t)
    W = mb(XYZ, wl)
    t_sfi_iteration_data += stime.process_time() - t_sfi_start

    t = stime.process_time()
    D, V = S_dp(XYZ, W, wl)
    t_sfi_iteration_dp += stime.process_time() - t

    t = stime.process_time()
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
    t_sfi_iteration_mfs += stime.process_time() - t

    t_sfi_end = stime.process_time()
    t_sfi_iteration += t_sfi_end - t_sfi_start
_, S = S_mfs.as_image()

t_sfield_image = stime.process_time()
("Sensitivity field imaging: ", t_sfield_image  - t_sfield_param )

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
ax[0].set_title("Bluebild Standardized Image")

I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1])
ax[1].set_title("Bluebild Least-Squares Image")
fig.savefig("test_ps.png")
fig.show()
plt.show()

t_end = stime.process_time()

tsumfile = open("lofar_bootes_ps_timing_timestep{0}.txt".format(args.timestep),'w')
outtext = [ "Timing Summary",
            "Total: {0:.2f} s".format(t_end - t_start),
            "Setup: {0:.2f} s".format(t_setup - t_start ),
            "  Observation setup: {0:.2f}".format(t_obssetup - t_start ),
            "  Instrument setup: {0:.2f}".format(t_instsetup -t_obssetup ),
            "  Data generation: {0:.2f}".format(t_datagen -t_obssetup  ),
            "  Imaging setup: {0:.2f}".format(t_imgsetup - t_datagen ),
            "Using {0} time iterations to estimate parameters".format(len(estimation_timesteps)) ,
            "Processing {0} time iterations for imaging".format(len(imaging_timesteps)) ,
            "Imaging:{0:.2f} ".format(t_sfield_image  - t_ifield ),
            "  Intensity field parameter estimation: {0:.2f}".format(t_ifield_param - t_ifield),
            "  Intensity field imaging: {0:.2f}".format(t_ifield_image  - t_ifield_param ),
            "    Iterating: {0:.2f} ({1:.2f} s per iteration)".format( t_ifi_iteration, t_ifi_iteration/len(imaging_timesteps)),
            "      Data prep: {0:.2f} ({1:.2f} s per iteration)".format(t_ifi_iteration_data,  t_ifi_iteration_data/len(imaging_timesteps)),
            "      Data processing: {0:.2f} ({1:.2f} s per iteration)".format(t_ifi_iteration_dp,  t_ifi_iteration_dp/len(imaging_timesteps)),
            "      Imaging: {0:.2f} ({1:.2f} s per iteration)".format(t_ifi_iteration_mfs,  t_ifi_iteration_mfs/len(imaging_timesteps)),
            "  Sensitivity field parameter estimation: {0:.2f}".format(t_sfield_param - t_sfield),
            "  Sensitivity field imaging: {0:.2f}".format(t_sfield_image  - t_sfield_param ),
            "    Iterating: {0:.2f} s ({1:.2f} s per iteration)".format( t_sfi_iteration, t_sfi_iteration/len(imaging_timesteps)),
            "      Data prep: {0:.2f} s ({1:.2f} s per iteration)".format(t_sfi_iteration_data,  t_sfi_iteration_data/len(imaging_timesteps)),
            "      Data processing:{0:.2f} s ({1:.2f} s per iteration)".format(t_sfi_iteration_dp,  t_sfi_iteration_dp/len(imaging_timesteps)),
            "      Imaging: {0:.2f} s ({1:.2f} s per iteration)".format(t_sfi_iteration_mfs,  t_sfi_iteration_mfs/len(imaging_timesteps)),
          ]
for t in outtext:
    print(t)
    tsumfile.write(t+"\n")


