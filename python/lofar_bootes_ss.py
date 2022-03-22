# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Real-data LOFAR imaging with Bluebild (StandardSynthesis).
"""

from tqdm import tqdm as ProgressBar
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import os
from pathlib import Path

import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set


# Instrument
N_station = 24
datasets_dir = Path.joinpath(Path(__file__).absolute().parents[1], "datasets")
if not os.path.isdir(datasets_dir):
    print(f"Fatal  : datasets_dir {datasets_dir} not existing!")
ms_file = Path.joinpath(datasets_dir, "gauss4/gauss4_t201806301100_SBL180.MS").as_posix()
ms = measurement_set.LofarMeasurementSet(ms_file, N_station)
gram = bb_gr.GramBlock()

# Observation
FoV = np.deg2rad(5)
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)


# Imaging
N_level = 4
N_bits = 32
maxtime = 100
_, _, px_colat, px_lon = grid.equal_angle(
    N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV
)
px_grid = transform.pol2cart(1, px_colat, px_lon)
print(px_grid.size)

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
    ms.visibilities(
        channel_id=[channel_id], time_id=slice(None, None, 200), column="DATA"
    )
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
c= 0 
for t, f, S in ProgressBar(
    ms.visibilities(channel_id=[channel_id], time_id=slice(None, maxtime, 1), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
    c += 1
    if c > 10: break
I_std, I_lsq = I_mfs.as_image()

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::200]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
c=0
for t, f, S in ProgressBar(
    ms.visibilities(channel_id=[channel_id], time_id=slice(None, maxtime, 1), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(G)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
    c += 1
    if c > 10: break
_, S = S_mfs.as_image()

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
Img = s2image.Image(I_std.data, I_std.grid)
I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
ax[0].set_title("Bluebild Standardized Image")

I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1])
ax[1].set_title("Bluebild Least-Squares Image")
fig.show()
plt.show()
# write all of this to a fits file
print ("Shapes:",I_std_eq.shape, Img.shape, I_std.data.shape )
I_std_eq.to_fits('bluebild_ss_image1_gauss4.fits')
Img.to_fits('bluebild_ss_image2_gauss4.fits')
