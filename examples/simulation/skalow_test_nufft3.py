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
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import scipy.sparse as sparse
import finufft
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.measurement_set as measurement_set
import imot_tools.io.fits as ifits
from imot_tools.math.func import SphericalDirichlet
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import time as tt

read_coords_from_ms = True

# Instrument
#ms_file = "/home/etolley/rascil_ska_sim/results_test/ska-pipeline_simulation.ms"
#ms_file = "/home/etolley/rascil/examples/pipelines/ska-pipelines/results_rascil_skalow_test/ska-pipeline_simulation.ms"
ms_file = "/work/ska/results_rascil_skalow_small/ska-pipeline_simulation.ms"
ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
gram = bb_gr.GramBlock()


if read_coords_from_ms:
    #cl_WCS = ifits.wcs("/home/etolley/rascil/examples/pipelines/ska-pipelines/results_rascil_skalow_test/imaging_dirty.fits")
    cl_WCS = ifits.wcs("/work/ska/results_rascil_skalow_small/wsclean-image.fits")
    cl_WCS = cl_WCS.sub(['celestial'])
    ##cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
    cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
    N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]

    width_px, height_px= 2*cl_WCS.wcs.crpix 
    cdelt_x, cdelt_y = cl_WCS.wcs.cdelt 
    FoV = np.deg2rad(abs(cdelt_x*width_px) )
    field_center = ms.field_center
else:
    field_center = coord.SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    #field_center = coord.SkyCoord(ra=90.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    FoV = np.deg2rad(5.5)
    

print("Reading {0}\n".format(ms_file))
print("FoV is ", np.rad2deg(FoV))

channel_id = 0
frequency = 1e8
wl = constants.speed_of_light / frequency
freq_ms = ms.channels["FREQUENCY"][channel_id]
assert freq_ms.to_value(u.Hz) == frequency
obs_start, obs_end = ms.time["TIME"][[0, -1]]
print("obs start: {0}, end: {1}".format(obs_start, obs_end))

# Field center coordinates

field_center_lon, field_center_lat = field_center.data.lon.rad, field_center.data.lat.rad
field_center_xyz = field_center.cartesian.xyz.value

# UVW reference frame
w_dir = field_center_xyz
u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
v_dir = np.array(
    [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
     np.cos(field_center_lat)])
uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)

# Imaging grid
lim = np.sin(FoV / 2)
N_pix = 256
pix_slice = np.linspace(-lim, lim, N_pix)
Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
pix_xyz = np.tensordot(uvw_frame, lmn_grid, axes=1)
_, pix_lat, pix_lon = transform.cart2eq(*pix_xyz)

# Imaging Parameters
t1 = tt.time()
N_level = 1
N_bits = 32
time_slice = 20

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, 200), column="DATA")
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
UVW_baselines = []
ICRS_baselines = []
gram_corrected_visibilities = []
baseline_rescaling = 2 * np.pi / wl

for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, None), column="DATA", return_UVW=False)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
    UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])
    ICRS_baselines_t = (XYZ.data[:, None, :] - XYZ.data[None, ...])
    UVW_baselines.append(baseline_rescaling * UVW_baselines_t)
    ICRS_baselines.append(baseline_rescaling * ICRS_baselines_t)
    W = ms.beamformer(XYZ, wl)
    #S = vis(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    W = W.data
    S_corrected = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    gram_corrected_visibilities.append(S_corrected)

UVW_baselines = np.stack(UVW_baselines, axis=0)
ICRS_baselines = np.stack(ICRS_baselines, axis=0).reshape(-1, 3)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=0).reshape(-1)

UVW_baselines=UVW_baselines.reshape(-1,3)
w_correction = np.exp(1j * UVW_baselines[:, -1])
gram_corrected_visibilities *= w_correction

lmn_grid = lmn_grid.reshape(3, -1)
grid_center = lmn_grid.mean(axis=-1)
lmn_grid -= grid_center[:, None]
lmn_grid = lmn_grid.reshape(3, -1)

#UVW_baselines = 2 * np.pi * UVW_baselines.T.reshape(3, -1) / wl
UVW_baselines =  UVW_baselines.T.reshape(3, -1) 
 

do3D=True
doPlan = True
outfilename = 'test_skalow_nufft_'

if do3D:
    outfilename += "3D"
    if doPlan:
        outfilename += "_plan"
        plan = finufft.Plan(nufft_type=3, n_modes_or_dim=3, eps=1e-4, isign=1)
        plan.setpts(x= UVW_baselines[0], y=UVW_baselines[1], z=UVW_baselines[2],
                                s=lmn_grid[0], t=lmn_grid[1],u=lmn_grid[2])
        V = gram_corrected_visibilities #*prephasing
        bb_image = np.real(plan.execute(V)) 
        bb_image = bb_image.reshape(pix_xyz.shape[1:])
    else:
        ##########################################################################################
        bb_image = finufft.nufft3d3(x= UVW_baselines[0],
                                y= UVW_baselines[1],
                                z= UVW_baselines[2],
                                s=lmn_grid[0],
                                t=lmn_grid[1],
                                u=lmn_grid[2],
                                c=gram_corrected_visibilities, eps=1e-4)
        bb_image = np.real(bb_image)
        bb_image = bb_image.reshape(pix_xyz.shape[1:])
        ##########################################################################################
else:
    outfilename += "2D"
    scaling = 2 * lim / N_pix  
    if doPlan:
        outfilename += "_plan"
        ##########################################################################################
        plan = finufft.Plan(nufft_type=1, n_modes_or_dim= (N_pix, N_pix), eps=1e-4, isign=1)
        plan.setpts(x=scaling * UVW_baselines[1], y=scaling * UVW_baselines[0])  
        V = gram_corrected_visibilities
        bb_image = np.real(plan.execute(V))  
        ########################################################################################## 
    else:
        ##########################################################################################
        bb_image = finufft.nufft2d1(x=scaling * UVW_baselines[1],
                                    y=scaling * UVW_baselines[0],
                                    c=gram_corrected_visibilities,
                                    n_modes=N_pix, eps=1e-4)
        bb_image = np.real(bb_image)
        ##########################################################################################

I_lsq_eq = s2image.Image(bb_image, pix_xyz)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

plt.figure()
ax = plt.gca()
I_lsq_eq.draw( ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax.set_title(outfilename)

plt.savefig(outfilename)

gaussian=np.exp(-(Lpix ** 2 + Mpix ** 2)/(4*lim))
gridded_visibilities=np.sqrt(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian*bb_image)))))
gridded_visibilities[int(gridded_visibilities.shape[0]/2)-2:int(gridded_visibilities.shape[0]/2)+2, int(gridded_visibilities.shape[1]/2)-2:int(gridded_visibilities.shape[1]/2)+2]=0
plt.figure()
plt.imshow(np.flipud(gridded_visibilities), cmap='cubehelix')

