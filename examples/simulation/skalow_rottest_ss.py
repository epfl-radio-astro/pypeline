"""
Simulated SKA-LOW imaging with Bluebild (PeriodicSynthesis).
Compare Bluebild image with WSCLEAN image.
"""

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import EarthLocation
from astroplan import Observer
import imot_tools.io.fits as ifits
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import scipy.constants as constants
import sys, time
import finufft
import pypeline.util.frame as frame

import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
#import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set
import imot_tools.math.sphere.interpolate as interpolate
import imot_tools.math.sphere.transform as transform
#import pycsou.linop as pyclop
#from imot_tools.math.func import SphericalDirichlet
import joblib as job
import math

start_time = time.process_time()

angle_deg = 0
use_fits_coords = False
use_uvw = True
rotation = np.deg2rad(angle_deg)*u.rad
out_str = "skalow_small-testng_uvw{1}_{2}".format(angle_deg, use_uvw, 'wcs' if use_fits_coords else 'rot{0}'.format(angle_deg))
# Instrument
cl_WCS = ifits.wcs("/work/ska/results_rascil_skalow_small/wsclean-image.fits")
ms_file = "/work/ska/results_rascil_skalow_small/ska-pipeline_simulation.ms"
ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
print("Reading {0}\n".format(ms_file))

gram = bb_gr.GramBlock()

# Observation
FoV = np.deg2rad(10)
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
#sky_model = source.from_tgss_catalog(ms.field_center, FoV, N_src=4)
obs_start, obs_end = ms.time["TIME"][[0, -1]]

print("obs start: {0}, end: {1}".format(obs_start, obs_end))
print("field center", ms.field_center, ms.field_center.cartesian.xyz)

#field_center_rot = ms.field_center.spherical_offsets_by(d_lon = rotation, d_lat = 0 * u.rad)
field_center_rot = coord.SkyCoord(ms.field_center.ra + angle_deg*u.deg, ms.field_center.dec, unit="deg")
print("field rotated", field_center_rot, field_center_rot.cartesian.xyz)
ms._field_center = field_center_rot
location = EarthLocation(lon=116.76444824 * u.deg, lat=-26.824722084 * u.deg, height=300.0)
print("telescope location", location.x, location.y, location.z)

uvw_frame = frame.uvw_basis(ms.field_center)

# Imaging
N_level = 1
N_bits = 32

if use_fits_coords:
    cl_WCS = cl_WCS.sub(['celestial'])
    cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10))) # downsample!
    print(cl_WCS)

    px_grid = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
    N_cl_lon, N_cl_lat = px_grid.shape[-2:]
    print('using grid', px_grid.shape, px_grid[0,0,0], '--', px_grid[0,-1,-1])
    print('                       ', px_grid[1,0,0], '--', px_grid[1,-1,-1])
    print('                       ', px_grid[2,0,0], '--', px_grid[2,-1,-1])
    #x = np.linspace( px_grid[0,0,0],  px_grid[0,-1,-1], N_cl_lon)
    #y = np.linspace( px_grid[1,0,0],  px_grid[1,-1,-1], N_cl_lat)
    #xv, yv = np.meshgrid(x, y)
    #px_grid[0,:,:] = xv
    #px_grid[1,:,:] = yv
    #px_grid[2,:,:] = px_grid[2,0,0] + np.zeros(px_grid[2,:,:].shape)
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 8))
    for i in range(3):
        ax[i].imshow(px_grid[i,:,:])
    plt.savefig(out_str + "_grid")


else:
    # Imaging grid
    lim = np.sin(FoV / 2)
    N_pix = 256
    pix_slice = np.linspace(-lim, lim, N_pix)
    Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
    Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
    lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
    px_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
    #px_grid = lmn_grid
    _, pix_lat, pix_lon = transform.cart2eq(*px_grid)

    '''_, _, px_colat, px_lon = grid.equal_angle(
        N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV
    )
    print(px_colat.shape, px_lon.shape)
    px_grid = transform.pol2cart(1, px_colat, px_lon)
    N_cl_lon, N_cl_lat = px_grid.shape[-2:]'''

    print('using grid', px_grid.shape, px_grid[0,0,0], '--', px_grid[0,-1,-1])
    print('                       ', px_grid[1,0,0], '--', px_grid[1,-1,-1])
    print('                       ', px_grid[2,0,0], '--', px_grid[2,-1,-1])
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 8))
    for i in range(3):
        ax[i].imshow(px_grid[i,:,:])
    plt.savefig(out_str + "_grid")
print("Grid size is:", px_grid.shape)
#px_grid = px_grid[:,::10,::10]# downsample!
print("Downsampled to:", px_grid.shape)
#sys.exit()

print("===================")

site = Observer(location=location)
ha = site.target_hour_angle(obs_start, ms.field_center).wrap_at("180d")
dec = ms.field_center.dec.rad
lat = location.geodetic[1].to("rad").value

'''
import math
import numpy as np
x = -2566715.7634307873
y = 5081138.507217909
r = math.sqrt(x**2 + y**2)
def rot(a):
     return x*np.cos(a*numpy.pi/180) - y*np.sin(a*numpy.pi/180)
'''

x, y, z = np.array(ms.instrument._layout['X']),np.array(ms.instrument._layout['Y']), np.array(ms.instrument._layout['Z'])
x0, y0, z0 = x[0], y[0], z[0]
# Format antenna positions and VLA center as EarthLocation.
angle1 = (90-116)*np.pi/180 
angle2 = (90 + 26)*np.pi/180
print(x0, y0, z0)

r = math.sqrt(x0**2 + y0**2)
print(r, x0/r, np.arccos(x0/r)*180/np.pi)

x1 = x0*np.cos(-angle1) - y0*np.sin(-angle1)
y1 = x0*np.sin(-angle1) + y0*np.cos(-angle1)
print(x1, y1, z0)


y2 = y1*np.cos(angle2) - z0*np.sin(angle2)
z2 = y1*np.sin(angle2) + z0*np.cos(angle2)

print(x1,y2,z2)

print("===================")

#px_grid = cl_pix_icrs
time_slice = 10


XYZ = ms.instrument.__call__(obs_start).data
print(XYZ[0,:])

UVW = (uvw_frame.transpose() @ XYZ.transpose()).transpose()
print(UVW[0,:])

print("HA", ha, ha.rad)


### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(
        ms.visibilities(
            channel_id=[channel_id], time_id=slice(0, None, 200), column="DATA"
        )
):
    wl = constants.speed_of_light / f.to_value(u.Hz)

    if use_uvw:
        #XYZ = ms.instrument.uvw(ha,dec,lat)
        #ms.instrument.uvw(ha,dec,lat)
        XYZ = ms.instrument.uvw2(t, location, ms.field_center)
    else:
        XYZ = ms.instrument(t, rotation)
    print('  uvw after call: ',XYZ.data[0,:])
    
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

print("N_eig:", N_eig)

# Imaging
print ("centroids = ", c_centroid)
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
#I_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, time_slice), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)

    if use_uvw:
        #XYZ = ms.instrument.uvw(ha,dec,lat)
        XYZ = ms.instrument.uvw2(t,location, ms.field_center)
    else:
        XYZ = ms.instrument(t, rotation)
    #XYZ = ms.instrument(t,rotation)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    print(c_idx)
    print(V.shape)
    #c_idx = [0,1,2,3]

    #_ = I_mfs(D, V, XYZ.data, W.data, c_idx)

    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.data)
    V_gpu  = cp.asarray(V)
    _ = I_mfs(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    
I_std, I_lsq = I_mfs.as_image()

end_time = time.process_time()
print("Time elapsed: {0}s".format(end_time - start_time))

'''### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time["TIME"][::200]):
    XYZ = ms.instrument(t,field_center = ms.field_center)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
print("Running sensitivity imaging")
# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
#S_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
for t, f, S in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, time_slice), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t,field_center = ms.field_center)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)
    D, V = S_dp(G)
    #_ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
    XYZ_gpu = cp.asarray(XYZ.data)
    W_gpu  = cp.asarray(W.data.toarray())
    V_gpu  = cp.asarray(V)
    _ = S_mfs(D, V_gpu, XYZ_gpu, W_gpu, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()'''

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=N_level, nrows=2, figsize=(16, 8))
#I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid) 
#I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid) 
I_std_eq = s2image.Image(I_std.data, I_std.grid) 
I_lsq_eq = s2image.Image(I_lsq.data, I_lsq.grid) 

for i in range(N_level):
    top_plot = ax[0] if N_level == 1 else ax[0,i]
    bottom_plot = ax[1] if N_level == 1 else ax[1,i]
    I_std_eq.draw(index=i, ax= top_plot)
    top_plot.set_title("Standardized Image Level = {0}".format(i))
    I_lsq_eq.draw(index=i, ax=bottom_plot)
    bottom_plot.set_title("Least-Squares Image Level = {0}".format(i))

plt.savefig(out_str)

# 5. Store the interpolated Bluebild image in standard-compliant FITS for view
# in AstroPy/DS9.
if use_fits_coords:
    f_interp = (I_lsq_eq.data  # We need to transpose axes due to the FORTRAN
                .reshape(N_level, N_cl_lon, N_cl_lat)  # indexing conventions of the FITS standard.
                .transpose(0, 2, 1))
    #f_interp = I_lsq_eq.data 
    #f_interp = np.rot90(f_interp, 2, axes=(1,2))
    #f_interp = np.flip(f_interp, axis=2)
    I_lsq_eq_interp = s2image.WCSImage(np.sum(f_interp,axis=0), cl_WCS)
    I_lsq_eq_interp.to_fits('bluebild_{0}_combined-test.fits'.format(out_str))
    I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
    I_lsq_eq_interp.to_fits('bluebild_{0}_levels-test.fits'.format(out_str))

    end_interp_time = time.process_time()

    print("Time to make BB image: {0}s".format(end_time - start_time))