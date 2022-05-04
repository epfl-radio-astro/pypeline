
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

start_time = time.process_time()

angle_deg = 0
use_fits_coords = True
use_uvw = True
rotation = np.deg2rad(angle_deg)*u.rad
out_str = "skalow_small_uvw{1}_{2}".format(angle_deg, use_uvw, 'wcs' if use_fits_coords else 'rot{0}'.format(angle_deg))
# Instrument
cl_WCS = ifits.wcs("/work/ska/results_rascil_skalow_small/wsclean-image.fits")
ms_file = "/work/ska/results_rascil_skalow_small/ska-pipeline_simulation.ms"
ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
print("Reading {0}\n".format(ms_file))

gram = bb_gr.GramBlock()

# Observation
FoV = np.deg2rad(5)
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

# Imaging
N_level = 4
N_bits = 32

if use_fits_coords:
    cl_WCS = cl_WCS.sub(['celestial'])
    cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10))) # downsample!
    px_grid = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
    N_cl_lon, N_cl_lat = px_grid.shape[-2:]
else:
    _, _, px_colat, px_lon = grid.equal_angle(
        N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV
    )
    px_grid = transform.pol2cart(1, px_colat, px_lon)
    N_cl_lon, N_cl_lat = px_grid.shape[-2:]
print("Grid size is:", px_grid.shape)
#px_grid = px_grid[:,::10,::10]# downsample!
print("Downsampled to:", px_grid.shape)

print("===================")


x, y, z = np.array(ms.instrument._layout['X']),np.array(ms.instrument._layout['Y']), np.array(ms.instrument._layout['Z'])

# Format antenna positions and VLA center as EarthLocation.
antpos_ap = coord.EarthLocation(x=x*u.m, y=y*u.m, z=z*u.m)
# Convert antenna pos terrestrial to celestial.  For astropy use
# get_gcrs_posvel(t)[0] rather than get_gcrs(t) because if a velocity
# is attached to the coordinate astropy will not allow us to do additional
# transformations with it (https://github.com/astropy/astropy/issues/6280)
tel_site_p, tel_site_v = location.get_gcrs_posvel(obs_start)
antpos_c_ap = coord.GCRS(antpos_ap.get_gcrs_posvel(obs_start)[0],
        obstime=obs_start, obsgeoloc=tel_site_p, obsgeovel=tel_site_v)

#frame_uvw = pointing_direction.skyoffset_frame() # ICRS
frame_uvw = ms.field_center.transform_to(antpos_c_ap).skyoffset_frame() # GCRS

# Rotate antenna positions into UVW frame.
antpos_uvw_ap = antpos_c_ap.transform_to(frame_uvw).cartesian

ant_uvw = np.array([antpos_uvw_ap.y,antpos_uvw_ap.z,antpos_uvw_ap.x]).T

print(ant_uvw.shape)
print(ant_uvw[0,:])

print("===================")

#px_grid = cl_pix_icrs
time_slice = 10


site = Observer(location=location)
ha = site.target_hour_angle(obs_start, ms.field_center).wrap_at("180d")
dec = ms.field_center.dec.rad
lat = location.geodetic[1].to("rad").value

print("HA", ha)


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
    c_idx = [0,1,2,3]

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
fig, ax = plt.subplots(ncols=N_level+1, nrows=2, figsize=(16, 8))
#I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid) 
#I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid) 
I_std_eq = s2image.Image(I_std.data, I_std.grid) 
I_lsq_eq = s2image.Image(I_lsq.data, I_lsq.grid) 

for i in range(N_level):
    I_std_eq.draw(index=i, ax=ax[0,i])
    ax[0,i].set_title("Standardized Image Level = {0}".format(i))
    I_lsq_eq.draw(index=i, ax=ax[1,i])
    ax[1,i].set_title("Least-Squares Image Level = {0}".format(i))
I_std_eq.draw(ax=ax[0,N_level])
ax[0,N_level].set_title("Standardized Image".format(i))
I_lsq_eq.draw(ax=ax[1,N_level])
ax[1,N_level].set_title("Least-Squares Image".format(i))

plt.savefig(out_str)

# 5. Store the interpolated Bluebild image in standard-compliant FITS for view
# in AstroPy/DS9.
sys.exit()
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