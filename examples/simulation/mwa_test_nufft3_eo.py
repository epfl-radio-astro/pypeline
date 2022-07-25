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
from  pypeline.util.frame import xyz_to_uvw, xyz_at_latitude
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
import sys


#source = coord.SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')


# From https://ska-telescope.gitlab.io/external/rascil/_modules/rascil/processing_components/simulation/configurations.html#create_configuration_from_file
#low_location = coord.EarthLocation(
#    lon=116.76444824 * u.deg, lat=-26.824722084 * u.deg, height=300.0
#)

ms_file = "/work/ska/MWA_1086366992/1086366992.ms"
          #/work/ska/MWA_1086366992/1086366992.ms
cl_WCS = ifits.wcs("/work/ska/MWA_1086366992/wsclean-mwa-chan100-image.fits")
ms = measurement_set.MwaMeasurementSet(ms_file) # stations 1 - N_station 

use_ms = False
use_raw_vis = True
do3D = True
doPlan = False
ITRS_XYZ = True

for use_ms in True, False:
#for use_ms in False,:

    print("####### use_ms =", use_ms)
    read_coords_from_ms = use_ms
    uvw_from_ms         = use_ms

    outfilename = 'test_mwa_nufft_' + ('msUVW_' if uvw_from_ms else '') + ('rawVis_' if use_raw_vis else 'BBVis')

    gram = bb_gr.GramBlock()

    cl_WCS = cl_WCS.sub(['celestial'])
    width_px, height_px= 2*cl_WCS.wcs.crpix 
    cdelt_x, cdelt_y = cl_WCS.wcs.cdelt 
    FoV = np.deg2rad(abs(cdelt_x*width_px) )

    source = ms.field_center
        
    print("Reading {0}\n".format(ms_file))
    print("FoV is ", np.rad2deg(FoV))

    channel_id = 100
    #frequency = 1e8
    freq_ms = ms.channels["FREQUENCY"][channel_id]
    frequency = freq_ms.to_value(u.Hz)
    wl = constants.speed_of_light / frequency
    #print(freq_ms.to_value(u.Hz), frequency)
    #assert freq_ms.to_value(u.Hz) == frequency
    obs_start, obs_end = ms.time["TIME"][[0, -1]]
    print("obs start: {0}, end: {1}".format(obs_start, obs_end))

    # Imaging Parameters
    #t1 = tt.time()
    N_level = 1

    ### Intensity Field ===========================================================
    # Parameter Estimation
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
    for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, 20), column="DATA"):
        print("t = ", t)
        wl = constants.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)
        W = ms.beamformer(XYZ, wl)
        G = gram(XYZ, W, wl)
        S, _ = measurement_set.filter_data(S, W)
        I_est.collect(S, G)

    N_eig, c_centroid = I_est.infer_parameters()
    #print(N_eig, c_centroid)

    # Imaging
    I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
    UVW_baselines = []
    gram_corrected_visibilities = []

    baseline_rescaling = 2 * np.pi / wl

    for t, f, S, uvw in ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, None), column="DATA", return_UVW=True):
        
        wl = constants.speed_of_light / f.to_value(u.Hz)
        print(f.to_value(u.Hz), wl)


        # ITRF position of antennae
        XYZ_TF = np.asarray(ms._instrument._layout)

        #print(XYZ_TF, type(XYZ_TF), XYZ_TF.shape)
        #print("BB ITRF first station ", XYZ_TF[0,:])

        #EO: to be exact, needs low_location (as defined in RASCIL)
        #    but sufficiently well approximated with mean "geocentric" coordinates of array
        #  : RASCIL stores as ITRF crd geocentric 

        # Computing h0, the hour angle of the source from local meridian
        # Hour Angle of star = Local Sidereal Time- Right Ascension of star
        if ITRS_XYZ:
            # Field center coordinates
            field_center_lon, field_center_lat = source.data.lon.rad, source.data.lat.rad
            field_center_xyz = source.cartesian.xyz.value

            # UVW reference frame
            w_dir = field_center_xyz
            u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
            v_dir = np.array(
                [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
                 np.cos(field_center_lat)])
            uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)
            UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
            bsl__uvw  = (UVW[:, None, :] - UVW[None, ...])

        else:
            lst = atime.Time(t, scale='utc', location=ms.location).sidereal_time('mean')
            print(f"source: {source}")
            print(f"   lst: {lst}")
            h0 = lst - source.ra
            print(f"BB h0 = {h0.deg:.3f}; EO: this should match the \"time\" used in RASCIL input for the simulation")

            cos_h0, sin_h0 = np.cos(h0.rad), np.sin(h0.rad)
            cos_dec, sin_dec = np.cos(source.dec.rad), np.sin(source.dec.rad)

            uvw_frame = np.array([[           sin_h0,            cos_h0,       0],
                                  [-sin_dec * cos_h0,  sin_dec * sin_h0, cos_dec],
                                  [ cos_dec * cos_h0, -cos_dec * sin_h0, sin_dec]])
            #uvw_frame = np.array([[           cos_h0,           -sin_h0,       0],
            #                      [ sin_dec * sin_h0,  sin_dec * cos_h0, cos_dec],
            #                      [-cos_dec * sin_h0, -cos_dec * cos_h0, sin_dec]])

            #EO: flip w dir to match the reference, but needs justification!
            print("uvw_frame\n", uvw_frame)
            uvw_frame[:,2] *= -1
            print("uvw_frame\n", uvw_frame)
        


            # Step 1: recover baseline ENU (RASCIL stores geocentric center + ENU from config)
            # todo: check how to recover "center" in MS table if written by RASCIL

            # Geocentric position of array center
            ACX, ACY, ACZ = np.mean(XYZ_TF, axis=0)
            AC = coord.EarthLocation.from_geocentric(x=ACX, y=ACY, z=ACZ, unit=u.meter)
            print(f"Array enter ITRF = {AC.geocentric}")

            # Geodetic coordinates of array's center
            ACLON, ACLAT, ACHGT = AC.to_geodetic()
            print(f"Array center lon, lat, height = {ACLON:.5f}, {ACLAT:.5f}, {ACHGT:.2f}")
            ACLON, ACLAT = ACLON.to(u.rad), ACLAT.to(u.rad)
            print(f"Array center lon, lat, height = {ACLON:.5f}, {ACLAT:.5f}, {ACHGT:.2f}")

            # Weird/bug?
            ENU__ = XYZ_TF.data - np.array([AC.x.value, AC.y.value, AC.z.value]) 
            bsl__enu = ENU__[:, None, :] - ENU__[None, ...]
            #print(f"bsl__neu (BB ENU reconstructed)\n {bsl__enu} {bsl__enu.shape}")
            NANT, _ = XYZ_TF.data.shape

            # Baselines enu -> xyz -> uvw
            bsl__uvw = np.zeros(bsl__enu.shape)
            for i in range(0, NANT):
                for j in range(0, NANT):
                    tmp = xyz_at_latitude(bsl__enu[i,j,:], ACLAT.rad)
                    bsl__uvw[i,j,:] = xyz_to_uvw(tmp, h0.rad, source.dec.rad)
            print(f"bsl__uvw BB\n {bsl__uvw}")

            print(f"uvw bsl MS\n {uvw}")

        # Imaging grid
        lim = np.sin(FoV / 2)
        N_pix = 256
        pix_slice = np.linspace(-lim, lim, N_pix)
        Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
        Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
        lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
        pix_xyz = np.tensordot(uvw_frame.transpose(), lmn_grid, axes=1)  #EO: check this one!
        

        if uvw_from_ms:
            print("BB using MS")
            UVW_baselines_t = uvw * -1 #EO revert *= -1 in measurement_set.py (check why it is needed)
        else:
            print("BB on his own")
            UVW_baselines_t = bsl__uvw

        UVW_baselines.append(baseline_rescaling * UVW_baselines_t)
        
        ###ICRS_baselines.append(baseline_rescaling * ICRS_baselines_t)
        W = ms.beamformer(XYZ, wl)
        #print(f"W {W.shape}\n", W)

        S, _ = measurement_set.filter_data(S, W)

        D, V, c_idx = I_dp(S, XYZ, W, wl)
        W = W.data
        S_corrected = (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
        #print('shapes', S_corrected.shape, S.shape)

        if use_raw_vis:
            gram_corrected_visibilities.append(S.data)
        else:
            gram_corrected_visibilities.append(S_corrected)
            #print("UVW_baselines_t",UVW_baselines_t)

    UVW_baselines = np.stack(UVW_baselines, axis=0)
    gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=0).reshape(-1)
    
    UVW_baselines=UVW_baselines.reshape(-1,3)
    w_correction = np.exp(1j * UVW_baselines[:, -1])

    if not use_raw_vis:
        gram_corrected_visibilities *= w_correction

    lmn_grid = lmn_grid.reshape(3, -1)
    grid_center = lmn_grid.mean(axis=-1)
    lmn_grid -= grid_center[:, None]
    lmn_grid = lmn_grid.reshape(3, -1)

    UVW_baselines = 2 * np.pi * UVW_baselines.T.reshape(3, -1) / wl
    #UVW_baselines =  UVW_baselines.T.reshape(3, -1) 

    if do3D:
        outfilename += "3D"
        if doPlan:
            outfilename += "_plan"
            plan = finufft.Plan(nufft_type=3, n_modes_or_dim=3, eps=1e-4, isign=1)       
            plan.setpts(x= UVW_baselines[0], y=UVW_baselines[1], z=UVW_baselines[2],
                        s=lmn_grid[0], t=lmn_grid[1],u=lmn_grid[2])
            V = gram_corrected_visibilities #*prephasing
            print('V', V)
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
    #print(f'Elapsed time: {t2 - t1} seconds.')

    plt.figure()
    ax = plt.gca()
    I_lsq_eq.draw( ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
    ax.set_title(outfilename)
    
    plt.savefig(outfilename)
    #plt.show()

    gaussian=np.exp(-(Lpix ** 2 + Mpix ** 2)/(4*lim))
    gridded_visibilities=np.sqrt(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian*bb_image)))))
    gridded_visibilities[int(gridded_visibilities.shape[0]/2)-2:int(gridded_visibilities.shape[0]/2)+2, int(gridded_visibilities.shape[1]/2)-2:int(gridded_visibilities.shape[1]/2)+2]=0
    #plt.figure()
    #plt.imshow(np.flipud(gridded_visibilities), cmap='cubehelix')
    #plt.show()
    
