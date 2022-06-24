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
#import matplotlib
#matplotlib.use('Qt5Agg')
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
import sys


ra  =  15.0 / 180 * np.pi
dec = -45.0 / 180 * np.pi

cos_ra,  sin_ra  = np.cos(ra),  np.sin(ra)
cos_dec, sin_dec = np.cos(dec), np.sin(dec)


use_ms = False
use_raw_vis = True
do3D = False
doPlan = False

#for use_ms in True, False:
for use_ms in False,:

    print("####### use_ms =", use_ms)
    read_coords_from_ms = use_ms
    uvw_from_ms         = use_ms

    outfilename = 'test_skalow_nufft_' + ('msUVW_' if uvw_from_ms else '') + ('rawVis_' if use_raw_vis else 'BBVis')

    # Instrument
    ms_file = "/work/ska/results_rascil_skalow_small/ska-pipeline_simulation.ms"
    #ms_file = "/work/ska/results_rascil_lofar/lofar-pipeline_simulation.ms"
    ms = measurement_set.SKALowMeasurementSet(ms_file) # stations 1 - N_station 
    gram = bb_gr.GramBlock()

    if read_coords_from_ms:
        cl_WCS = ifits.wcs("/work/ska/results_rascil_skalow_small/wsclean-image.fits")
        print(cl_WCS)
        cl_WCS = cl_WCS.sub(['celestial'])
        print("cl_WCS =", cl_WCS)
        ##cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
        cl_pix_icrs = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
        N_cl_lon, N_cl_lat = cl_pix_icrs.shape[-2:]
        
        width_px, height_px= 2*cl_WCS.wcs.crpix 
        cdelt_x, cdelt_y = cl_WCS.wcs.cdelt 
        FoV = np.deg2rad(abs(cdelt_x*width_px) )
        field_center = ms.field_center
        print("field_center", type(field_center), field_center)
    else:
        field_center = coord.SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame='icrs', equinox='J2000')
        #field_center = coord.SkyCoord(ra=90.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        FoV = np.deg2rad(5.56111)
        print("field_center", type(field_center), field_center)

    FoV =  np.deg2rad(8.0)


    print("Reading {0}\n".format(ms_file))
    print("FoV is ", np.rad2deg(FoV))
    #continue

    channel_id = 0
    frequency = 1e8
    #frequency = 145e6
    wl = constants.speed_of_light / frequency
    freq_ms = ms.channels["FREQUENCY"][channel_id]
    print(freq_ms.to_value(u.Hz))
    assert freq_ms.to_value(u.Hz) == frequency
    obs_start, obs_end = ms.time["TIME"][[0, -1]]
    print("obs start: {0}, end: {1}".format(obs_start, obs_end))

    # Field center coordinates
    field_center_lon, field_center_lat = field_center.data.lon.rad, field_center.data.lat.rad
    print("fov center lon, lat =", field_center_lon, field_center_lat)
    print("vs ", ra, dec) 
    field_center_xyz = field_center.cartesian.xyz.value
    print("field_center.cartesian.xyz.value", field_center.cartesian.xyz.value)
    #sys.exit(0)

    # UVW reference frame
    #w_dir = field_center_xyz * -1 #EO!!!!!!!!!!!!! with uvw from ms!
    w_dir = field_center_xyz #EO!!!!!!!!!!!!!
    u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
    v_dir = np.array(
        [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
         np.cos(field_center_lat)])
    uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)
    print("ORIGINAL uvw_frame\n", uvw_frame)

    """
    new_frame1 = np.array([[           sin_ra,            cos_ra, 0],
                           [-sin_dec * cos_ra,  sin_dec * sin_ra, cos_dec],
                           [ cos_dec * cos_ra, -cos_dec * sin_ra, sin_dec]])

    #uvw_frame = new_frame1
    print("uvw_frame\n", uvw_frame)
    """

    # Imaging Parameters
    #t1 = tt.time()
    N_level = 1

    ### Intensity Field ===========================================================
    # Parameter Estimation
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
    for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, 200), column="DATA"):
        print("t = ", t)
        wl = constants.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)
        W = ms.beamformer(XYZ, wl)
        G = gram(XYZ, W, wl)
        S, _ = measurement_set.filter_data(S, W)
        I_est.collect(S, G)

    #continue

    N_eig, c_centroid = I_est.infer_parameters()
    print(N_eig, c_centroid)

    # Imaging
    I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
    UVW_baselines = []
    ###ICRS_baselines = []
    gram_corrected_visibilities = []

    baseline_rescaling = 2 * np.pi / wl

    from astropy.coordinates import ICRS, ITRS

    for t, f, S, uvw in ms.visibilities(channel_id=[channel_id], time_id=slice(0, None, None), column="DATA", return_UVW=True):

        wl = constants.speed_of_light / f.to_value(u.Hz)
        print(f.to_value(u.Hz), wl)
        
        # uvw frame

        # Imaging grid
        lim = np.sin(FoV / 2)
        N_pix = 512#256
        pix_slice = np.linspace(-lim, lim, N_pix)
        #print("pix_slice =", pix_slice)
        Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
        #print("Lpix\n", Lpix)
        #print("Mpix\n", Mpix)
        Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
        #print("Jpix =", Jpix)
        lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
        print(lmn_grid.shape, lmn_grid)
        #sys.exit(0)
        pix_xyz = np.tensordot(uvw_frame.transpose(), lmn_grid, axes=1)  #EO: check this one!
        print(pix_xyz.shape, pix_xyz)
        #pix_xyz = np.tensordot(new_frame1.transpose(), lmn_grid, axes=1)  #EO: check this one!
        #pix_xyz = lmn_grid  #EO: check this one!

        

        ###############################################################
        XYZ = ms.instrument(t)

        t0 = atime.Time(int(t.mjd), format='mjd', scale='utc')
        XYZ0 = ms.instrument(t0)
        t1 = atime.Time(int(t.mjd) + 3/24, format='mjd', scale='utc')
        XYZ1 = ms.instrument(t1)
        t2 = atime.Time(int(t.mjd) + 6/24, format='mjd', scale='utc')
        XYZ2 = ms.instrument(t2)
        t3 = atime.Time(int(t.mjd) + 9/24, format='mjd', scale='utc')
        XYZ3 = ms.instrument(t3)
        ###t3 = atime.Time(int(t.mjd) + 10.5/24, format='mjd', scale='utc')
        ###XYZ3 = ms.instrument(t3)
        t4 = atime.Time(int(t.mjd) + 12/24, format='mjd', scale='utc')
        XYZ4 = ms.instrument(t4)
        t5 = atime.Time(int(t.mjd) + 15/24, format='mjd', scale='utc')
        XYZ5 = ms.instrument(t5)
        t6 = atime.Time(int(t.mjd) + 18/24, format='mjd', scale='utc')
        XYZ6 = ms.instrument(t6)
        t7 = atime.Time(int(t.mjd) + 15/24, format='mjd', scale='utc')
        XYZ7 = ms.instrument(t7)

        """
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
        axes[0][0].plot(XYZ0.data[:,0], XYZ0.data[:,1], 'r+')
        axes[0][1].plot(XYZ1.data[:,0], XYZ1.data[:,1], 'b+')
        axes[0][2].plot(XYZ2.data[:,0], XYZ2.data[:,1], 'b+')
        axes[0][3].plot(XYZ3.data[:,0], XYZ3.data[:,1], 'r+')
        axes[1][0].plot(XYZ4.data[:,0], XYZ4.data[:,1], 'b+')
        axes[1][1].plot(XYZ5.data[:,0], XYZ5.data[:,1], 'b+')
        axes[1][2].plot(XYZ6.data[:,0], XYZ6.data[:,1], 'b+')
        axes[1][3].plot(XYZ7.data[:,0], XYZ7.data[:,1], 'b+')
        plt.show()

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
        axes[0][0].plot((XYZ0.data[:, None, :] - XYZ0.data[None, ...]).flatten(), 'g+')
        """

        #beta  = dec #  0 if N, pi/2 on eq, pi on S
        #gamma = np.pi/2 + ra # 
        beta  = np.pi/2 - dec #  0 if N, pi/2 on eq, pi on S
        gamma = np.pi - ra #np.pi/2 + ra/2 # 

        Ry = np.array([[ np.cos(beta), 0, np.sin(beta)],
                       [ 0,            1, 0          ],
                       [-np.sin(beta), 0, np.cos(beta)]])

        Rz = np.array([[ np.cos(gamma), -np.sin(gamma), 0],
                       [ np.sin(gamma),  np.cos(gamma), 0],
                       [ 0,              0,             1]])
        
        uvw_frame = Rz @ Ry
        uvw_frame = uvw_frame.transpose()
        print(f"new uvw_frame beta = {beta}, gamma = {gamma}\n", uvw_frame)

        pix_frame = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pix_frame = uvw_frame
        pix_xyz = np.tensordot(pix_frame.transpose(), lmn_grid, axes=1)  #EO: check this one!
       
        UVW_ = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
        bsl_ = UVW_[:, None, :] - UVW_[None, ...]

        bsl = XYZ.data[:, None, :] - XYZ.data[None, ...]

        bsl0 = XYZ0.data[:, None, :] - XYZ0.data[None, ...]
        bsl1 = XYZ1.data[:, None, :] - XYZ1.data[None, ...]
        bsl2 = XYZ2.data[:, None, :] - XYZ2.data[None, ...]
        bsl3 = XYZ3.data[:, None, :] - XYZ3.data[None, ...]
        
        print(bsl0.shape)
        #print(bsl0)
        
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
        axes[0][0].plot(bsl0[:,:,0], bsl0[:,:,1], 'g+')
        axes[0][1].plot(bsl1[:,:,0], bsl1[:,:,1], 'g+')
        axes[0][2].plot(bsl2[:,:,0], bsl2[:,:,1], 'g+')
        axes[0][3].plot(bsl3[:,:,0], bsl3[:,:,1], 'g+')
        axes[1][0].plot(bsl[:,:,0],  bsl[:,:,1],  'b+')
        axes[1][1].plot(uvw[:,:,0],  uvw[:,:,1],  'r+')
        axes[1][2].plot(bsl_[:,:,0], bsl_[:,:,1],  'm+')
        plt.savefig('abc.png')
        #plt.show()

        
        #sys.exit(0)

        """

        XYZ0 = ms._ITRF
        #print(XYZ0)
        #sys.exit(0)

        mean_X, mean_Y, mean_Z = np.mean(XYZ.data, axis=0)
        mean_X0, mean_Y0, mean_Z0 = np.mean(XYZ0.data, axis=0)
        print(f"mean pos  = {mean_X}, {mean_Y}, {mean_Z}")
        print(f"mean pos0 = {mean_X0}, {mean_Y0}, {mean_Z0}")

        itrs_position1 = coord.SkyCoord(x=1, y=1, z=0, obstime=t, frame="itrs")
        itrs1 = itrs_position1.cartesian.xyz
        r1 = np.linalg.norm(itrs1)
        itrs_position2 = coord.SkyCoord(x=2, y=2, z=0, obstime=t, frame="itrs")
        itrs2 = itrs_position2.cartesian.xyz
        r2 = np.linalg.norm(itrs2)
        print(r1, r2)
        print(itrs_position1)
        
        icrs_position1 = r1 * (itrs_position1.transform_to("icrs").cartesian.xyz)
        icrs_position2 = r2 * (itrs_position2.transform_to("icrs").cartesian.xyz)
        
        print(np.sqrt(icrs_position1[0]**2 + icrs_position1[1]**2 + icrs_position1[2]**2))
        print(np.sqrt(icrs_position2[0]**2 + icrs_position2[1]**2 + icrs_position2[2]**2))

        print(np.sqrt((icrs_position2[0] - icrs_position1[0])**2 +
                      (icrs_position2[1] - icrs_position1[1])**2 +
                      (icrs_position2[2] - icrs_position1[2])**2))
        print(np.sqrt((itrs2[0] - itrs1[0])**2 +
                      (itrs2[1] - itrs1[1])**2 +
                      (itrs2[2] - itrs1[2])**2))

        print(itrs_position2.cartesian.xyz - itrs_position1.cartesian.xyz)
        print(icrs_position2 - icrs_position1)

        #sys.exit(1)

        print(XYZ.data.shape)
        #print(f"XYZ at {t}\n", XYZ)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
        axes[0].plot(XYZ.data[:,0] - mean_X,   XYZ.data[:,1] - mean_Y,   'r+')
        axes[1].plot(XYZ0.data[:,0] - mean_X0, XYZ0.data[:,1] - mean_Y0, 'b+')
        plt.show()
        sys.exit(0)
        


        obs_pos = coord.EarthLocation.from_geocentric(x=mean_X0, y=mean_Y0, z=mean_Z0, unit=u.meter)
        print("obs_pos =", obs_pos)
        obs_lon, obs_lat, obs_height = obs_pos.to_geodetic()
        print(f"Array center lon, lat, height = {obs_lon}, {obs_lat}, {obs_height}")
        obs_lon = obs_lon.to(u.rad)
        obs_lat = obs_lat.to(u.rad)
        print(f"Array center lon, lat, height = {obs_lon}, {obs_lat}, {obs_height}")

        bsl_t = (XYZ.data[:, None, :] - XYZ.data[None, ...])
        print("bsl_t shape =", bsl_t.shape)
        bsl_t0 = (XYZ0.data[:, None, :] - XYZ0.data[None, ...])

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
        axes[0].plot(bsl_t[:,:,0],  bsl_t[:,:,1],  'r+')
        axes[1].plot(bsl_t0[:,:,0], bsl_t0[:,:,1], 'b+')
        plt.show()
        sys.exit(0)
        
        #plt.plot(bsl_t[:,:,0], bsl_t[:,:,1],   'r+')
        #plt.show()
        #sys.exit(0)
        
        #print("bsl_t =", bsl_t)
        #print("bsl_t lengths =", np.linalg.norm(bsl_t, axis=2),  np.min(np.linalg.norm(bsl_t, axis=2)), np.max(np.linalg.norm(bsl_t, axis=2)))
        #R1_a = np.pi / 2 - obs_lat
        #R3_a = np.pi / 2 + obs_lon
        #R = R1 * R3

        cos_lon, sin_lon = np.cos(obs_lon), np.sin(obs_lon)
        cos_lat, sin_lat = np.cos(obs_lat), np.sin(obs_lat)

        R = np.array([[-sin_lon,            cos_lon,           0      ],
                      [-cos_lon * sin_lat, -sin_lon * sin_lat, cos_lat],
                      [ cos_lon * cos_lat,  sin_lon * cos_lat, sin_lat]])
        print("R =\n", R)

        
        #XYZ = np.zeros(bsl_t.shape)

        for i in range(bsl_t.shape[0]):

            for j in range(bsl_t.shape[0]):

                enu = R @ bsl_t[i,j,:] # not neu!!!
                
                bsl_t[i,j,:] = enu

                b = np.linalg.norm(enu)
                if b == 0: continue
                e = np.arcsin(enu[2] / b)
                a = np.arctan(enu[0] / enu[1])
                #print(f"b = {b} meters, u = {bsl_t[i,j,2]} e = {e} rad, a = {a} rad")
                sin_e, cos_e = np.sin(e), np.cos(e)
                sin_a, cos_a = np.sin(a), np.cos(a)
                
                #X = b / wl * (cos_lat * sin_e - sin_lat * cos_e * cos_a)
                #Y = b / wl * (cos_e * sin_a)
                #Z = b / wl * (sin_lat * sin_e + cos_lat * cos_e * cos_a)

                XYZ_ = b * np.array([cos_lat * sin_e - sin_lat * cos_e * cos_a,
                                     cos_e * sin_a,
                                     sin_lat * sin_e + cos_lat * cos_e * cos_a])
                
                #bsl_t[i,j,:] = new_frame1 @ XYZ_
                #bsl_t[i,j,:] = XYZ_
                
                
                #print(f"baseline {i},{j} = {np.linalg.norm(bsl_t[i,j,:])}")
                #if i == 1:
                #    sys.exit(1)
                                

        print("bsl_t is now in local u,v,w")
        #print("bsl_t", bsl_t)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 13))
        axes[0].plot(uvw[:,:,0],    uvw[:,:,1],   'r+')
        axes[1].plot(bsl_t0[:,:,0], bsl_t0[:,:,1],   'b+')
        axes[2].plot(bsl_t[:,:,0],  bsl_t[:,:,1], 'g+')
        fig.tight_layout()
        plt.show()
        
        sys.exit(1)
        """

        if uvw_from_ms:
            UVW_baselines_t = uvw
        else:
            #print(f"ref uvw {uvw.shape}\n", uvw)
            #UVW = (uvw_frame.transpose() @ XYZ.data.transpose()).transpose()
            
            #UVW = (XYZ.data @ uvw_frame.transpose())
            #UVW = (XYZ.data @ uvw_frame.transpose())
            #print(f"UVW t {UVW.shape}\n", UVW)
            #print(f"UVW t UVW[:, None, :]\n", UVW[:, None, :])
            #print(f"UVW[None, ...] =\n", UVW[None, ...])
            #UVW_baselines_t = (UVW[:, None, :] - UVW[None, ...])
            #UVW_baselines_t = (UVW[None, ...] - UVW[:, None, :])
            UVW_baselines_t = bsl_
            #UVW_baselines_t = uvw

        #print("UVW_baselines_t", UVW_baselines_t)

        #UVW_baselines_t = uvw
        #continue

        ###ICRS_baselines_t = (XYZ.data[:, None, :] - XYZ.data[None, ...])
        
        ##UVW_baselines_t = uvw

        UVW_baselines.append(baseline_rescaling * UVW_baselines_t)
        #UVW_baselines.append(UVW_baselines_t) #EO watch out rescaling 
        
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
    #print(UVW_baselines.shape)
    #print(UVW_baselines[0,:,:,1] - UVW_baselines[0,:,:,2])

    #print("\n\n")
    #continue


    ##ICRS_baselines = np.stack(ICRS_baselines, axis=0).reshape(-1, 3)
    gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=0).reshape(-1)
    
    UVW_baselines=UVW_baselines.reshape(-1,3)
    w_correction = np.exp(1j * UVW_baselines[:, -1])

    if not use_raw_vis:
        gram_corrected_visibilities *= w_correction

    print("lmn_grid shape =", lmn_grid.shape)
    lmn_grid = lmn_grid.reshape(3, -1)
    grid_center = lmn_grid.mean(axis=-1)
    lmn_grid -= grid_center[:, None]
    lmn_grid = lmn_grid.reshape(3, -1)
    print("lmn_grid shape =", lmn_grid.shape)

    #UVW_baselines = 2 * np.pi * UVW_baselines.T.reshape(3, -1) / wl
    UVW_baselines =  UVW_baselines.T.reshape(3, -1) 

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
    
