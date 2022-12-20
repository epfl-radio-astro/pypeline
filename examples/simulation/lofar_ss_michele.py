import sys
import os
import numpy as np
import scipy.constants as constants
from time import perf_counter
import astropy.units as u
from imot_tools.io import fits as ifits, s2image
from pypeline.phased_array import measurement_set as m_s
from pypeline.phased_array.bluebild import gram as bb_gr
from pypeline.phased_array.bluebild import data_processor as bb_dp
from pypeline.phased_array.bluebild import parameter_estimator as bb_pe
from pypeline.phased_array.bluebild.imager import spatial_domain as bb_sd
import pypeline.phased_array.instrument as instrument
import bluebild
#import bb_tb

np.random.seed(0)

#args = bb_tb.check_args(sys.argv)

N_bits  = 64 
#if args.precision == 'single' else 64
# dtype_f = np.float32   if N_bits == 32 else np.float64
# dtype_c = np.complex64 if N_bits == 32 else np.complex128

#ctx = None if args.processing_unit == None else bluebild.Context(args.processing_unit)

print("-I- processing unit:", args.processing_unit)
print("-I- precision: ", args.precision, N_bits)


time_slice = None #
time_slice = 1
N_station  = 60
N_level    = 20

filename = "lofar30MHz1"
subdir   = "LB_8hr"
#subdir   = "LB_1hr"
path_in  = os.path.join("/work/ska/CSCS/c31", filename, subdir)
assert os.path.isdir(path_in) == True
ms_file  = os.path.join(path_in, ("%s_t201806301100_SBL153.MS/" % filename))
assert os.path.isdir(ms_file) == True
fits_file = os.path.join(path_in, (filename + "-image.fits"))
assert os.path.isfile(fits_file) == True

path_out = os.path.join('./out_michele', subdir, 'gpu')
if not os.path.exists(path_out):
    os.makedirs(path_out)

data_col = "MODEL_DATA"

print("path_in  :", path_in)
print("ms_file  :", ms_file)
print("fits_file:", fits_file)
print("path_out :", path_out)

dev = instrument.LofarBlock(N_station)

# Measurement Set
ms = m_s.LofarMeasurementSet(ms_file, N_station)
channel_id = 1
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)

# Observation
FoV = np.deg2rad((2000*2.*u.arcsec).to(u.deg).value)
field_center = ms.field_center
if time_slice is not None:
    times = ms.time['TIME'][:time_slice]
else:
    times = ms.time['TIME']
print(len(times))

# Instrument
gram = bb_gr.GramBlock()


### Imaging parameters ===========================================================
cl_WCS = ifits.wcs(fits_file)
cl_WCS = cl_WCS.sub(['celestial']) 
#cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
px_grid = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
N_cl_lon, N_cl_lat = px_grid.shape[-2:]
assert N_cl_lon == N_cl_lat
N_pix = N_cl_lon

### Bypassing parameter estimation !!!!
N_eig, c_centroid = N_level, list(range(N_level)) 


####################################################################
#### INTENSITY FIELD: Imaging
####################################################################
I_dp  = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits, ctx)

timing_ifim = -perf_counter()
timing_vis  = 0.0
timing_idp  = 0.0
timing_imfs = 0.0
for i_t, ti in enumerate(times):
    d2h  = True if i_t == times[-1] else False
    tic = perf_counter()
    tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column=data_col))
    tvis = perf_counter() - tic
    XYZ  = ms.instrument(tobs)
    W    = ms.beamformer(XYZ, wl)
    S, _ = m_s.filter_data(S, W)
    tic = perf_counter()
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    tidp = perf_counter() - tic
    c_idx = np.arange(0, N_level)        # bypass c_idx
    tic = perf_counter()
    I_mfs(D, V, XYZ.data, W.data, c_idx, d2h)
    timfs = perf_counter() - tic
    timing_vis  += tvis
    timing_idp  += tidp
    timing_imfs += timfs
    print(f"-R- Intensity at: {i_t}/{len(times)}, {ti.value:.7f}: tvis={tvis:.3f} timfs={timfs:.3f}")

timing_ifim += perf_counter()

sys.exit(0)

# ref_name = 'lbssmich'
# sol_name = 'lbssmich'
# bb_tb.dump_stats(I_mfs, f"{sol_name}_{args.processing_unit_name}_imfs", args.outdir)

# bb_tb.dump_json(v_shape=V.shape, w_shape=W.shape, grid_shape=px_grid.shape,
#                 t_ifim=timing_ifim, t_vis=timing_vis, t_idp=timing_idp, t_imfs=timing_imfs,
#                 filename=f"{sol_name}_{args.processing_unit_name}.json", outdir=args.outdir)

# if args.outdir:
#     ref_path = os.path.join(args.outdir, f"{ref_name}_none.json")
#     sol_path = os.path.join(args.outdir, f"{sol_name}_{args.processing_unit_name}.json")
#     bb_tb.compare_solutions(ref = ref_path, sol = sol_path)


# """
# sys.exit(0)

# I_std_ss, I_lsq_ss = I_mfs_ss.as_image()


# ####################################################################
# #### SENSITIVITY FIELD: Imaging
# ####################################################################
# S_dp     = bb_dp.SensitivityFieldDataProcessorBlock(N_eig, ctx=None)
# S_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits, ctx)

# for i_t, ti in enumerate(times):
#     print("Sensitivity at:", i_t, ti)
#     d2h = True if ti == times[-1] else False
#     tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column=data_col))
#     XYZ  = ms.instrument(tobs)
#     W    = ms.beamformer(XYZ, wl)
#     D, V = S_dp(XYZ, W, wl)
#     tic = perf_counter()
#     S_mfs_ss(D, V, XYZ.data, W.data, np.zeros(N_eig, dtype=int), d2h)
#     time_cpp += (perf_counter() - tic)

# # Save eigen-values
# np.save(path.join(path_out, ("D_" + filename)), D.reshape(-1, 1, 1))

# _, S_ss = S_mfs_ss.as_image()

# # Image gridding
# I_lsq_eq_ss = s2image.Image(I_lsq_ss.data / S_ss.data, I_lsq_ss.grid)

# # Save eigen-vectors for Standard Synthesis
# np.save(path.join(path_out, ("I_ss_" + filename)), I_lsq_eq_ss.data)

# # Interpolate image to MS grid-frame for Standard Synthesis
# f_interp = (I_lsq_eq_ss.data.reshape(N_level, N_cl_lon, N_cl_lat).transpose(0, 2, 1))
# I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
# I_lsq_eq_interp.to_fits(path.join(path_out, ("I_ss_" + filename + ".fits")))


# # EO: compare I .npy to reference python
# #if ctx is not None:
# #    i_ref  = np.load(path.join((path_out + '_python'), ("I_ss_" + filename + '.npy')))
# #    i_sol  = np.load(path.join(path_out, ("I_ss_" + filename + '.npy')))
# #    i_diff = i_sol - i_ref
# #    print("i_diff =", np.min(i_diff), np.max(i_diff))

# """
