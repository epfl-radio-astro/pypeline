import pypeline.phased_array.measurement_set as measurement_set
import pypeline.phased_array.mscudf as mscudf
import pypeline.phased_array.daskcudf as dskcudf
import pypeline.phased_array.bluebild.gram as bb_gr
import pathlib
import cudf
import astropy.units as u
import casacore.tables as ct
import astropy.coordinates as coord
import astropy.table as tb
import astropy.time as time
import astropy.units as u
import cupy as cp
import numpy as np
from time import perf_counter
import scipy.constants as constants
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe

# import rmm
# pool = rmm.mr.PoolMemoryResource(
#     rmm.mr.CudaMemoryResource(),
#     # initial_pool_size=2**34,
#     # maximum_pool_size=2**35
# )
# rmm.mr.set_current_device_resource(pool)
# import rmm
# cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

N_station = 37
N_level = 4

ms_file = "/work/backup/ska/gauss4/gauss4_t201806301100_SBL180.MS"
#ms_file = "/work/ska/CSCS/c31/lofar30MHz1/LB_8hr/lofar30MHz1_t201806301100_SBL153.MS"
ms = measurement_set.LofarMeasurementSet(ms_file, N_station)

cudf_file = "/work/ska/cudfoutput/gauss4_t201806301100_SBL180.parquet"
#cudf_file = "/work/ska/cudfoutput/lofar30MHz1_t201806301100_SBL153.parquet"
cudfms = mscudf.LofarMeasurementSet(cudf_file, N_station) 

channel_id = 0
data_col = 'DATA'

casatime = []
cudftime = []

# for endtime in [1,10,100,1000]:
#     print(endtime)
#     time_start = perf_counter()
#     # for t,f,S in ms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA"):
#     #     vis = cp.array(S.data)
#     ms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA")
#     time_end = perf_counter()

#     casatime.append(time_end-time_start)

#     time_start = perf_counter()
#     # for t, f, S in cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA"):
#     #     vis = S
#     cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA")
#     time_end = perf_counter()

#     cudftime.append(time_end-time_start)
    
# print(casatime)
# print(cudftime)

endtime = [1,10,100,1000]

for item in endtime:
    for t,f,S in ms.visibilities(channel_id=[channel_id], time_id=slice(0, item, 1), column="DATA"):
        vis = cp.array(S.data)