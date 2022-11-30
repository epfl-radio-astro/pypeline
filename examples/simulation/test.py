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
import sys
import nvtx

import rmm
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),
    # initial_pool_size=2**34,
    # maximum_pool_size=2**35
)
rmm.mr.set_current_device_resource(pool)
import rmm
cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

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
N=1
endtime = 5
for i in range(N):
    tic = perf_counter()
    with nvtx.annotate("msviscpu", color="green"):
        for t,f,S in ms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA"):
            vis = cp.asarray(S.data)
            print(vis)
    toc = perf_counter()
    print(i, toc-tic)
    
print('--------------')

for i in range(N):
    tic = perf_counter()
    with nvtx.annotate("cudfgpu", color="blue"):
        for t,f,S in cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA"):
            vis = S.data
    toc = perf_counter()
    print(i, toc-tic)
    
sys.exit(0)
    
        
        
# for endtime in [100]:
#     print(endtime)
#     tic = perf_counter()
#     obj = ms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA")
#     for t,f,S in obj:
#         continue
#     toc = perf_counter()
#     print(toc-tic)
    
#     time_start = perf_counter()
#     for t,f,S in ms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA"):
#         #vis = cp.array(S.data)
#         continue
#     time_end = perf_counter()
#     print(time_end - time_start)
    
#     time_start = perf_counter()
#     for t,f,S in ms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA"):
#         #vis = cp.array(S.data)
#         continue
#     time_end = perf_counter()
#     print(time_end - time_start)
    
#     tic = perf_counter()
#     obj = ms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA")
#     print(obj)
#     for t,f,S in obj:
#         continue
#     toc = perf_counter()
#     print(toc-tic)
    
#     #time_end = perf_counter()


#     time_start = perf_counter()
#     for t, f, S in cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA"):
#         continue
#     #cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA")
#     time_end = perf_counter()

#     print(time_end-time_start)
    
#     time_start = perf_counter()
#     obj = cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA")
#     print(obj)
#     time_end = perf_counter()

#     print(time_end-time_start)
    
#     time_start = perf_counter()
#     obj = cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, endtime, 1), column="DATA")
#     for t, f, S in obj:
#         continue
#     time_end = perf_counter()

#     print(time_end-time_start)
    
    
    
#print(casatime)
#print(cudftime)


# ############ For profiling

# endtime = [1000]

# # for item in endtime:
# #     for t,f,S in ms.visibilities(channel_id=[channel_id], time_id=slice(0, item, 1), column="DATA"):
# #         vis = cp.array(S.data)

# for item in endtime:
#     for t, f, S in cudfms.visibilities(channel_id=[channel_id], time_id=slice(0, item, 1), column="DATA"):
#         vis = S