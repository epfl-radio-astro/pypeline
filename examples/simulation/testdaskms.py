import dask
import dask.array as da
import numpy as np
import os
from casacore import tables
from itertools import cycle
import astropy
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as time
from numba import cuda
import casacore.tables as ct
import astropy.time as astime
import astropy.table as tb
from time import perf_counter
import time
from daskms import xds_from_ms
from dask import delayed

ms_file = "/work/backup/ska/gauss4/gauss4_t201806301100_SBL180.MS"

tic = perf_counter()
#read dataset
datasets = xds_from_ms(ms_file, columns=["TIME", "ANTENNA1", "ANTENNA2", "FLAG", "DATA"])
ds = datasets[0]

#extract the time, antenna1, antenna2, flag, data from the datasets
timearray, antenna1, antenna2, flag, data = dask.compute(ds.TIME.data, ds.ANTENNA1.data, ds.ANTENNA2.data, ds.FLAG.data, ds.DATA.data, scheduler='single-threaded')

#do the processing
utime, idx, cnt = np.unique(timearray, return_index=True, return_counts=True)
time_id=slice(0, 1000, 1)
N_time = len(timearray)
time_id.indices(N_time)
time_start, time_stop, time_step = time_id.indices(N_time)
utime = utime[time_start:time_stop:time_step]
idx = idx[time_start:time_stop:time_step]
cnt = cnt[time_start:time_stop:time_step]

for t in range(len(utime)):
    start=idx[t]
    end=start+cnt[t]-1
    ut_a1 = antenna1[start:end]
    ut_a2 = antenna2[start:end]
    ut_flag = flag[start:end]
    ut_data = data[start:end]

toc = perf_counter()
print(toc-tic)
    