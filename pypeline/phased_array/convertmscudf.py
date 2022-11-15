# #############################################################################
# convertmscudf.py
# ==================
# Author : Arpan Das [arpan.das@epfl.ch]
# #############################################################################

"""
Cudf file readers and tools.
"""

import dask
import dask_cudf
import cudf
import numpy as np
import os
from casacore import tables
import astropy
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as time

def convert_ms_cudf(infile = '', subtable = '', timecols = None, ignore = None, savefile = False, outpath = '', filename = ''):
    """
        Convert a CASA MS file to a cudf parquet file
        Parameters
        ----------
        infile : string
            path to the MS file needs to be converted.
        subtable : string
            if a subtable needs to be converted then name of the subtable
        ignore : list
            columns to be ignored 
        timecols : list
            list of strings specifying column names to convert to datetime format from casacore time.  Default is None
        savefile : {False, True}, optional 
            If True it will save an output file 
            If False it will only return the data format 
        outpath : string 
            If mentioned it will save the file in that path 
            If not mentioned then it will save in the current directory
        filename : string     
            Name of the output file 
        Returns
        -------
        df : parquet format
            The cudf data format
        """
    if timecols is None: timecols = []
    if ignore is None: ignore = []

    t = tables.table(os.path.join(infile, subtable), readonly=True)
    if t.nrows() == 0:
        t.close()
        return cudf.DataFrame()

    cols = t.colnames()
    df = cudf.DataFrame()

    for col in cols:
        tc = None
        if col in ignore:
            continue
        if col == 'TIME':
            tc = np.array(t.calc("MJD(TIME)"))
        elif col == 'TIME_CENTROID':
            tc = np.array(t.calc("MJD(TIME_CENTROID)"))
        else:
            tc = np.array(t.getcol(col))
        if tc.dtype != 'complex32' and tc.dtype != 'complex64' and tc.dtype!= 'complex128':
            df[col] = tc.tolist()
        else:
            casted = tc.view(np.float32)
            #reshaped = casted.reshape(tc.shape + (2,))
            df[col] = casted.tolist()
            
    if savefile == True:
        if outpath == '':
            outpath = os. getcwd()
        if filename == '':
            path, file = os.path.split(infile)
            filename = os.path.splitext(file)[0]
            
        df.to_parquet(outpath+filename+'.parquet') 
            
    return df

