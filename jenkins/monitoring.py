"""

This module contains all the common information for monitoring and plotting.

"""
import os
import re
import collections
import argparse
import pathlib


def check_cl_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory',     help="Path to input directory",                type=pathlib.Path, required=True)
    parser.add_argument('--output_directory',    help="Path to output directory",               type=pathlib.Path, required=True)
    parser.add_argument('--reference_directory', help="Path to output directory",               type=pathlib.Path) # only used by imap.py
    parser.add_argument('--stat_file',           help="Path to output statistics file",         type=pathlib.Path)
    parser.add_argument('--last_build',          help="Last Jenkins build ID",                  type=int, default=-1)
    parser.add_argument('--ignore_up_to',        help="Ignore Jenkins build up to that number", type=int, default=0)
    args = parser.parse_args()
    print(args)

    return args

"""
Function to scan the indicated directory for time-labelled Jenkins solutions

"""
def scan(dir, ignore_upto):
    builds = {}
    with os.scandir(dir) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_\d+", entry.name):
                #print(entry.name)
                info = re.split('T|Z_', entry.name)
                build = int(info[2])
                #print(f"found build {build}")
                if build > ignore_upto:
                    builds[build] = [info[0], info[1], entry.name, {}]
    return builds


def define_solutions():

    # Define an entry for each labelled timing to be monitored
    # Important thing is to set the pattern as defined in the python script
    """
    Solution = collections.namedtuple('Solution', ['directory', 'label', 'marker', 'color', 'pattern'])
    
    SC    = Solution(directory='test_standard_cpu',            label='Std CPU', marker='o', color='blue', pattern='Serial')
    SG    = Solution(directory='test_standard_gpu',            label='Std GPU', marker='o', color='red', pattern='Serial')
    LBSSi = Solution(directory='lofar_bootes_ss',              label='Lofar Bootes SS - intensity field imaging', marker='o', color='gray', pattern='#@#IFIM')
    LBSSt = Solution(directory='lofar_bootes_ss',              label='Lofar Bootes SS - total', marker='o', color='black', pattern='#@#TOT')
    LBNi  = Solution(directory='lofar_bootes_nufft_small_fov', label='Lofar Bootes nufft - intensity field imaging', marker='o', color='lightgreen', pattern='#@#IFIM')
    LBNt  = Solution(directory='lofar_bootes_nufft_small_fov', label='Lofar Bootes nufft - total', marker='o', color='green', pattern='#@#TOT')
    LBN3i = Solution(directory='lofar_bootes_nufft3',          label='Lofar Bootes nufft3 - intensity field imaging', marker='o', color='violet', pattern='#@#IFIM')
    LBN3t = Solution(directory='lofar_bootes_nufft3',          label='Lofar Bootes nufft3 - total', marker='o', color='blueviolet', pattern='#@#TOT')
    """
    
    Solution = collections.namedtuple('Solution', ['directory', 'label',
                                                   'filename', 'refname', 'gridname', 'refgrid',
                                                   'marker', 'color', 'pattern',
                                                   'show_gridlines'])

    SC   = Solution(directory='test_standard_cpu', label='Standard Synthesizer - CPU',
                    filename='stats_combined.npy', gridname='grid.npy', refname='stats_combined.npy', refgrid='grid.npy',
                    marker='o', color='blue', pattern='Serial',
                    show_gridlines=1)
    SG   = Solution(directory='test_standard_gpu', label='Standard Synthesizer - GPU', 
                    filename='stats_combined.npy', gridname='grid.npy', refname='stats_combined.npy', refgrid='grid.npy',
                    marker='o', color='red', pattern='Serial',
                    show_gridlines=1)

    LBSSt = Solution(directory='lofar_bootes_ss', label='Lofar Bootes SS - total',
                     filename='I_lsq_eq_data.npy', gridname='I_lsq_eq_grid.npy', refname='I_lsq_eq_data.npy', refgrid='I_lsq_eq_grid.npy',
                     marker='^', color='black', pattern='#@#TOT',
                     show_gridlines=0)
    LBSSi = Solution(directory='lofar_bootes_ss', label='Lofar Bootes SS - intensity field imaging',
                     filename='', gridname='', refname='', refgrid='',
                     marker='^', color='gray', pattern='#@#IFIM',
                     show_gridlines=0)

    LBNt  = Solution(directory='lofar_bootes_nufft_small_fov', label='Lofar Bootes nufft - total',
                     filename='I_lsq_eq_data.npy', gridname='I_lsq_eq_grid.npy', refname='I_lsq_eq_data.npy', refgrid='I_lsq_eq_grid.npy',
                     marker='P', color='green', pattern='#@#TOT',
                     show_gridlines=0)
    LBNi  = Solution(directory='lofar_bootes_nufft_small_fov', label='Lofar Bootes nufft - intensity field imaging',
                     filename='', gridname='', refname='', refgrid='',
                     marker='P', color='lightgreen', pattern='#@#IFIM',
                     show_gridlines=0)

    LBN3t = Solution(directory='lofar_bootes_nufft3', label='Lofar Bootes nufft3 - total',
                     filename='I_lsq_eq_data.npy', gridname='I_lsq_eq_grid.npy', refname='I_lsq_eq_data.npy', refgrid='I_lsq_eq_grid.npy',
                     marker='v', color='blueviolet', pattern='#@#TOT',
                     show_gridlines=0)
    LBN3i = Solution(directory='lofar_bootes_nufft3', label='Lofar Bootes nufft3 - intensity field imaging',
                     filename='', gridname='', refname='', refgrid='',
                     marker='v', color='violet', pattern='#@#IFIM',
                     show_gridlines=0)

    LBN3cct = Solution(directory='lofar_bootes_nufft3_cpp_cpu', label='Lofar Bootes nufft3 C++ CPU - total',
                       filename='I_lsq_eq_data.npy', gridname='I_lsq_eq_grid.npy', refname='I_lsq_eq_data.npy', refgrid='I_lsq_eq_grid.npy',
                       marker='x', color='hotpink', pattern='#@#TOT',
                       show_gridlines=0)
    LBN3cci = Solution(directory='lofar_bootes_nufft3_cpp_cpu', label='Lofar Bootes nufft3 C++ CPU - intensity field imaging',
                       filename='', gridname='', refname='', refgrid='',
                       marker='x', color='pink', pattern='#@#IFIM',
                       show_gridlines=0)
    
    LBN3cgt = Solution(directory='lofar_bootes_nufft3_cpp_gpu', label='Lofar Bootes nufft3 C++ GPU - total',
                       filename='I_lsq_eq_data.npy', gridname='I_lsq_eq_grid.npy', refname='I_lsq_eq_data.npy', refgrid='I_lsq_eq_grid.npy',
                       marker='D', color='darkgrey', pattern='#@#TOT',
                       show_gridlines=0)
    LBN3cgi = Solution(directory='lofar_bootes_nufft3_cpp_gpu', label='Lofar Bootes nufft3 C++ GPU - intensity field imaging',
                       filename='', gridname='', refname='', refgrid='',
                       marker='D', color='lightgrey', pattern='#@#IFIM',
                       show_gridlines=0)

    Solutions = {
        'SC'     : SC,
        'SG'     : SG,
        'LBNi'   : LBNi,
        'LBNt'   : LBNt,
        'LBSSi'  : LBSSi,
        'LBSSt'  : LBSSt,
        'LBN3i'  : LBN3i,
        'LBN3t'  : LBN3t,
        'LBN3cci': LBN3cci,
        'LBN3cct': LBN3cct,
        'LBN3cgi': LBN3cgi,
        'LBN3cgt': LBN3cgt
    }

    return Solutions
