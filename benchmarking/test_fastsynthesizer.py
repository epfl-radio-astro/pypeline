
import os
if os.getenv('OMP_NUM_THREADS') == None : os.environ['OMP_NUM_THREADS'] = "1"

import bluebild_tools.cupy_util as bbt_cupy
use_cupy = bbt_cupy.is_cupy_usable()

import sys,timing
import numpy as np
import scipy.sparse as sparse
from pathlib import Path
import imot_tools.io.s2image as image
import imot_tools.math.sphere.transform as transform
import astropy.time as atime
import matplotlib.pyplot as plt
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as synth_periodic
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth_standard
import utils.paths as pypaths
import dummy_synthesis 
from dummy_synthesis import synthesize, synthesize_stack
from data_gen_utils import RandomDataGen, SimulatedDataGen, RealDataGen


# For CuPy agnostic code
# ----------------------
xp = bbt_cupy.cupy if use_cupy else np


def draw_comparison(stats_standard, field_periodic, pix, icrs_grid):
    img_standard = image.Image(stats_standard, pix)
    img_periodic = image.Image(field_periodic, icrs_grid)
    img_periodic_rot = image.Image(np.rot90(field_periodic,2), pix)
    img_standard_rot = image.Image(np.rot90(stats_standard,2), icrs_grid)
    img_diff_pix = image.Image(stats_standard - np.rot90(field_periodic,2), pix)
    img_diff_icrs_grid = image.Image(np.rot90(stats_standard,2) - field_periodic, icrs_grid)

    print("Difference between pix grid and  & icrs_grid:",  np.average(pix - icrs_grid))

    fig, ax = plt.subplots(ncols=3, nrows = 2)
    fig.tight_layout(pad = 2.0)
    grid_kwargs = {"ticks": False}
    color_diff = "RdBu"
    img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": "GnBu_r"}, grid_kwargs = grid_kwargs)
    ax[0,0].set_title("Bluebild Standard Image\nSS Grid")
    img_periodic_rot.draw(ax=ax[0,1], data_kwargs = {"cmap": "GnBu_r"}, grid_kwargs = grid_kwargs)
    ax[0,1].set_title("Bluebild Periodic Image\nRotated 180$^\circ$\nSS Grid")
    img_diff_pix.draw(ax=ax[0,2], data_kwargs =  {"cmap": color_diff},grid_kwargs = grid_kwargs)
    ax[0,2].set_title("Difference\nSS Grid")

    img_standard_rot.draw(ax=ax[1,0], data_kwargs = {"cmap": "GnBu_r"}, grid_kwargs = grid_kwargs)
    ax[1,0].set_title("Bluebild Standard Image\nRotated 180$^\circ$\nPS Grid")
    img_periodic.draw(ax=ax[1,1], data_kwargs = {"cmap": "GnBu_r"}, grid_kwargs = grid_kwargs)
    ax[1,1].set_title("Bluebild Periodic Image\nPS Grid")
    img_diff_icrs_grid.draw(ax=ax[1,2], data_kwargs =  {"cmap": color_diff}, grid_kwargs = grid_kwargs)
    ax[1,2].set_title("Difference\nPS Grid")


    fig.savefig("test_compare.png")
    fig.show()
    plt.show()

def draw_levels(stats_standard, field_periodic, stats_standard_norm, field_periodic_norm, pix, icrs_grid):
    grid_kwargs = {"ticks": False}
    img_standard = image.Image(stats_standard, pix)
    img_periodic = image.Image(field_periodic, icrs_grid)
    img_standard_norm = image.Image(stats_standard_norm, pix)
    img_periodic_norm = image.Image(field_periodic_norm, icrs_grid)

    fig, ax = plt.subplots(ncols=data.N_level+2, nrows = 2, figsize=(7, 3))
    #fig.tight_layout(pad = 2.0)
    img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[0,0].set_title("Standard Image\nAll Levels")
    img_periodic.draw(ax=ax[1,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[1,0].set_title("Periodic Image\nAll Levels")
    img_standard_norm.draw(ax=ax[0,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[0,1].set_title("Standard Image\nAll Levels, Normalized")
    img_periodic_norm.draw(ax=ax[1,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[1,1].set_title("Periodic Image\nAll Levels, Normalized")
    for i in range(0,data.N_level):
        print(i)
        img_standard_norm.draw(ax=ax[0,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[0,i+2].set_title("Standard Image\nNormalized Level {0}".format(i))
        img_periodic_norm.draw(ax=ax[1,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[1,i+2].set_title("Periodic Image\nNormalized Level {0}".format(i))
    fig.savefig("test_compare_levels.png")
    fig.show()
    plt.show()

def draw_standard_levels(stats_standard, stats_standard_norm, pix):
    grid_kwargs = {"ticks": False}
    img_standard = image.Image(stats_standard, pix)
    img_standard_norm = image.Image(stats_standard_norm, pix)

    fig, ax = plt.subplots(ncols=data.N_level+2, nrows = 1, figsize=(7, 3))
    #fig.tight_layout(pad = 2.0)
    img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[0,0].set_title("Standard Image\nAll Levels")
    img_periodic.draw(ax=ax[1,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[1,0].set_title("Periodic Image\nAll Levels")
    img_standard_norm.draw(ax=ax[0,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[0,1].set_title("Standard Image\nAll Levels, Normalized")
    img_periodic_norm.draw(ax=ax[1,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[1,1].set_title("Periodic Image\nAll Levels, Normalized")
    for i in range(0,data.N_level):
        print(i)
        img_standard_norm.draw(ax=ax[0,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[0,i+2].set_title("Standard Image\nNormalized Level {0}".format(i))
        img_periodic_norm.draw(ax=ax[1,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[1,i+2].set_title("Periodic Image\nNormalized Level {0}".format(i))
    fig.show()
    plt.show()

if __name__ == "__main__":

    ###### make simulated dataset ###### 

    precision = 32 # 32 or 64
    datasets_dir = pypaths.get_datasets_path()
    #data = SimulatedDataGen(frequency = 145e6)
    data = RealDataGen(Path.joinpath(datasets_dir, "gauss4/gauss4_t201806301100_SBL180.MS").as_posix(), N_level = 4, N_station = 24 ) # n level = # eigenimages
    #data = dummy_synthesis.RandomDataGen()

    ################################### 

    # timer
    timer = timing.Timer()
    # function to turn timing on or off
    #timer.off() 

    pix = data.getPixGrid()

    # The Fast Periodic Synthesis Kernel
    synthesizer_periodic  = synth_periodic.FourierFieldSynthesizerBlock(data.wl, data.px_colat_periodic, data.px_lon_periodic, data.N_FS, data.T_kernel, data.R, precision)
    synthesizer_periodic.set_timer(timer, "Periodic ")

    synthesizer_standard = synth_standard.SpatialFieldSynthesizerBlock(data.wl, pix, precision)
    synthesizer_standard.set_timer(timer, "Standard ")

    # iterate though timesteps
    # increase the range to run through more calls
    stats_standard_combined = None
    stats_periodic_combined = None
    stats_standard_normcombined = None
    stats_periodic_normcombined = None
    icrs_grid = None

    for t in range(0,10):
        (V, XYZ, W, D) = data.getVXYZWD(t)
        print("t = {0}".format(t))
        
        if isinstance(W, sparse.csr.csr_matrix) or isinstance(W, sparse.csc.csc_matrix):
            W = W.toarray()

        print("use_cupy =", use_cupy)

        if use_cupy:
            XYZ = xp.asarray(XYZ)
            W   = xp.asarray(W)
            V   = xp.asarray(V)

        stats_standard = synthesizer_standard(V, XYZ, W)
        print("stats_standard = ", type(stats_standard))
        stats_periodic = synthesizer_periodic(V, XYZ, W)
        print("stats_periodic = ", type(stats_periodic))

        # call the Bluebild Synthesis Kernels
        #stats_standard = synthesizer_standard(V,XYZ,W)
        stats_standard_gpu = synthesizer_standard(V_gpu,XYZ_gpu,W_gpu)
        stats_standard = stats_standard_gpu#.get()

        D_r =  D.reshape(-1, 1, 1)

        stats_standard_norm = stats_standard * D_r
        stats_periodic_norm = stats_periodic * D_r

        # transform the periodic field statistics to periodic eigenimages
        field_periodic      = synthesizer_periodic.synthesize(stats_periodic)
        field_periodic_norm = synthesizer_periodic.synthesize(stats_periodic_norm)

        bfsf_grid = transform.pol2cart(1, data.px_colat_periodic, data.px_lon_periodic)
        icrs_grid = np.tensordot(synthesizer_periodic._R.T, bfsf_grid, axes=1)

        try:
            print("Difference in results between standard & periodic synthesizers:", np.average( stats_standard - np.rot90(field_periodic,2)))
        except:
            print("Shapes are different between standard & periodic synthesizers. Standard: {0}, periodic: {1}".format(stats_standard.shape, field_periodic.shape ))
            print("Trimming down PS grid")
            icrs_grid = icrs_grid[:,:,:-1]
            field_periodic = field_periodic[:,:,:-1]
            field_periodic_norm = field_periodic_norm[:,:,:-1]

        try:    stats_standard_combined += stats_standard
        except: stats_standard_combined = stats_standard
        try:    stats_periodic_combined += field_periodic
        except: stats_periodic_combined = field_periodic
        try:    stats_standard_normcombined += stats_standard_norm
        except: stats_standard_normcombined = stats_standard_norm
        try:    stats_periodic_normcombined += field_periodic_norm
        except: stats_periodic_normcombined = field_periodic_norm

    draw_comparison(stats_standard, field_periodic, pix, icrs_grid)
    draw_levels(stats_standard_combined, stats_periodic_combined, stats_standard_normcombined, stats_periodic_normcombined, pix, icrs_grid)
    #draw_standard_levels(stats_standard_combined, stats_standard_normcombined, pix)
   
    print(timer.summary())
