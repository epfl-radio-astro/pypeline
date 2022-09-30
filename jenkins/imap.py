import sys
import os
import re
import numpy as np
import collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import getopt
import math
import imot_tools.io.s2image as image
import monitoring


def np_load(filepath):
    if not os.path.isfile(filepath):
        print(f"Fatal  : file {filepath} not found!")
        sys.exit(1)
    with open(filepath, 'rb') as f:
        img = np.load(f, allow_pickle=True)
    return img


def check_images_shapes(img1, img2):
    if np.shape(img1) != np.shape(img2):
        print(f"Fatal  : Trying to compare images of different shapes ({np.shape(img1)} vs {np.shape(img2)}.")
        sys.exit(1)

def check_grids_definitions(grd1, grd2):
    #if not np.array_equal(grd1, grd2):
    if not np.isclose(grd1, grd2, atol=1e-6).any():
        print(f"Fatal  : Grids from both solutions must be the same.")
        sys.exit(1)

def plot(plot, args):

    fstats = open(args.stat_file, 'w')
    print(f"Writing statistics to file {args.stat_file}")

    builds = monitoring.scan(args.input_directory, args.ignore_up_to)

    sols = plot['sols']

    # Issue warning if expected lastb solution is missing or if stats exceed threshold

    for sol in sorted(sols.keys()):

        print(f"\n##### {sol}")

        # Reference image and grid (if any)
        file_iref = os.path.join(args.reference_directory, sols.get(sol).directory, sols.get(sol).refname)
        file_gref = os.path.join(args.reference_directory, sols.get(sol).directory, sols.get(sol).refgrid)
        iref = np_load(file_iref)
        gref = np_load(file_gref)

        # Solution image and grid (emit warning if missing)
        file_isol = os.path.join(args.input_directory, builds.get(args.last_build)[3], sols.get(sol).directory, sols.get(sol).filename)
        file_gsol = os.path.join(args.input_directory, builds.get(args.last_build)[3], sols.get(sol).directory, sols.get(sol).gridname)

        # Warnings to be recovered by Jenkins to notify Slack channel
        if not os.path.isfile(file_isol):
            msg = f"{sol} solution map {file_isol} not found. _WARNING_"
            fstats.write(msg + "\n")
            print(msg)
            continue

        if not os.path.isfile(file_gsol):
            msg = f"{sol} solution grid {file_gsol} not found. _WARNING_"
            fstats.write(msg + "\n")
            print(msg)
            continue

        # Load only if file paths do exist
        isol = np_load(file_isol)
        gsol = np_load(file_gsol)

        print(f"image reference: {file_iref}")
        print(f"image solution : {file_isol}")
        print("Shape of iref: ", iref.shape)
        print("Shape of isol: ", isol.shape)
        print("Shape of gref: ", gref.shape)
        print("Shape of gsol: ", gsol.shape)
        
        check_images_shapes(iref, isol)
        check_grids_definitions(gref, gsol)

        img_ref  = image.Image(iref, gref)
        img_sol  = image.Image(isol, gref)
        img_diff = image.Image(iref - isol, gref)
 
        rmse = np.sqrt(((img_diff.data) ** 2).mean())
        msg = (f"{sol} RMSE = {rmse:.6f} {file_isol} - {file_iref} on {file_gref}")
        threshold = 0.001
        if rmse > threshold:
            msg += f" _WARNING_ RMSE > {threshold}"
        print(msg)

        fstats.write(msg + "\n")

        # Produce a plot img ref diff
        fig, ax = plt.subplots(1, 3, figsize=(11.7, 8.3))
        fig.tight_layout(rect=(0,0,0.95,0.95))
        grid_kwargs = {"ticks": False}
        color_diff = "RdBu"
        show_gridlines = True
        if sols.get(sol).show_gridlines == 0:
            show_gridlines = False
        img_sol.draw(ax=ax[0], data_kwargs = {"cmap": "GnBu_r"}, show_gridlines=show_gridlines, grid_kwargs = grid_kwargs)
        ax[0].set_title("build " + str(args.last_build))
        img_ref.draw(ax=ax[1], data_kwargs = {"cmap": "GnBu_r"}, show_gridlines=show_gridlines, grid_kwargs = grid_kwargs)
        ax[1].set_title("Reference")
        img_diff.draw(ax=ax[2], data_kwargs = {"cmap": color_diff}, show_gridlines=show_gridlines, grid_kwargs = grid_kwargs)
        ax[2].set_title(f"Difference RMSE = {rmse:.3f}")
        fig.suptitle(sols.get(sol).label, fontsize=20, y=0.8)
        fig.savefig(os.path.join(args.output_directory, sol + ".png"))

    fstats.close()


def compare_one_solution_to_another(Solutions, sol1_name, build1, sol2_name, build2):

    sol1 = Solutions[sol1_name]
    sol2 = Solutions[sol2_name]

    print(f"\nComparing [{sol1.label} / {build1}] to [{sol2.label} / {build2}]")
    builds = monitoring.scan(args.input_directory, args.ignore_up_to)
    file_isol1 = os.path.join(args.input_directory, builds.get(build1)[3], sol1.directory, sol1.filename)
    file_gsol1 = os.path.join(args.input_directory, builds.get(build1)[3], sol1.directory, sol1.gridname)
    file_isol2 = os.path.join(args.input_directory, builds.get(build2)[3], sol2.directory, sol2.filename)
    file_gsol2 = os.path.join(args.input_directory, builds.get(build2)[3], sol2.directory, sol2.gridname)
    print(f"Info   : {file_isol1}")
    print(f"Info   : {file_isol2}")
    isol1 = np_load(file_isol1)
    gsol1 = np_load(file_gsol1)
    isol2 = np_load(file_isol2)
    gsol2 = np_load(file_gsol2)
    check_images_shapes(isol1, isol2)
    check_grids_definitions(gsol1, gsol2)
    img1 = image.Image(isol1, gsol1)
    img2 = image.Image(isol2, gsol2)
    img_diff = image.Image(isol2 - isol1, gsol1)
    rmse = np.sqrt(((img_diff.data) ** 2).mean())

    # Produce a plot img ref diff
    fig, ax = plt.subplots(1, 3, figsize=(11.7, 8.3))
    fig.tight_layout(rect=(0,0,0.95,0.95))
    grid_kwargs = {"ticks": False}
    color_diff = "RdBu"
    show_gridlines = True
    if sol1.show_gridlines == 0 : show_gridlines = False
    img1.draw(ax=ax[0], data_kwargs = {"cmap": "GnBu_r"}, show_gridlines=show_gridlines, grid_kwargs = grid_kwargs)
    ax[0].set_title(sol1.label + "\n\n" + "build " + str(build1))
    img2.draw(ax=ax[1], data_kwargs = {"cmap": "GnBu_r"}, show_gridlines=show_gridlines, grid_kwargs = grid_kwargs)
    ax[1].set_title(sol2.label + "\n\n" + "build " + str(build2))
    img_diff.draw(ax=ax[2], data_kwargs = {"cmap": color_diff}, show_gridlines=show_gridlines, grid_kwargs = grid_kwargs)
    ax[2].set_title(f"Difference\n\nRMSE = {rmse:.3f}")
    #plt.show()
    figname = 'img_' + sol1_name + "_" + str(build1) + "_vs_" + sol2_name + "_" + str(build2)
    fig.savefig(os.path.join(args.output_directory, figname + ".png"))


if __name__ == "__main__":

    args = monitoring.check_cl_arguments()

    if args.reference_directory == None:
        print("Error  : you must pass a reference directory when using ", sys.argv[0])

    Solutions = monitoring.define_solutions()

    """
    sols = {
        'img_SC'      : Solutions['SC'],
        'img_SG'      : Solutions['SG'],
        'img_LBSSt'   : Solutions['LBSSt'],
        'img_LBNt'    : Solutions['LBNt'],
        'img_LBN3t'   : Solutions['LBN3t'],
        'img_LBN3cct' : Solutions['LBN3cct'],
        'img_LBN3cgt' : Solutions['LBN3cgt']
    }
    """

    sols = {
        'img_lb_n3_c64'  : Solutions['lb_n3_c64_i'],
        'img_lb_n3_cc64' : Solutions['lb_n3_cc64_i'],
        'img_lb_n3_cg64' : Solutions['lb_n3_cg64_i'],
        'img_lb_ss_c64'  : Solutions['lb_ss_c64_i'],
        'img_lb_ss_cc64' : Solutions['lb_ss_cc64_i'],
        'img_lb_ss_cg64' : Solutions['lb_ss_cg64_i']
    }

    plots = (
        {'sols': sols},
    )

    # Generating maps and stats of last build vs reference
    for plot_ in plots:
        plot(plot_, args)

    # Comparing pairs of solutions
    #compare_one_solution_to_another(Solutions, 'LBN3cct', args.last_build, 'LBN3cgt', args.last_build)
    #compare_one_solution_to_another(Solutions, 'LBN3cct', args.last_build, 'LBN3t',   args.last_build)
    #compare_one_solution_to_another(Solutions, 'LBSSt', args.last_build,   'LBN3t',   args.last_build)
    #compare_one_solution_to_another(Solutions, 'LBSSt', args.last_build,   'LBN3cgt', args.last_build)
    #compare_one_solution_to_another('LBN3cct', args.last_build, 'LBSSt',  args.last_build)

    compare_one_solution_to_another(Solutions, 'lb_n3_c64_i',  args.last_build, 'lb_n3_cc64_i', args.last_build)
    compare_one_solution_to_another(Solutions, 'lb_n3_c64_i',  args.last_build, 'lb_n3_cg64_i', args.last_build)

    compare_one_solution_to_another(Solutions, 'lb_ss_c64_i',  args.last_build, 'lb_ss_cc64_i', args.last_build)
    compare_one_solution_to_another(Solutions, 'lb_ss_c64_i',  args.last_build, 'lb_ss_cg64_i', args.last_build)

    compare_one_solution_to_another(Solutions, 'lb_n3_c64_i',  args.last_build, 'lb_ss_c64_i',  args.last_build)
    compare_one_solution_to_another(Solutions, 'lb_n3_cc64_i', args.last_build, 'lb_ss_cc64_i', args.last_build)
    compare_one_solution_to_another(Solutions, 'lb_n3_cg64_i', args.last_build, 'lb_ss_cg64_i', args.last_build)
