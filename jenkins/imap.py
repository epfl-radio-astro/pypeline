import sys
import os
import re
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import getopt
import math
import imot_tools.io.s2image as image


def scan(dir, ignore_upto):
    builds = {}
    with os.scandir(dir) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_\d+", entry.name):
                #print(entry.name)
                info = re.split('T|Z_', entry.name)
                build = int(info[2])
                if build > ignore_upto:
                    builds[build] = [info[0], info[1], entry.name, tts.copy()]
    return builds


def main(argv):

    indir  = ''
    outdir = ''
    refdir = ''
    lastb  = -1
    fstat  = ''
    fromb  = 0

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:b:f:r:s:")
    except getopt.GetoptError as e:
        print('Error:', e)
        print(f'{argv[0]} -i </path/to/input/directory> -r </path/to/reference/directory> -o </path/to/output/directory> [-b <last build id>]')
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print(f'{argv[0]} -i </path/to/input/directory> -r </path/to/reference/directory> -o </path/to/output/directory> -b <last build id> -f </path/to/filestat')
            sys.exit(1)
        elif opt == '-i':
            indir = arg
        elif opt in '-o':
            outdir = arg
        elif opt in '-b':
            lastb = int(arg)
        elif opt in '-f':
            fstat = arg
        elif opt in '-r':
            refdir = arg
        elif opt in '-s':
            fromb = int(arg)

    if indir == '':
        print(f'Fatal: argument -i </path/to/input/directory> not found.')
        sys.exit(1)
    if outdir == '':
        print(f'Fatal: argument -o </path/to/output/directory> not found.')
        sys.exit(1)
    if refdir == '':
        print(f'Fatal: argument -r </path/to/reference/directory> not found.')
        sys.exit(1)
    if fstat == '':
        print(f'Fatal: argument -f </path/to/filestat> not found.')
        sys.exit(1)

    print(f"indir  is {indir}")
    print(f"outdir is {outdir}")
    print(f"refdir is {refdir}")
    print(f"fstat  is {fstat}")
    print(f"fromb  is {fromb}")


    fstats = open(fstat, 'w')
    print(f"Writing statistics to file {fstat}")


    builds = scan(indir, fromb)

    # Issue warning if expected lastb solution is missing or if stats exceed threshold

    for sol in sorted(sols.keys()):

        print(f"\n##### {sol}")

        # Reference image and grid (if any)
        file_iref = os.path.join(refdir, sols.get(sol).directory, sols.get(sol).refname)
        print(f"image reference: {file_iref}")
        if not os.path.isfile(file_iref):
            print(f"Fatal. Reference solution {file_iref} not found!")
            sys.exit(1)
        with open(file_iref, 'rb') as f:
            iref = np.load(f, allow_pickle=True)
                    
        file_gref = os.path.join(refdir, sols.get(sol).directory, sols.get(sol).refgrid)
        if not os.path.isfile(file_gref):
            print(f"Fatal. Reference grid {file_gref} not found!")
            sys.exit(1)
        with open(file_gref, 'rb') as f:
            gref = np.load(f, allow_pickle=True)


        # Solution image and grid
        file_isol = os.path.join(indir, builds.get(lastb)[2], sols.get(sol).directory, sols.get(sol).filename)
        print(f"image solution: {file_isol}")
        if not os.path.isfile(file_isol):
            msg = f"{sol} solution map {file_isol} not found. _WARNING_"
            fstats.write(msg + "\n")
            print(msg)
            continue
        with open(file_isol, 'rb') as f:
            isol = np.load(f, allow_pickle=True)

        file_gsol = os.path.join(indir, builds.get(lastb)[2], sols.get(sol).directory, sols.get(sol).gridname)
        if not os.path.isfile(file_gsol):
            msg = f"{sol} solution grid {file_gsol} not found. _WARNING_"
            fstats.write(msg + "\n")
            print(msg)
            continue
        with open(file_gsol, 'rb') as f:
            gsol = np.load(f, allow_pickle=True)
        
        print("Shape of iref = ", iref.shape)
        print("Shape of isol = ", isol.shape)
        print("Shape of gref = ", gref.shape)
        print("Shape of gsol = ", gsol.shape)

        if np.shape(iref) != np.shape(isol):
            print(f"Fatal. Similar i shapes expected iref is {np.shape(iref)} whereas isol is {np.shape(isol)}.")
            sys.exit(1)

        # Ref and img grid must be the same
        if not np.array_equal(gref, gsol):
            print(f"Fatal. Grids for iref and isol must be the same.")
            sys.exit(1)

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
        ax[0].set_title("build " + str(lastb))
        img_ref.draw(ax=ax[1], data_kwargs = {"cmap": "GnBu_r"}, show_gridlines=show_gridlines, grid_kwargs = grid_kwargs)
        ax[1].set_title("Reference")
        img_diff.draw(ax=ax[2], data_kwargs = {"cmap": color_diff}, show_gridlines=show_gridlines, grid_kwargs = grid_kwargs)
        ax[2].set_title("Difference")
        fig.suptitle(sols.get(sol).label + "\n\n" + sols.get(sol).directory, fontsize=20, y=0.9)
        plt.show()
        fig.savefig(os.path.join(outdir, sol + ".png"))

    fstats.close()


if __name__ == "__main__":

    tts = {}

    # Define matrix of solutions to monitor for image differences
    #TODO@EO: should be defined separatly to be shared with tts.py
    Solution = collections.namedtuple('Solution', ['directory', 'label', 'filename', 'refname',
                                                   'gridname', 'refgrid', 'show_gridlines'])

    SC   = Solution(directory='test_standard_cpu', label='Standard Synthesizer - CPU',
                    filename='stats_combined.npy', gridname='grid.npy', refname='stats_combined.npy', refgrid='grid.npy',
                    show_gridlines=1)
    
    SG   = Solution(directory='test_standard_gpu', label='Standard Synthesizer - GPU', 
                    filename='stats_combined.npy', gridname='grid.npy', refname='stats_combined.npy', refgrid='grid.npy',
                    show_gridlines=1)
    
    LBSS = Solution(directory='lofar_bootes_ss', label='Lofar Bootes SS - intensity field imaging',
                    filename='I_lsq_eq_data.npy', gridname='I_lsq_eq_grid.npy', refname='I_lsq_eq_data.npy', refgrid='I_lsq_eq_grid.npy',
                    show_gridlines=0)

    LBN  = Solution(directory='lofar_bootes_nufft_small_fov', label='Bluebild least-squares, sensitivity-corrected image (NUFFT)',
                    filename='I_lsq_eq_data.npy', gridname='I_lsq_eq_grid.npy', refname='I_lsq_eq_data.npy', refgrid='I_lsq_eq_grid.npy',
                    show_gridlines=0)

    LBN3 = Solution(directory='lofar_bootes_nufft3', label='Bluebild least-squares, sensitivity-corrected image (NUFFT3)',
                   filename='I_lsq_eq_data.npy', gridname='I_lsq_eq_grid.npy', refname='I_lsq_eq_data.npy', refgrid='I_lsq_eq_grid.npy',
                   show_gridlines=0)

    # Solutions to plot
    sols = {
        'SC': SC,
        'SG': SG,
        'LBSS': LBSS,
        'LBN': LBN,
        'LBN3': LBN3
    }
    for sol in sorted(sols.keys()):
        print(sols.get(sol))


    main(sys.argv)
