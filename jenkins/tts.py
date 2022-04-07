import sys
import os
import re
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import getopt
import math
from pathlib import Path

import monitoring


def collect_runtimes(sols, dir, ignore_up_to_build):
    
    builds = monitoring.scan(dir, ignore_up_to_build)

    for build in sorted(builds.keys()):
        for sol in sorted(sols.keys()):
            soldir = os.path.join(dir, builds.get(build)[2], sols.get(sol).directory)
            pattern = sols.get(sol).pattern
            #print(f">Scanning solution {sol} in build {build:4d}: {soldir}")
            if os.path.isdir(soldir):
                with os.scandir(soldir) as it:
                    for entry in it:
                        if re.match(r"^slurm-\d+.out", entry.name):
                            slurm = os.path.join(soldir, entry.name)
                            print(f"slurm {slurm}")
                            with open(slurm, "r") as file:
                                for line in file:
                                    if re.search(pattern, line):
                                        info = re.split('\s+', line)
                                        builds.get(build)[3][sol] = info[1]
                                        break
                            break # there should only be a single slurm out
    return builds


def check_presence_lastb(builds, lastb):
    if sorted(builds.keys())[-1] == lastb:
        return True
    else:
        return False


def stats_n_plots(plot, dir, builds, lastb, fstat):

    sols = plot['sols']
    filename = plot['filename']

    isin = check_presence_lastb(builds, lastb)
    print(f"lastb is in builds?", isin)

    #flist = font_manager.get_fontconfig_fonts()
    #names = [font_manager.FontProperties(fname=fname).get_name() for fname in flist]
    #print(names)

    if plot['w']:
        fstats = open(fstat, 'w')
        print(f"Writing statistics to file {fstat}")

    plt.subplots(figsize=(11.7, 8.3))

    for sol in sorted(sols.keys()):
        print(f"sol = {sol}")
        x = []
        y = []
        for build in sorted(builds):
            print(f"{build} -> {builds.get(build)} {builds.get(build)[3].get(sol)} {type({builds.get(build)[3].get(sol)})}")
            if builds.get(build)[3].get(sol) != None:
                x.append(build)
                y.append(builds.get(build)[3].get(sol))   
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=float)
        print(x)
        #print(y)
        #sys.exit(0)

        if len(x) < 2:
            msg = f"glob {sol} - - {len(x)} {0:7.3f} {0:7.3f} {sols.get(sol).directory:20s} \"{sols.get(sol).label}\""
            msg += f"  _WARNING_ not enough data to compute statistics."
        else:
            # On plot: global stats
            mean = np.nanmean(y)
            std  = np.nanstd(y)
            color = sols.get(sol).color
            plt.axhline(y=mean, color=color, linestyle="dotted", linewidth=0.5)
            plt.scatter(x, y, marker=sols.get(sol).marker, color=color,
                        label=sols.get(sol).label + f" {mean:6.2f}+/-{std:5.2f} sec")

            msg = f"all {sol} {x[0]} {x[-1]} {len(x)} {mean:7.3f} {std:7.3f} {sols.get(sol).directory:20s} \"{sols.get(sol).label}\""

            # For monitoring, consider last N points (sliding window)
            N = 10            
            if len(x) < N: N = len(x)
            mean_sw = np.nanmean(y[-N:-1])
            std_sw  = np.nanstd(y[-N:-1])
            msg += f"\ns_w {sol} {x[-N]} {x[-1]} {N} {mean_sw:7.3f} {std_sw:7.3f} {sols.get(sol).directory:20s} \"{sols.get(sol).label}\""


            if lastb > 0:
                lastb_sol = builds.get(lastb)[3].get(sol)
                if lastb_sol == None:
                    msg += f"  _WARNING_  last build ({lastb}) missing!"
                else:
                    lastb_rt = float(builds.get(lastb)[3].get(sol))
                    threshold = mean_sw + 3.0 * std_sw
                    if lastb_rt > threshold:
                        msg += f"  _WARNING_  last build ({lastb}) significantly slower: {lastb_rt:.2f} > {threshold:.2f} ({mean_sw:.3f} + 3 x {std_sw:.3f})"

        if plot['w']:
            fstats.write(msg + "\n")
        print(msg)
        
    if plot['w']:
        fstats.close()

    plt.xlabel("Jenkins build number")
    plt.ylabel("time [sec]")
    font = font_manager.FontProperties(family='DejaVu Sans Mono')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    png = os.path.join(dir, filename)
    plt.savefig(png)
    print(f"Saved plot {png}")
    plt.show()


def plot(plot, args):
    builds = collect_runtimes(plot['sols'], args.input_directory, args.ignore_up_to)
    stats_n_plots(plot, args.output_directory, builds, args.last_build, args.stat_file)


if __name__ == "__main__":

    args = monitoring.check_cl_arguments()

    Solutions = monitoring.define_solutions()

    nufft3i = {
        #'A' : Solutions['LBSSi'],
        'B' : Solutions['LBNi'],
        'C' : Solutions['LBN3i'],
        'D' : Solutions['LBN3cci'],
        'E' : Solutions['LBN3cgi']
    }
    nufft3t = {
        #'A' : Solutions['LBSSi'],
        'B' : Solutions['LBNt'],
        'C' : Solutions['LBN3t'],
        'D' : Solutions['LBN3cct'],
        'E' : Solutions['LBN3cgt']
    }

    # Set 'w' to True for the whole solution (so that all solutions are written)
    plots = (
        {'filename': 'tts_all.png',     'sols': Solutions, 'w': True},
        {'filename': 'tts_nufft3i.png', 'sols': nufft3i,   'w': False},
        {'filename': 'tts_nufft3t.png', 'sols': nufft3t,   'w': False}
    )
    
    for plot_ in plots:
        print("Processing plot ", plot_['filename'])
        plot(plot_, args)
        for sol in sorted(plot_.keys()):
            print(plot_.get(sol))


# To test locally
# ---------------
"""
cd to pypeline

conda activate pype-111

export BUILD_ID=21 GIT_BRANCH=ci-master OUTPUT_DIR=/tmp/ TEST_FSTAT_RT=/tmp/file_rt.tst TEST_IGNORE_UPTO=0 WORK_DIR=/work/backup/ska/ci-jenkins/izar-ska/ TEST_DIR=.

python ./jenkins/tts.py --input_directory ${WORK_DIR}/${GIT_BRANCH}  --output_directory $OUTPUT_DIR --stat_file $TEST_FSTAT_RT  --last_build $BUILD_ID --ignore_up_to $TEST_IGNORE_UPTO

"""
