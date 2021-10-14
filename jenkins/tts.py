import sys
import os
import re
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import getopt
import math

def scan(dir):
    builds = {}
    with os.scandir(dir) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_\d+", entry.name):
                #print(entry.name)
                info = re.split('T|Z_', entry.name)
                build = int(info[2])
                if build > 50:
                    builds[build] = [info[0], info[1], entry.name, tts.copy()]
    return builds


def collect_runtimes(dir, builds):
    for build in sorted(builds.keys()):
        for sol in sorted(sols.keys()):
            soldir = os.path.join(dir, builds.get(build)[2], sols.get(sol).directory)
            #print(f">Scanning solution {sol} in build {build:4d}: {soldir}")
            if os.path.isdir(soldir):
                with os.scandir(soldir) as it:
                    for entry in it:
                        if re.match(r"^slurm-\d+.out", entry.name):
                            slurm = os.path.join(soldir, entry.name)
                            print(f"slurm {slurm}")
                            with open(slurm, "r") as file:
                                for line in file:
                                    if re.search('Serial', line):
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


def stats_n_plots(dir, builds, lastb, fstat):

    isin = check_presence_lastb(builds, lastb)
    print(f"lastb is in builds?", isin)

    #flist = font_manager.get_fontconfig_fonts()
    #names = [font_manager.FontProperties(fname=fname).get_name() for fname in flist]
    #print(names)

    fstats = open(fstat, 'w')
    print(f"Writing statistics to file {fstat}")

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

        fstats.write(msg + "\n")
        print(msg)
        
                
    fstats.close()

    plt.xlabel("Jenkins build number")
    plt.ylabel("Main loop time [sec]")
    font = font_manager.FontProperties(family='DejaVu Sans Mono')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    png = os.path.join(dir, 'runtimes_all.png')
    plt.savefig(png)
    print(f"Saved plot {png}")
    #plt.show()


def main(argv):

    indir  = ''
    outdir = ''
    lastb  = -1
    fstat  = ''

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:b:f:")
    except getopt.GetoptError:
        print(f'{argv[0]} -i </path/to/input/directory><input> -o </path/to/output/directory> [-b <last build id>]')
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print(f'{argv[0]} -i </path/to/input/directory> -o </path/to/output/directory> -b <last build id> -f </path/to/filestat')
            sys.exit(1)
        elif opt == '-i':
            indir = arg
        elif opt in '-o':
            outdir = arg
        elif opt in '-b':
            lastb = int(arg)
        elif opt in '-f':
            fstat = arg
    if indir == '':
        print(f'Fatal: argument -i </path/to/input/directory> not found.')
        sys.exit(1)
    if outdir == '':
        print(f'Fatal: argument -o </path/to/output/directory> not found.')
        sys.exit(1)
    if fstat == '':
        print(f'Fatal: argument -f </path/to/filestat> not found.')
        sys.exit(1)

    print(f"indir  is {indir}")
    print(f"outdir is {outdir}")
    print(f"fstat  is {fstat}")

    builds = scan(indir)
    builds = collect_runtimes(indir, builds)
    stats_n_plots(outdir, builds, lastb, fstat)


if __name__ == "__main__":

    tts = {}

    Solution = collections.namedtuple('Solution', ['directory', 'label', 'marker', 'color'])
    SC = Solution(directory='test_standard_cpu', label='Std CPU', marker='o', color='blue')
    SG = Solution(directory='test_standard_gpu', label='Std GPU', marker='o', color='red')

    # Solutions to plot
    sols = {
        'SC': SC,
        'SG': SG
    }
    for sol in sorted(sols.keys()):
        print(sols.get(sol))


    main(sys.argv)
