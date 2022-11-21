import sys
import os
import argparse
from pathlib import Path
import numpy
import json
import bluebild


PROCESSING_UNIT = {'auto': bluebild.ProcessingUnit.AUTO,
                   'none': None,
                   'cpu' : bluebild.ProcessingUnit.CPU,
                   'gpu' : bluebild.ProcessingUnit.GPU}


def dump_json(v_shape, w_shape, grid_shape, t_ifim, t_vis, t_idp, t_imfs, filename, outdir):

    Nb, Ne     = v_shape
    Na, Nb     = w_shape
    Nc, Nh, Nw = grid_shape

    stats = { 
        "timings": {
            'ifim': t_ifim,
            'ivis': t_vis,
            'idp':  t_idp,
            'imfs': t_imfs
        },
        "setup" : {
            'Na': Na, 'Nb': Nb, 'Nc': Nc,
            'Ne': Ne, 'Nh': Nh, 'Nw': Nw
        }
    }

    if outdir:
        with open(os.path.join(outdir, filename), "w") as outfile:
            outfile.write(json.dumps(stats, indent=4))


def compare_solutions(ref, sol):
    print("-R- reference:", ref, "\n-R- solution :", sol)
    with open(ref, 'r') as openfile: ref_stats = json.load(openfile)
    with open(sol, 'r') as openfile: sol_stats = json.load(openfile)
    speedup_ifim = ref_stats['timings']['ifim'] / sol_stats['timings']['ifim']
    speedup_ivis = ref_stats['timings']['ivis'] / sol_stats['timings']['ivis']
    speedup_idp  = ref_stats['timings']['idp']  / sol_stats['timings']['idp']
    speedup_imfs = ref_stats['timings']['imfs'] / sol_stats['timings']['imfs']
    print(f"-R- speedups : ifim = {speedup_ifim:5.1f}, ivis = {speedup_ivis:3.1f}, ",
          f"idp = {speedup_idp:4.1f}, imfs = {speedup_imfs:5.1f}")

    # Compare imfs npy
    ref_std = numpy.load(ref.rsplit('.', 1)[0] + '_imfs_std.npy', allow_pickle=True)
    ref_lsq = numpy.load(ref.rsplit('.', 1)[0] + '_imfs_lsq.npy', allow_pickle=True)
    sol_std = numpy.load(sol.rsplit('.', 1)[0] + '_imfs_std.npy', allow_pickle=True)
    sol_lsq = numpy.load(sol.rsplit('.', 1)[0] + '_imfs_lsq.npy', allow_pickle=True)
    rmse_lsq, max_abs_err_lsq = stats_image_diff(ref_lsq, sol_lsq)
    rmse_std, max_abs_err_std = stats_image_diff(ref_std, sol_std)
    print(f"-R- LSQ stats: rmse = {rmse_lsq:.2E}, max abs err = {max_abs_err_lsq:.2E}")
    print(f"-R- STD stats: rmse = {rmse_std:.2E}, max abs err = {max_abs_err_std:.2E}")


def dump_data(stats, filename, outdir):
    if outdir:
        fp = os.path.join(outdir, filename + '.npy')
        with open(fp, 'wb') as f:
            numpy.save(f, stats)

def dump_stats(stats, filename, outdir):
    if outdir:
        I_std, I_lsq = stats.as_image()
        fp = os.path.join(outdir, filename + '_lsq.npy')
        with open(fp, 'wb') as f:
            numpy.save(f, I_lsq.data)
            print("-I- wrote", fp)
        fp = os.path.join(outdir, filename + '_std.npy')
        with open(fp, 'wb') as f:
            numpy.save(f, I_std.data)
            print("-I- wrote", fp)


def check_args(args_in):
    #print("args =\n", args_in)
    parser = argparse.ArgumentParser(args_in)
    parser.add_argument("--outdir", help="Path to dumping location (no dumps if not set)")
    parser.add_argument("--processing_unit",  help="Bluebild processing unit (for ctx definition)", choices=['auto', 'cpu', 'gpu', 'none'], default='auto')
    parser.add_argument("--precision", help="Floating point calculation precision", choices=['single', 'double'], default='double')
    args = parser.parse_args()
    if args.outdir:
        if not os.path.exists(args.outdir):
            print('-E- --outdir ('+args.outdir+') must exist if set')
            sys.exit(1)
        print("-I- dumping directory: ", args.outdir)
    else:
        print("-W- will not dump anything since --outdir was not set")

    args.processing_unit_name = args.processing_unit
    args.processing_unit  = PROCESSING_UNIT[args.processing_unit]

    return args


# Compute the RMSE between two image
def stats_image_diff(image1, image2):
    assert image1.shape == image2.shape, \
        f"-E- shapes of images to compare do not match {image1.data.shape} vs {image2.data.shape}"
    print("-I- comparing images with shape ", image1.shape)
    diff = image2 - image1
    rmse = numpy.sqrt(numpy.sum(diff**2)/numpy.size(diff))
    max_abs = numpy.max(numpy.abs(diff))
    return rmse, max_abs
