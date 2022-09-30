import sys
import os
import argparse
import numpy
import bluebild


PROCESSING_UNIT = {'auto': bluebild.ProcessingUnit.AUTO,
                   'none': None,
                   'cpu' : bluebild.ProcessingUnit.CPU,
                   'gpu' : bluebild.ProcessingUnit.GPU}


def dump_data(stats, filename, outdir):
    if outdir:
        fp = os.path.join(outdir, filename + '.npy')
        with open(fp, 'wb') as f:
            numpy.save(f, stats)
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

    args.processing_unit = PROCESSING_UNIT[args.processing_unit]

    return args
