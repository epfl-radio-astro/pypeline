import os, sys, timing, argparse
import numpy as np
import cupy as cp
import scipy.sparse as sparse
import nvtx
import multiprocessing as mp
import time
from functools import partial

import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as synth_periodic
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth_standard
import dummy_synthesis 
from dummy_synthesis import synthesize, synthesize_stack
from data_gen_utils import RandomDataGen, SimulatedDataGen, RealDataGen

#EO: make it a cl arg
np.random.seed(1234)


def worker_info(wid):
    print("printing info on pool worker ", mp.current_process().name, " (input ",wid,")")

    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    time.sleep(10)


@nvtx.annotate(color="yellow")
def t_stats(t, data, args, synthesizer):

    with cp.cuda.profile():
        #pass
        
        with nvtx.annotate("Get data", color="blue"):
            (V, XYZ, W, D) = data.getVXYZWD(t)

        D_r = D.reshape(-1, 1, 1)
    
        if isinstance(W, sparse.csr.csr_matrix) or isinstance(W, sparse.csc.csc_matrix):
            with nvtx.annotate("sparse", color="aqua"):
                W = W.toarray()

        if args.gpu:
            with nvtx.annotate("XYZ,W,V asarray", color="green"):
                XYZ_gpu = cp.asarray(XYZ)
                W_gpu   = cp.asarray(W)
                V_gpu   = cp.asarray(V)
            with nvtx.annotate("Synthesizer", color="red"):
                stats   = synthesizer(V_gpu, XYZ_gpu, W_gpu)
                stats   = stats.get()
        else:
            with nvtx.annotate("Synthesizer", color="red"):
                stats   = synthesizer(V, XYZ, W)
            
        stats_norm = stats * D_r

        if args.periodic:    # transform the periodic field statistics to periodic eigenimages
            stats      = synthesizer.synthesize(stats)
            stats_norm = synthesizer.synthesize(stats_norm)

    return (stats, stats_norm)


# Dump data to args.outdir if defined
def dump_data(stats, filename):
    if args.outdir:
        fp = os.path.join(args.outdir, filename + '.npy')
        with open(fp, 'wb') as f:
            np.save(f, stats)
            print("Wrote ", fp)


if __name__ == "__main__":

    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",   help="Path to dumping location (no dumps if not set)")
    parser.add_argument("--gpu",      help="Use GPU (default is CPU only)", action="store_true")
    parser.add_argument("--periodic", help="Use periodic algorithm (default is standard one)", action="store_true")
    parser.add_argument("--bench",    help="Run a multi-processing benchmark", action="store_true")
    parser.add_argument("--t_range",  help="Number of time steps to consider", type=int, default=10)

    args = parser.parse_args()

    t_range = args.t_range
    print("t_range =", t_range)

    arch = 'cpu'
    algo = 'standard'

    if args.outdir:
        if not os.path.exists(args.outdir):
            print('fatal: --outdir ('+args.outdir+') must exists if defined.')
            sys.exit(1)
        print("Dumping directory: ", args.outdir)        
    else:
        print("Will not dump anything, --outdir not set.")
    if args.gpu:
        print("Will use GPU")
        arch = 'gpu'
    if args.periodic:
        print("Will use periodic algorithm")
        algo = 'periodic'

    precision = 32 # 32 or 64

    #data = SimulatedDataGen(frequency = 145e6)
    MS_file = "/work/scitas-share/SKA/data/gauss4/gauss4_t201806301100_SBL180.MS"
    data = RealDataGen(MS_file, N_level = 4, N_station = 24) # n level = # eigenimages
    #data = dummy_synthesis.RandomDataGen()

    timer = timing.Timer()


    if args.periodic:
        synthesizer = synth_periodic.FourierFieldSynthesizerBlock(data.wl, data.px_colat_periodic, data.px_lon_periodic, data.N_FS, data.T_kernel, data.R, precision)
        synthesizer.set_timer(timer, "Periodic ")
        bfsf_grid = transform.pol2cart(1, data.px_colat_periodic, data.px_lon_periodic)
        grid = np.tensordot(synthesizer._R.T, bfsf_grid, axes=1)
    else:
        grid = data.getPixGrid()
        synthesizer = synth_standard.SpatialFieldSynthesizerBlock(data.wl, grid, precision)
        synthesizer.set_timer(timer, "Standard ")

    #print("grid has type", type(grid), " and shape ", grid.shape)


    # iterate though timesteps; increase the range to run through more calls

    stats_combined = None
    stats_normcombined = None

    # NCPUS: size of pool of workers
    # Watch out potential conflit with backgroud blas multi-threading
    # (e.g. you might need to export OPENBLAS_NUM_THREADS=1)
    NCPUS = len(os.sched_getaffinity(0))
    print("NCPUS = len(os.sched_getaffinity(0) = ", NCPUS)
    print("OPENBLAS_NUM_THREADS =", os.environ.get('OPENBLAS_NUM_THREADS'))
    

    ### Serial

    tic = time.perf_counter()
    with nvtx.annotate("Main loop", color="purple"):
        stats_combined = None
        stats_normcombined = None
        with cp.cuda.profile():
            for t in range(0, t_range):
                (stats_, stats_norm_) = t_stats(t, data, args, synthesizer)
                try:    stats_combined += stats_
                except: stats_combined  = stats_
                try:    stats_normcombined += stats_norm_
                except: stats_normcombined  = stats_norm_

    dump_data(stats_combined, 'stats_combined')
    dump_data(stats_normcombined, 'stats_normcombined')
    dump_data(grid, 'grid')
    toc = time.perf_counter()
    print(f"Serial {toc-tic:12.6f} sec")


    ### Multi-processing

    if args.bench:

        ncpus = 2

        while ncpus <= NCPUS:

            tic = time.perf_counter()

            with mp.Pool(ncpus) as pool:
                t_stats_partial = partial(t_stats, data=data, args=args, synthesizer=synthesizer)
                all_stats = pool.map(t_stats_partial, range(0,t_range))
                #pool.map(worker_info, range(0,t_range))

            toc = time.perf_counter()

            stats_ = None
            stats_norm_ = None
            for stats_tup in all_stats:
                try:    stats_ = np.add(stats_, stats_tup[0])
                except: stats_ = stats_tup[0]
                try:    stats_norm_ = np.add(stats_norm_, stats_tup[1])
                except: stats_norm_ = stats_tup[1]
            
            s_equal  = np.array_equal(stats_, stats_combined)
            sn_equal = np.array_equal(stats_norm_, stats_normcombined)
            s_close  = np.allclose(stats_, stats_combined, atol=1e-06)

            print(f'M-P {ncpus:2d} {toc-tic:12.6f} sec; Stats equal? {s_equal}/{sn_equal}')

            dump_data(stats_, 'stats_combined_mp' + f'{ncpus:02d}' )

            ncpus *= 2

    print(timer.summary())
