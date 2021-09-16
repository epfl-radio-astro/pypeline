import os, sys, timing, argparse
import numpy as np
import cupy as cp
import scipy.sparse as sparse
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as synth_periodic
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth_standard
import dummy_synthesis 
from dummy_synthesis import synthesize, synthesize_stack
from data_gen_utils import RandomDataGen, SimulatedDataGen, RealDataGen

#EO: make it a cl arg
np.random.seed(1234)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",   help="Path to dumping location")
    parser.add_argument("--gpu",      help="Use GPU (default is CPU only)", action="store_true")
    parser.add_argument("--periodic", help="Use periodic algorithm (default is standard one)", action="store_true")
    args = parser.parse_args()

    arch = 'cpu'
    algo = 'standard'

    if args.outdir:
        if not os.path.exists(args.outdir):
            print('fatal: --outdir ('+args.outdir+') must exists if defined.')
            sys.exit(1)
        print("Dumping directory: ", args.outdir)        
    else:
        print("Will not dump anything")
    if args.gpu:
        print("Will use GPU")
        arch = 'gpu'
    if args.periodic:
        print("Will use periodic algorithm")
        algo = 'periodic'

    ###### make simulated dataset ###### 
    precision = 32 # 32 or 64
    data = SimulatedDataGen(frequency = 145e6)
    #data = RealDataGen("/work/scitas-share/SKA/data/gauss4/gauss4_t201806301100_SBL180.MS", N_level = 4, N_station = 37) # n level = # eigenimages
    #data = dummy_synthesis.RandomDataGen()
    ################################### 

    timer = timing.Timer()

    if args.periodic:
        synthesizer = synth_periodic.FourierFieldSynthesizerBlock(data.wl, data.px_colat_periodic, data.px_lon_periodic, data.N_FS, data.T_kernel, data.R, precision)
        synthesizer.set_timer(timer, "Periodic ")
        bfsf_grid = transform.pol2cart(1, data.px_colat_periodic, data.px_lon_periodic)
        icrs_grid = np.tensordot(synthesizer._R.T, bfsf_grid, axes=1)
        print("icrs_grid has type", type(icrs_grid), " and shape ", icrs_grid.shape)
    else:
        pix = data.getPixGrid()
        print("pix has type", type(pix), " and shape ", pix.shape)
        synthesizer = synth_standard.SpatialFieldSynthesizerBlock(data.wl, pix, precision)
        synthesizer.set_timer(timer, "Standard ")

    # iterate though timesteps; increase the range to run through more calls
    stats_combined = None
    stats_normcombined = None

    for t in range(0, 100):

        print("t = {0}".format(t))

        (V, XYZ, W, D) = data.getVXYZWD(t)

        D_r = D.reshape(-1, 1, 1)

        if isinstance(W, sparse.csr.csr_matrix) or isinstance(W, sparse.csc.csc_matrix):
            W = W.toarray()

        if args.gpu:
            XYZ_gpu = cp.asarray(XYZ)
            W_gpu   = cp.asarray(W)
            V_gpu   = cp.asarray(V)
            stats   = synthesizer(V_gpu, XYZ_gpu, W_gpu)
            stats   = stats.get()
        else:
            stats   = synthesizer(V, XYZ, W)

        stats_norm = stats * D_r

        if args.periodic:    # transform the periodic field statistics to periodic eigenimages
            stats      = synthesizer.synthesize(stats)
            stats_norm = synthesizer.synthesize(stats_norm)
            bfsf_grid_ = transform.pol2cart(1, data.px_colat_periodic, data.px_lon_periodic)
            icrs_grid_ = np.tensordot(synthesizer._R.T, bfsf_grid_, axes=1)
            print(np.max(np.absolute(icrs_grid_ - icrs_grid)))
        """
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
        """

        try:    stats_combined += stats
        except: stats_combined  = stats

        try:    stats_normcombined += stats_norm
        except: stats_normcombined  = stats_norm


    # Dump combined stats
    if args.outdir:
        outname = ''
        #outname = '_' + algo + '_' + arch
        outcomb = os.path.join(args.outdir, 'stats_combined' + outname + '.npy')
        outnormcomb = os.path.join(args.outdir, 'stats_normcombined' + outname + '.npy')
        with open(outcomb, 'wb') as f:
            np.save(f, stats_combined)
            print("Wrote ", outcomb)
        with open(outnormcomb, 'wb') as f:
            np.save(f, stats_normcombined)
            print("Wrote ", outnormcomb)

    print(timer.summary())
