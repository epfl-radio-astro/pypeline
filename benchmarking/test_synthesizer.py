
import numpy as np
import sys
import scipy.constants as constants
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
import astropy.units as u
import astropy.time as atime
import astropy.coordinates as coord
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth
#import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain_optimized as synth_test
import timing
import dummy_synthesis 
from dummy_synthesis import RandomDataGen, synthesize, synthesize_stack


class SimulatedDataGen():
    def __init__(self, wl):
        self.wl = wl
        obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
        field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
        FoV = np.deg2rad(5)

        # Instrument
        N_station = 24  #24
        self.dev = instrument.LofarBlock(N_station)
        mb_cfg = [(_, _, field_center) for _ in range(N_station)]
        self.mb = beamforming.MatchedBeamformerBlock(mb_cfg)
        self.gram = bb_gr.GramBlock()

        # generation
        T_integration = 8
        sky_model = source.from_tgss_catalog(field_center, FoV, N_src=20)
        self.vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
        _, _, px_colat, px_lon = grid.equal_angle(
            N=self.dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=FoV
        )
        self.pix_grid = transform.pol2cart(1, px_colat, px_lon)
        self.time = obs_start + (T_integration * u.s) * np.arange(3595)
        self.i = 0

        self.estimateParams()

    def estimateParams(self):
        # Parameter Estimation
        I_est = bb_pe.IntensityFieldParameterEstimator(4, sigma=0.95)
        for t in self.time[::200]:
            XYZ = self.dev(t)
            W = self.mb(XYZ, self.wl)
            S = self.vis(XYZ, W, self.wl)
            G = self.gram(XYZ, W, self.wl)

            I_est.collect(S, G)

        N_eig, c_centroid = I_est.infer_parameters()
        self.I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)

    def getPixGrid(self):
        return self.pix_grid
    def getVXYZW(self, i):
        try:
            t = self.time[i]
        except:
            print(i,"is an invalid time index")

        XYZ = self.dev(t)
        W = self.mb(XYZ, self.wl)
        S = self.vis(XYZ, W, self.wl)
        G = self.gram(XYZ, W, self.wl)
        __, V, __ = self.I_dp(S, G)
        return (V,XYZ.data, W.data)


if __name__ == "__main__":

    ###### make simulated dataset ###### 
    # parameters
    frequency = 145e6
    wl = constants.speed_of_light / frequency
    precision = 32 # 32 or 64

    # random or simulated dataset
    #data = SimulatedDataGen(wl)
    data = dummy_synthesis.RandomDataGen()


    ################################### 

    # timer
    timer = timing.Timer()
    # function to turn timing on or off
    #timer.off() 

    pix = data.getPixGrid()

    # The Standard Synthesis Kernel
    synthesizer      = synth.SpatialFieldSynthesizerBlock(wl, pix, precision)
    synthesizer.set_timer(timer,)

    # a copy of the Standard Synthesis Kernel that will be used for testing
    #synthesizer_test = synth_test.SpatialFieldSynthesizerOptimizedBlock(wl, pix, precision)
    #synthesizer_test.set_timer(timer, "Test ")

    # iterate though timesteps
    # increase the range to run through more calls
    for t in range(1,100):

        (V, XYZ, W) = data.getVXYZW(t)

        if t % 10 == 0:
            print("t = {0}".format(t))


        #do some copying for inputs which get modified by the synthesizer
        #V1 = np.copy(V) 
        #XYZ1 = np.copy(XYZ) 
        #V2 = np.copy(V)
        #XYZ2 = np.copy(XYZ) 
        
        # call the Bluebild Standard Synthesis Kernel
        stat_bbss = synthesizer(V,XYZ,W)
        
        #eo call opt ss kernel
        #stat_bbss_gpu = synthesizer_test(V1, XYZ1, W)

        #print("Difference in results between standard & optimized synthesizers:", np.average(stat_bbss_gpu - stat_bbss))


        # call the dummy synthesis kernal
        '''
        stat_dum  = dummy_synthesis.synthesize(pix,V1,XYZ1,W, wl)

        # call an alternate dummy synthesis kernel which reshapes the matrices
        stat_sdum = dummy_synthesis.synthesize_reshape(pix,V2,XYZ2,W, wl)

        # call an alternate dummy synthesis kernel which uses a special ZGEMM
        stat_zdum = dummy_synthesis.synthesize_reshape(pix,V2,XYZ2,W, wl)

        print("Difference in results between dummy & optimized synthesizers:", np.average( stat_dum - stat_bbss))
        print("Avg diff between dummy & dummy reshape synthesizers:", np.average( stat_dum - stat_sdum))
        print("Avg diff between dummy & ZGEMM synthesizers:", np.max( np.abs(stat_dum - stat_zdum)))
        '''
    print(timer.summary())
