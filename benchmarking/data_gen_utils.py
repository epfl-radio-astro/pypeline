import numpy as np

import scipy.constants as constants
import imot_tools.math.sphere.grid as grid
import astropy.units as u
import astropy.time as atime
import astropy.coordinates as coord

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.measurement_set as measurement_set

import imot_tools.math.sphere.transform as transform
#################################################################################
# Data Generators
#################################################################################
class RandomDataGen():
    def __init__(self, precision = 32, N_station = 24, order='F'):

        if (precision == 32):
            self.ftype = np.float32
            self.ctype = np.complex64
        elif (precision == 64):
            self.ftype = np.float64
            self.ctype = np.complex128
        else:
            raise Exception("Precision {0} not known".format(precision))

        # input parameters
        if N_station == 24 :
            self.N_height  = 149#248
            self.N_width   = 74#124
            self.N_antenna = 1066#550
            self.N_beam = 24
            self.N_eig  = 4
        if N_station == 37 :
            self.N_height  = 4911#248
            self.N_width   = 2455#124
            self.N_antenna = 1631#550
            self.N_beam = 37
            self.N_eig  = 4
        self.order=order
        frequency = 145e6
        self.wl = constants.speed_of_light / frequency

    # pixel grid must have dimensions (3, N_height, N_width).
    def getPixGrid(self):
        pixGrid = np.random.rand(3, self.N_height, self.N_width)*2-1
        #return pixGrid.astype(self.ftype)
        return pixGrid.astype(self.ftype, order=self.order)

    # visibilities matrix, (N_beam, N_eig) complex-valued eigenvectors.
    def getV(self, i = 0):
        V = np.random.rand(self.N_beam, self.N_eig)-0.5 + 1j*np.random.rand(self.N_beam, self.N_eig)-0.5j
        return V.astype(self.ctype, order=self.order)

    #  (N_antenna, 3) Cartesian instrument geometry.
    def getXYZ(self, i = 0):
        XYZ = np.random.rand(self.N_antenna,3)
        #return XYZ.astype(self.ftype)
        return XYZ.astype(self.ftype, order=self.order)

    # beamforming weights (N_antenna, N_beam) synthesis beamweights.
    def getW(self, i=0):
        W = np.random.rand(self.N_antenna, self.N_beam)-0.5 + 1j*np.random.rand(self.N_antenna, self.N_beam)-0.5j
        return W.astype(self.ctype, order=self.order)

    def getVXYZW(self, i):
        return (self.getV(i),self.getXYZ(i),self.getW(i))
#################################################################################
class SimulatedDataGen():
    def __init__(self, frequency, N_level = 4):
            # parameters
        self.wl = constants.speed_of_light / frequency
        self.N_level = N_level

        self.obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
        T_integration = 8
        self.time = self.obs_start + (T_integration * u.s) * np.arange(3595)
        field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
        self.FoV = np.deg2rad(5)

        # Instrument
        N_station = 24  #24
        self.dev = instrument.LofarBlock(N_station)
        mb_cfg = [(_, _, field_center) for _ in range(N_station)]
        self.mb = beamforming.MatchedBeamformerBlock(mb_cfg)
        self.gram = bb_gr.GramBlock()

        # generation

        self.R = self.dev.icrs2bfsf_rot(self.obs_start, self.time[-1])
        self.sky_model = source.from_tgss_catalog(field_center, self.FoV, N_src=5)
        self.vis = statistics.VisibilityGeneratorBlock(self.sky_model, T_integration, fs=196000, SNR=np.inf)

        _, _, self.px_colat_periodic, self.px_lon_periodic = grid.equal_angle(
            N=self.dev.nyquist_rate(self.wl),
            direction= self.R @ field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
            FoV=self.FoV
        )

        _, _, self.px_colat, self.px_lon = grid.equal_angle(
            N=self.dev.nyquist_rate(self.wl),
            direction=field_center.cartesian.xyz.value,
            FoV=self.FoV
        )
        self.pix_grid = transform.pol2cart(1, self.px_colat, self.px_lon)
        self.T_kernel = np.deg2rad(10)
        self.N_FS = self.dev.bfsf_kernel_bandwidth(self.wl, self.obs_start, self.time[-1])

        self.estimateParams()

    def estimateParams(self):
        # Parameter Estimation
        I_est = bb_pe.IntensityFieldParameterEstimator(self.N_level, sigma=0.95)
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
    def getVXYZWD(self, i):
        try:
            t = self.time[i]
        except:
            print(i,"is an invalid time index")

        XYZ = self.dev(t)
        W = self.mb(XYZ, self.wl)
        S = self.vis(XYZ, W, self.wl)
        G = self.gram(XYZ, W, self.wl)
        D, V, __ = self.I_dp(S, G)
        return (V,XYZ.data, W.data, D)
#################################################################################
class RealDataGen():
    def __init__(self, ms_file, N_level = 4, N_station = 24):
        #24
        self.N_level = N_level
        self.ms = measurement_set.LofarMeasurementSet(ms_file, N_station)
        self.gram = bb_gr.GramBlock()

        self.FoV = np.deg2rad(5)
        self.channel_id = 0
        frequency = self.ms.channels["FREQUENCY"][self.channel_id]
        self.wl = constants.speed_of_light / frequency.to_value(u.Hz)
        self.obs_start, self.obs_end = self.ms.time["TIME"][[0, -1]]
        self.R = self.ms.instrument.icrs2bfsf_rot(self.obs_start, self.obs_end)
        self.sky_model = source.from_tgss_catalog(self.ms.field_center, self.FoV, N_src=4)

        # Instrument        
        _, _, self.px_colat_periodic, self.px_lon_periodic = grid.equal_angle(
            N=self.ms.instrument.nyquist_rate(self.wl),
            direction=self.R @ self.ms.field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
            FoV=self.FoV,
        )

        _, _, self.px_colat, self.px_lon = grid.equal_angle(
             N=self.ms.instrument.nyquist_rate(self.wl),
             direction=self.ms.field_center.cartesian.xyz.value,
             FoV=self.FoV
        )
        print("pix_grid ", self.px_colat.shape, "x", self.px_lon.shape)

        #self.pix_grid = transform.pol2cart(1, self.px_colat, self.px_lon).reshape(3, -1)
        self.pix_grid = transform.pol2cart(1, self.px_colat, self.px_lon)

        self.i = 0
        self.N_FS, self.T_kernel = self.ms.instrument.bfsf_kernel_bandwidth(self.wl, self.obs_start, self.obs_end), np.deg2rad(10)

        self.estimateParams()

    def estimateParams(self):
        # Intensity Field Parameter Estimation
        I_est = bb_pe.IntensityFieldParameterEstimator(self.N_level, sigma=0.95)
        # Sensitivity Field Parameter Estimation
        S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
        for t, f, S in self.ms.visibilities(channel_id=[self.channel_id], time_id=slice(None, None, 200), column="DATA"):
            wl = constants.speed_of_light / f.to_value(u.Hz)
            XYZ = self.ms.instrument(t)
            W = self.ms.beamformer(XYZ, wl)
            G = self.gram(XYZ, W, wl)
            S, _ = measurement_set.filter_data(S, W)

            I_est.collect(S, G)
            S_est.collect(G)
        N_eig, c_centroid = I_est.infer_parameters()
        print("test reco:", N_eig, c_centroid)
        N_eig_S = S_est.infer_parameters()
        self.I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
        self.S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig_S)

    def getPixGrid(self):
        return self.pix_grid

    def getVXYZWD(self, i):
        t, f, S = next(self.ms.visibilities(channel_id=[self.channel_id], time_id=slice(i, i+1, None), column="DATA"))

        XYZ = self.ms.instrument(t)
        W = self.ms.beamformer(XYZ, self.wl)
        G = self.gram(XYZ, W, self.wl)
        S, _ = measurement_set.filter_data(S, W)
        D, V, c_idx = self.I_dp(S, G)

        return (V,XYZ.data, W.data,D)
    def getInputs(self, i):
        t, f, S = next(self.ms.visibilities(channel_id=[self.channel_id], time_id=slice(i, i+1, None), column="DATA"))

        wl = constants.speed_of_light / f.to_value(u.Hz) #self.wl
        XYZ = self.ms.instrument(t)
        W = self.ms.beamformer(XYZ, wl)
        G = self.gram(XYZ, W, wl)
        S, _ = measurement_set.filter_data(S, W)
        D, V, c_idx = self.I_dp(S, G)
        Ds, Vs = self.S_dp(G)

        return (V, Vs, XYZ.data, W.data,D, Ds)
