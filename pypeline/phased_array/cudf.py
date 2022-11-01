# #############################################################################
# cudf.py
# ==================
# Author : Arpan Das [arpan.das@epfl.ch]
# #############################################################################

"""
Cudf file readers and tools.
"""

import dask
import dask.array as da
import dask_cudf
import cudf
import numpy as np

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.data_gen.statistics as vis

@chk.check(
    dict(S=chk.is_instance(vis.VisibilityMatrix), W=chk.is_instance(beamforming.BeamWeights))
)
def filter_data(S, W):
    """
    Fix mis-matches to make data streams compatible.

    Visibility matrices from MS files typically include broken beams and/or may not match beams
    specified in beamforming matrices.
    This mis-match causes computations further down the imaging pypeline to be less efficient or
    completely wrong.
    This function applies 2 corrections to visibility and beamforming matrices to make them
    compliant:

    * Drop beams in `S` that do not appear in `W`;
    * Insert 0s in `W` where `S` has broken beams.

    Parameters
    ----------
    S : :py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`
        (N_beam1, N_beam1) visibility matrix.
    W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
        (N_antenna, N_beam2) beamforming matrix.

    Returns
    -------
    S : :py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`
        (N_beam2, N_beam2) filtered visibility matrix.
    W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
        (N_antenna, N_beam2) filtered beamforming matrix.
    """
    # Stage 1: Drop beams in S that do not appear in W
    beam_idx1 = S.index[0]
    beam_idx2 = W.index[1]
    beams_to_drop = beam_idx1.difference(beam_idx2)
    beams_to_keep = beam_idx1.drop(beams_to_drop)

    mask = np.any(beam_idx1.values.reshape(-1, 1) == beams_to_keep.values.reshape(1, -1), axis=1)
    S_f = vis.VisibilityMatrix(data=S.data[np.ix_(mask, mask)], beam_idx=beam_idx1[mask])

    # Stage 2: Insert 0s in W where S had broken beams
    broken_beam_idx = beam_idx2[np.isclose(np.sum(S_f.data, axis=1), 0)]
    mask = np.any(beam_idx2.values.reshape(-1, 1) == broken_beam_idx.values.reshape(1, -1), axis=1)

    if np.any(mask) and sparse.isspmatrix(W.data):
        w_lil = W.data.tolil()  # for efficiency
        w_lil[:, mask] = 0
        w_f = w_lil.tocsr()
    else:
        w_f = W.data.copy()
        w_f[:, mask] = 0
    W_f = beamforming.BeamWeights(data=w_f, ant_idx=W.index[0], beam_idx=beam_idx2)

    return S_f, W_f

class Cudfparquet:
    """
    Cudf parquet data format reader.

    This class contains the high-level interface all sub-classes must implement.

    Focus is given to reading parquet files from phased-arrays for the moment (i.e, not dish arrays).
    """
    
    @chk.check("file_name", chk.is_instance(str))
    def __init__(self, file_name):
        """
        Parameters
        ----------
        file_name : str
            Name of the MS file.
        """
        path = pathlib.Path(file_name).absolute()

        if not path.exists():
            raise FileNotFoundError(f"{file_name} does not exist.")

        if not path.is_dir():
            raise NotADirectoryError(f"{file_name} is not a directory, so cannot be an parquet file.")

        self._cudf = str(path)

        # Buffered attributes
        self._field_center = None
        self._channels = None
        self._time = None
        self._instrument = None
        self._beamformer = None
        
    @property
    def field_center(self):
        """
        Returns
        -------
        :py:class:`~astropy.coordinates.SkyCoord`
            Observed field's center.
        """

        if self._field_center is None:
            df = cudf.read_parquet(self._cudf+'_FIELD'+'.parquet')

            lon, lat = df['REFERENCE_DIR'].explode()[0]
            self._field_center = coord.SkyCoord(ra=lon * u.rad, dec=lat * u.rad, frame="icrs")

        return self._field_center
    
    @property
    def channels(self):
        """
        Frequency channels available.

        Returns
        -------
        :py:class:`~astropy.table.QTable`
            (N_channel, 2) table with columns

            * CHANNEL_ID : int
            * FREQUENCY : :py:class:`~astropy.units.Quantity`
        """
        
        if self._channels is None:
            df = cudf.read_parquet(self._cudf+'_SPECTRAL_WINDOW'+'.parquet')

            f = df['CHAN_FREQ'].explode()[0] * u.Hz
            f_id = range(len(f))
            self._channels = tb.QTable(dict(CHANNEL_ID=f_id, FREQUENCY=f))

        return self._channels
    
    @property
    def time(self):
        """
        Visibility acquisition times.

        Returns
        -------
        :py:class:`~astropy.table.QTable`
            (N_time, 2) table with columns

            * TIME_ID : int
            * TIME : :py:class:`~astropy.time.Time`
        """
        if self._time is None:
            df = cudf.read_parquet(self._cudf+'.parquet')
            
            time_array = df['TIME'].explode().to_numpy()
            t = time.Time(np.unique(time_array), format="mjd", scale="utc")
            t_id = range(len(t))
            self._time = tb.QTable(dict(TIME_ID=t_id, TIME=t))

        return self._time
    
    @property
    def instrument(self):
        """
        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock`
            Instrument position computer.
        """
        raise NotImplementedError

    @property
    def beamformer(self):
        """
        Each dataset has been beamformed in a specific way.
        This property outputs the correct beamformer to compute the beamforming weights.

        Returns
        -------
        :py:class:`~pypeline.phased_array.beamforming.BeamformerBlock`
            Beamweight computer.
        """
        raise NotImplementedError

    @chk.check(
        dict(
            channel_id=chk.accept_any(chk.has_integers, chk.is_instance(slice)),
            time_id=chk.accept_any(chk.is_integer, chk.is_instance(slice)),
            column=chk.is_instance(str),
        )
    )
    def visibilities(self, channel_id, time_id, column):
        
        """
        Extract visibility matrices.

        Parameters
        ----------
        channel_id : array-like(int) or slice
            Several CHANNEL_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.channels`.
        time_id : int or slice
            Several TIME_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.time`.
        column : str
            Column name from MAIN table where visibility data resides.

            (This is required since several visibility-holding columns can co-exist.)

        Returns
        -------
        iterable

            Generator object returning (time, freq, S) triplets with:

            * time (:py:class:`~astropy.time.Time`): moment the visibility was formed;
            * freq (:py:class:`~astropy.units.Quantity`): center frequency of the visibility;
            * S (:py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`)
        """
        df = cudf.read_parquet(self._cudf+'.parquet')
        
        if column not in df.columns:
            raise ValueError(f"column={column} does not exist in {self._cudf}::MAIN.")
            
        channel_id = self.channels["CHANNEL_ID"][channel_id]
        if chk.is_integer(time_id):
            time_id = slice(time_id, time_id + 1, 1)
        N_time = len(self.time)
        time_start, time_stop, time_step = time_id.indices(N_time)