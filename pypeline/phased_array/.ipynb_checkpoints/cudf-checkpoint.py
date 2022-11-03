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
        
        obstime = df['TIME'].explode().to_numpy()
        unique_time = np.unique(obstime)[time_start:time_stop:time_step]
        
        df2 = df.loc[df['TIME'].isin(unique_time)]
        
        for t in unique_time:
            df_sub = df2.loc[df2['TIME'] == t]
            beam_id_0 = df_sub.ANTENNA1.to_numpy()
            beam_id_1 = df_sub.ANTENNA2.to_numpy()
            data_flag = df_sub.FLAG.explode().explode().to_numpy().reshape(len(df_sub.FLAG),len(df_sub.FLAG[0]),len(df_sub.FLAG[0][0]))
            data = df_sub.data.explode().explode().to_numpy(dtype=np.float32).reshape(len(df_sub.data),len(df_sub.data[0]),len(df_sub.data[0][0])).view(np.complex64)
            
            # We only want XX and YY correlations
            data = np.average(data[:, :, [0, 3]], axis=2)[:, channel_id]
            data_flag = np.any(data_flag[:, :, [0, 3]], axis=2)[:, channel_id]
            
            # DataFrame description of visibility data.
            # Each column represents a different channel.
            S_full_idx = pd.MultiIndex.from_arrays((beam_id_0, beam_id_1), names=("B_0", "B_1"))
            S_full = pd.DataFrame(data=data, columns=channel_id, index=S_full_idx)

            # Drop rows of `S_full` corresponding to unwanted beams.
            beam_id = np.unique(self.instrument._layout.index.get_level_values("STATION_ID"))
            N_beam = len(beam_id)
            i, j = np.triu_indices(N_beam, k=0)
            wanted_index = pd.MultiIndex.from_arrays((beam_id[i], beam_id[j]), names=("B_0", "B_1"))
            index_to_drop = S_full_idx.difference(wanted_index)
            S_trunc = S_full.drop(index=index_to_drop)

            # Depending on the dataset, some (ANTENNA1, ANTENNA2) pairs that have correlation=0 are
            # omitted in the table.
            # This is problematic as the previous DataFrame construction could be potentially
            # missing entire antenna ranges.
            # To fix this issue, we augment the dataframe to always make sure `S_trunc` matches the
            # desired shape.
            index_diff = wanted_index.difference(S_trunc.index)
            N_diff = len(index_diff)

            S_fill_in = pd.DataFrame(
                data=np.zeros((N_diff, len(channel_id)), dtype=data.dtype),
                columns=channel_id,
                index=index_diff,
            )
            S = pd.concat([S_trunc, S_fill_in], axis=0, ignore_index=False).sort_index(
                level=["B_0", "B_1"]
            )

            # Break S into columns and stream out
            t = time.Time(sub_table.calc("MJD(TIME)")[0], format="mjd", scale="utc")
            f = self.channels["FREQUENCY"]
            beam_idx = pd.Index(beam_id, name="BEAM_ID")
            for ch_id in channel_id:
                v = _series2array(S[ch_id].rename("S", inplace=True))
                visibility = vis.VisibilityMatrix(v, beam_idx)
                yield t, f[ch_id], visibility
                

def _series2array(visibility: pd.Series) -> np.ndarray:
    b_idx_0 = visibility.index.get_level_values("B_0").to_series()
    b_idx_1 = visibility.index.get_level_values("B_1").to_series()

    row_map = (
        pd.concat(objs=(b_idx_0, b_idx_1), ignore_index=True)
        .drop_duplicates()
        .to_frame(name="BEAM_ID")
        .assign(ROW_ID=lambda df: np.arange(len(df)))
    )
    col_map = row_map.rename(columns={"ROW_ID": "COL_ID"})

    data = (
        visibility.reset_index()
        .merge(row_map, left_on="B_0", right_on="BEAM_ID")
        .merge(col_map, left_on="B_1", right_on="BEAM_ID")
        .loc[:, ["ROW_ID", "COL_ID", "S"]]
    )

    N_beam = len(row_map)
    S = np.zeros(shape=(N_beam, N_beam), dtype=complex)
    S[data.ROW_ID.values, data.COL_ID.values] = data.S.values
    S_diag = np.diag(S)
    S = S + S.conj().T
    S[np.diag_indices_from(S)] = S_diag
    return S

