import casacore.tables as ct
import astropy.units as u
import pathlib
import numpy as np
import sys

def examine(filename):

	print("ms file:", filename)

	path = pathlib.Path(filename).absolute()

	if not path.exists():
	    raise FileNotFoundError(f"{filename} does not exist.")

	if not path.is_dir():
	    raise NotADirectoryError(f"{filename} is not a directory, so cannot be an MS file.")
	####### ALL COLUMN NAMES ##################################################

	query = f"select * from {filename}"
	column_names = ct.taql(query).colnames()
	print (column_names)

	####### FIELD CENTER #################################################

	query = f"select REFERENCE_DIR, TIME from {filename}::FIELD"
	table = ct.taql(query)

	lon, lat = table.getcell("REFERENCE_DIR", 0).flatten()
	print ("Field Center")
	print (f'RA is {lon} rad ({np.rad2deg(lon)} deg) and dec is {lat} ({np.rad2deg(lat)} deg) rad')
	#######################################################################
	######### FREQUENCY ###################################################
	query = f"select CHAN_FREQ, CHAN_WIDTH from {filename}::SPECTRAL_WINDOW"
	table = ct.taql(query)

	f = table.getcell("CHAN_FREQ", 0).flatten() * u.Hz
	f_w = table.getcell("CHAN_WIDTH", 0) * u.Hz
	print("\nFREQUENCY")
	print (f'Frequencies are {f}')
	print (f'Channel Width is {f_w}')
	print (f'Number of frequencies are {len(f)}')
	#########################################################################
	######### TIME ##########################################################

	query = f"select * from {filename}"
	table = ct.taql(query)

	t = np.unique(table.calc("MJD(TIME)"))
	print ("\nTIME")
	print (f'Time start :{t[0]} Time End: {t[-1]} Time Steps: {len(t)}')
	print (f'Time in days: {t[-1] - t[0]}, in hours: {(t[-1] - t[0]) * 24.0}, in minutes:  {(t[-1] - t[0])* 24 * 60}, in seconds: {(t[-1] - t[0])* 24 * 60 * 60}')
	#########################################################################
	######### INSTRUMENT ####################################################

	print ("\nINSTRUMENT")
	query = f"select POSITION from {filename}::ANTENNA"
	table = ct.taql(query)
	station_mean = table.getcol("POSITION")

	N_station = len(station_mean)
	print (f'Number of Stations: {N_station}')
	print (f'Antenna Positions: \n {station_mean}')

	###########################################################################
	######### FLAGGING ########################################################

	print ("\nFLAGGING")
	query = (f"select * from {filename}")#" where MJD(TIME) in " # "MJD(TIME)" instead of TIME
            #f"(select unique MJD(TIME) from {self._msf} limit {time_start}:{time_stop}:{time_step})" # MJD(TIME) instead of TIME
            
	table = ct.taql(query)
	subTableIndex = 0
	for sub_table in table.iter("TIME", sort=True): # MJD(TIME) instead of TIME
		subTableIndex += 1

		beam_id_0 = sub_table.getcol("ANTENNA1")  # (N_entry,)
		beam_id_1 = sub_table.getcol("ANTENNA2")  # (N_entry,)
		data_flag = sub_table.getcol("FLAG")  # (N_entry, N_channel, 4)
		data = sub_table.getcol("CORRECTED_DATA")  # (N_entry, N_channel, 4)
		uvw = sub_table.getcol("UVW")
		uvw *= -1

		# print ("flag checks", np.shape(data_flag), np.count_nonzero(data_flag[:,:, 0]), np.count_nonzero(data_flag[:,:, 1]), np.count_nonzero(data_flag[:,:, 2]), np.count_nonzero(data_flag[:,:, 3]) )
		# We only want XX and YY correlations
		data = np.average(data[:, :, [0, 3]], axis=2)[:, np.arange(len(f))]
		data_flag = np.any(data_flag[:, :, [0, 3]], axis=2)[:, np.arange(len(f))]

		print (f'{subTableIndex} Data shape: {data.shape}, flagged data shape:{data_flag.shape}, nonzero data: {np.count_nonzero(data)}, percentage of non zero data: {np.count_nonzero(data)/(np.count_nonzero(data) + np.count_nonzero(data ==0))}')

		# Set broken visibilities to 0
		# Set Flagging off for MeerKAT Data
		data[data_flag] = 0
		print (f'Data shape: {data.shape}, flagged data shape:{data_flag.shape}, nonzero data: {np.count_nonzero(data)}, percentage of non zero data: {np.count_nonzero(data)/(np.count_nonzero(data) + np.count_nonzero(data ==0))}')


	

ms_files= [sys.argv[1]]
#ms_files = ["/scratch/izar/krishna/MWA/MS_Files/1133149192-084-085_Sun_10s_cal.ms"]  #1133149192-103-104_Sun_10s_cal.ms  1133149192-125-126_Sun_10s_cal.ms  1133149192-153-154_Sun_10s_cal.ms  1133149192-187-188_Sun_10s_cal.ms  Solar_testcase.zip1133149192-093-094_Sun_10s_cal.ms  1133149192-113-114_Sun_10s_cal.ms  1133149192-139-140_Sun_10s_cal.ms  1133149192-169-170_Sun_10s_cal.ms  1133149192-187-188_Sun_10s.ms"]
for m in ms_files: examine(m)



