import casacore.tables as ct
import astropy.units as u
import pathlib
import numpy as np

def examine(filename):

	print("ms file:", filename)

	path = pathlib.Path(filename).absolute()

	if not path.exists():
	    raise FileNotFoundError(f"{filename} does not exist.")

	if not path.is_dir():
	    raise NotADirectoryError(f"{filename} is not a directory, so cannot be an MS file.")

	# ALL TABLES
	print ("Columns are:")
	print(ct.taql(f"select * from {filename}").colnames())

	#####################################
	# _field_center #####################
	#####################################
	print("\n_field_center from measurement set requires \"REFERENCE_DIR\" and \"TIME\" from {0}::FIELD".format(filename))
	query = f"select REFERENCE_DIR, TIME from {filename}::FIELD"
	table = ct.taql(query)
	lon, lat = table.getcell("REFERENCE_DIR", 0).flatten()
	print ("Latitude :{0}, Longitude: {1}".format(lon, lat))

	####################################
	# _CHANNELS ########################
	####################################
	print("\n_channel from measurement set requires \"CHAN_FREQ\" and \"CHAN_WIDTH\" from {0}::SPECTRAL_WINDOW".format(filename))
	query = f"select CHAN_FREQ, CHAN_WIDTH from {filename}::SPECTRAL_WINDOW"
	table = ct.taql(query)
	freq = table.getcell("CHAN_FREQ", 0).flatten() * u.Hz
	print ("No. of Frequencies: {0}\nFrequencies are :".format(freq.shape[0]))
	print(freq)

	#################################### 
	# _time ############################
	####################################

	print("\n_time from measurement set requires unique times from {0}, but we ask for all tables and then query for unique time as it is faster".format(filename))
	query = f"select * from {filename}"
	table = ct.taql(query)

	t = np.unique(table.calc("MJD(TIME)"))
	print ("Number of integration midpoints: {0}\nTime:\n from {1} to {2}".format(t.shape[0], t[0], t[-1]))

	####################################
	# _instrument ######################
	####################################

	print("\n_instrument from the MWA Measurement set class requires \"POSITION\" from {0}::ANTENNA ".format(filename))

	query = f"select POSITION from {filename}::ANTENNA"
	table = ct.taql(query)
	station_mean = table.getcol("POSITION")

	print ("Antenna Number: {0} \nAntenna Mean Positions:\n".format(station_mean.shape[0]), station_mean)
	np.savetxt("MeerKAT_Antennae_Positions.txt", np.vstack((np.arange(station_mean.shape[0]), np.arange(station_mean.shape[0]), station_mean.T)).T, ["%i","%i","%10.3f","%10.3f","%10.3f"], delimiter=",", header="STATION_ID,ANTENNA_ID,X,Y,Z")

	
	print ("FLAG info: ")

	query = (f"select * from {filename}") # where TIME in "
            #f"(select unique TIME from {filename} limit {0}:{182}:{200})")
	table = ct.taql(query)
	subtableindex =0
	for sub_table in table.iter("TIME", sort=True):
		print("subtableindex" ,subtableindex)
		subtableindex += 1
		flag = sub_table.getcol("FLAG")
		#flag_cat = table.getcol("FLAG_CATEGORY")
		flag_row = sub_table.getcol("FLAG_ROW")
		print("FLAG:", np.count_nonzero(flag==True), np.count_nonzero(flag==False), flag.shape)
		#print("FLAG CATEGORY:\n", flag_cat)
		print("FLAG ROW:", np.count_nonzero(flag_row==True), np.count_nonzero(flag_row == False), flag_row.shape)


ms_files = ["/work/ska/MeerKAT/1569274256_sdp_l0_wtspec_J0159.0-3413.ms"]#,"/work/ska/MeerKAT/1569371160_sdp_l0_wtspec_J0248.1-0216.ms"]#,"/home/krishna/bluebild/testData/gauss4_t201806301100_SBL180.MS"]
for m in ms_files: examine(m)



