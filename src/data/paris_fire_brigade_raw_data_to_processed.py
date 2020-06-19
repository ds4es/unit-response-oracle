""" 
Script to turn Paris Fire Brigade data challenge raw data :
- x_test.zip
- x_test_additional_file.zip
- x_train.zip
- x_train_additional_file.zip
into preprocessed data (X_TRAIN_PREPROCESSED_FILE, X_TEST_PREPROCESSED_FILE)

All the others script will use X_TRAIN_PREPROCESSED_FILE, X_TEST_PREPROCESSED_FILE 
because an information system should be able to directly provide a query to an API 
with data as it is under X_TRAIN_PREPROCESSED_FILE, X_TEST_PREPROCESSED_FILE

Execution
---------
	# Load your Python environment
	# Add to the PYTHONPATH variable the project root directory
	export PYTHONPATH=$PYTHONPATH:$(pwd)
	 
	# Call the __main__ function to build the train and test preprocessed datasets
	python src/data/paris_fire_brigade_raw_data_to_processed.py

"""

import os
import sys
import pandas as pd
import swifter
import json
import polyline
import logging
from time import time, strftime

from src.config import *
from src.utils import utils

X_TRAIN_FILE = RAW_DATA + "paris_fire_brigade_2018/x_train.zip"
X_TRAIN_ADDITIONAL_FILE = RAW_DATA + "paris_fire_brigade_2018/x_train_additional_file.zip"
X_TEST_FILE = RAW_DATA + "paris_fire_brigade_2018/x_test.zip"
X_TEST_ADDITIONAL_FILE = RAW_DATA + "paris_fire_brigade_2018/x_test_additional_file.zip"

def data_extraction(df) -> pd.DataFrame:

	# convert to timestamp
	df["selection time"] = df["selection time"].astype(int) // 10 ** 9
	# value between 0 to 1 over a 24 hours period
	df["time day"] = (df["selection time"] % SECONDS_IN_A_DAY) / SECONDS_IN_A_DAY
	# value between 0 to 1 over a week period
	df["time week"] = (df["selection time"] % SECONDS_IN_A_WEEK) / SECONDS_IN_A_WEEK
	# value between 0 to 1 over a year period
	df["time year"] = (df["selection time"] % SECONDS_IN_A_YEAR) / SECONDS_IN_A_YEAR

	# replace "delta position gps previous departure-departure" NaN values by 0
	df["delta position gps previous departure-departure"].fillna(0, inplace=True)

	##############
	# Set an Id for the different parking sites under "departure center" 
	#
	rentré = df["status preceding selection"] == "Rentré"
	# Set NaN values for "delta position gps previous departure-departure" to 0
	df.loc[rentré, "delta position gps previous departure-departure"] = 0
	coords_before_departure = [
		"longitude before departure",
		"latitude before departure",
	]
	departure_centers_set = set(
		df[coords_before_departure][rentré].itertuples(index=False)
	)
	
	departure_centers = dict()
	for i, d in enumerate(departure_centers_set, 1):
		departure_centers[d] = i

	df["departure center"] = (
		df[coords_before_departure]
		.swifter.progress_bar(False)
		.apply(
			lambda row: departure_centers[tuple(row)]
			if tuple(row) in departure_centers
			else 0,
			axis=1,
		)
	)
	#
	##############

	df["GPS tracks count"] = df["GPS tracks departure-presentation"].apply(
		lambda x: 0 if pd.isnull(x) else len(x.split(";"))
	)

	# Group under a unique column estimated remaining distances from the last known position
	df["routing engine estimated distance from last observed GPS position"].fillna(
		df["routing engine estimated distance"], inplace=True
	)

	# Group under a unique column estimated remaining time from the last known position
	df["routing engine estimated duration from last observed GPS position"].fillna(
		df["routing engine estimated duration"], inplace=True
	)

	missing_gps = df["GPS tracks count"] == 0

	df.loc[
		missing_gps, "time elapsed between selection and last observed GPS position"
	] = 0

	df.loc[missing_gps, "routing engine estimate from last observed GPS position"] = df.loc[
		missing_gps, "routing engine response"
	]

	df["estimated speed"] = (
		df["routing engine estimated distance"]
		- df["routing engine estimated distance from last observed GPS position"]
	) / df["time elapsed between selection and last observed GPS position"]

	df["estimated duration from speed"] = df[
		"time elapsed between selection and last observed GPS position"
	] + (
		df["routing engine estimated distance from last observed GPS position"]
		/ df["estimated speed"]
	)

	df["estimated time factor"] = (
		df["routing engine estimated duration"]
		- df["routing engine estimated duration from last observed GPS position"]
	) / df["time elapsed between selection and last observed GPS position"]

	df["estimated duration from time"] = df[
		"time elapsed between selection and last observed GPS position"
	] + (
		df["routing engine estimated duration from last observed GPS position"]
		/ df["estimated time factor"]
	)

	df["routing engine response"] = (
		df["routing engine response"].swifter.progress_bar(False).apply(json.loads)
	)

	df["routing engine estimate from last observed GPS position"] = (
		df["routing engine estimate from last observed GPS position"]
		.swifter.progress_bar(False)
		.apply(json.loads)
	)

	df["intervention count"] = (
		df["intervention"].value_counts().loc[df["intervention"]].values
	)

	utils.rescale_minmax(
		df,
		{"longitude before departure": "rescaled longitude before departure", "longitude intervention": "rescaled longitude intervention"},
		LON_MIN,
		LON_MAX,
	)
	utils.rescale_minmax(
		df,
		{"latitude before departure": "rescaled latitude before departure", "latitude intervention": "rescaled latitude intervention"},
		LAT_MIN,
		LAT_MAX,
	)

	df.drop(DROP_BEFORE_DUMP, axis=1, inplace=True, errors='ignore')

	for f in CAT_FEATURES:
		df[f] = utils.to_int_keys_pd(df[f])
		logging.info(f"categorical feature {f}: {len(set(df[f]))} values")

	for f in NUM_FEATURES:
		df[f] = df[f].astype("float")
		logging.info(
			f"numerical feature {f}: {df[f].isnull().mean() * 100:.02f}% NaN values"
		)

	features = set()
	for f in CAT_FEATURES + NUM_FEATURES + IGNORED + OBJECTIVES + [ID]:
		if f in features:
			logging.error(f"duplicated feature {f}")
		features.add(f)
	for f in df.columns:
		if not f in features:
			logging.warning(f"ignored feature {f}")

	return df


if __name__ == '__main__':

	script_start_time = time()

	print("\n*** START MAIN FUNCTION FROM", os.path.dirname(os.path.abspath(__file__)) + os.sep + os.path.basename(__file__), "***\n")

	print("*******************************************************************")
	print("*                                                                 *")
	print("*  This script preprocesses raw data from the Paris Fire Brigade  *")  
	print("*  into preprocessed data saved under:							 *")
	print("*  - X_TRAIN_PREPROCESSED_FILE							 		 *")
	print("*  - X_TEST_PREPROCESSED_FILE							 		 *")
	print("*                                                                 *")
	print("*  All the others scripts will use X_TRAIN_PREPROCESSED_FILE and  *")
	print("*  X_TEST_PREPROCESSED_FILE as input because an information 		 *")
	print("*  system should be able to directly provide a query to an API	 *")
	print("*  with data as provided by these preprocessed data files.  		 *")
	print("*                                                                 *")
	print("*******************************************************************\n")

	if (len(sys.argv) == 7):
		x_train_file = sys.argv[1]
		x_train_additional_file = sys.argv[2]
		x_train_preprocessed_file = sys.argv[3]
		x_test_file = sys.argv[4]
		x_test_additional_file = sys.argv[5]
		x_test_preprocessed_file = sys.argv[6]

	else:
		# Default input
		x_train_file = X_TRAIN_FILE
		x_train_additional_file = X_TRAIN_ADDITIONAL_FILE
		x_train_preprocessed_file = X_TRAIN_PREPROCESSED_FILE
		x_test_file = X_TEST_FILE
		x_test_additional_file = X_TEST_ADDITIONAL_FILE
		x_test_preprocessed_file = X_TEST_PREPROCESSED_FILE

	print("*** EQUIVALENT LAUNCHED COMMAND ***")
	print(sys.executable, sys.argv[0],x_train_file,x_train_additional_file,x_train_preprocessed_file,x_test_file,x_test_additional_file,x_test_preprocessed_file,"\n")

	print("*** INPUT FILES ***")
	print("- Train data file:", x_train_file)
	print("- Additional train data file to merge:", x_train_additional_file)
	print("- Output preprocessed train data CSV file:", x_train_preprocessed_file)
	print("- Test data file:", x_test_file)
	print("- Additional test data file to merge:", x_test_additional_file)
	print("- Output preprocessed test data CSV file:", x_test_preprocessed_file)
	print("")

	if (os.path.isfile(x_train_file) \
		and os.path.isfile(x_train_additional_file) \
		and os.path.isfile(x_train_preprocessed_file) \
		and os.path.isfile(x_test_file) \
		and os.path.isfile(x_test_additional_file) \
		and os.path.isfile(x_test_preprocessed_file)) == False:
		print("USAGE: python src/data/paris_fire_brigade_raw_data_to_processed.py [X_TRAIN_FILE] [X_TRAIN_ADDITIONAL_FILE] [X_TRAIN_PREPROCESSED_FILE] [X_TEST_FILE] [X_TEST_ADDITIONAL_FILE] [X_TEST_PREPROCESSED_FILE]")
		sys.exit()

	print("*** START DATA PROCESSING ***")

	print(strftime('%H:%M:%S'), "- Start loading raw data")
	start_time = time()
	X = pd.read_csv(x_train_file, index_col=ID, parse_dates=["selection time"])
	additional_data = pd.read_csv(x_train_additional_file, index_col=ID)
	X = X.merge(additional_data, how="left", on=ID)
	X_train_index = X.index 
	X_test = pd.read_csv(x_test_file, index_col=ID, parse_dates=["selection time"])
	additional_data_test = pd.read_csv(x_test_additional_file, index_col=ID)
	X_test = X_test.merge(additional_data_test, how="left", on=ID)
	X_test_index = X_test.index
	X = pd.concat((X, X_test), sort=False)
	print("Raw data loaded in", utils.time_me(time() - start_time), "\n")

	print(strftime('%H:%M:%S'), "- Start data extraction")
	start_time = time()
	X = data_extraction(X)
	print("Extraction completed in", utils.time_me(time() - start_time), "\n")

	print(len(list(X)),"columns to export:")
	print([X.index.name] + list(X))
	print("")

	# Export training input dataset
	print(strftime('%H:%M:%S'), "- Start data export to", x_train_preprocessed_file)
	start_time = time()
	compression_opts = dict(method='zip', archive_name='x_train_preprocessed.csv')  
	X.loc[X_train_index,:].to_csv(X_TRAIN_PREPROCESSED_FILE, compression=compression_opts, sep=',', encoding='utf-8')
	print("Data export completed in", utils.time_me(time() - start_time), "\n")

	# Export testing input dataset
	print(strftime('%H:%M:%S'), "- Start data export to", x_test_preprocessed_file)
	start_time = time()
	compression_opts = dict(method='zip', archive_name='x_test_preprocessed.csv')  
	X.loc[X_test_index,:].to_csv(X_TEST_PREPROCESSED_FILE, compression="zip", sep=',', encoding='utf-8')
	print("Data export completed in", utils.time_me(time() - start_time), "\n")

	print("*** SCRIPT COMPLETED IN", utils.time_me(time() - script_start_time), "***")
