"""Make a query on a random data selection from X_TEST_PREPROCESSED_FILE,
submit it to http://127.0.0.1:5000/api and print the response message

Execution
---------
	# Load your Python environment
	# Add to the PYTHONPATH variable the project root directory
	export PYTHONPATH=$PYTHONPATH:$(pwd)
	 
	# Launch the script
	python api/make_me_a_query_for_paris_fire_brigade_2018_data.py

"""

# Authors: Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com

import numpy as np
import random
import pandas as pd
from pandas.io.json import json_normalize
import json
import os
from time import time, strftime

from src.config import *
from src.utils import utils


if __name__ == '__main__':

	x_test_file = X_TEST_PREPROCESSED_FILE
	if WITH_CROSS_VALIDATION == True:
		lgbm_model_file = LGBM_MODEL_WITH_CV_FILE
		y_predicted_file = Y_PREDICTED_WITH_CV_FILE
	else:
		lgbm_model_file = LGBM_MODEL_WITHOUT_CV_FILE
		y_predicted_file = Y_PREDICTED_WITHOUT_CV_FILE

	X_test = pd.read_csv(x_test_file)
	
	inter_count = X_test.groupby(['intervention']).size().to_frame()

	print("\nIntervention selection on a random number of units mobilized for it")
	number_of_units = random.randint(1,max(inter_count[0]))
	intervention = random.choice(inter_count.loc[inter_count[0] == number_of_units].index.to_list())
	print('Intervention:',intervention)
	print('Number of units:',number_of_units)

	"""
	print("Selection an intervention on a random number of intervention")
	intervention = random.choice(inter_count.index)
	print('Intervention:',intervention)
	number_of_units = inter_count.loc[intervention][0]
	print('Number of units:',number_of_units)
	"""

	intervention_infos = X_test.loc[X_test['intervention'] == intervention][['intervention',
	 'alert reason category',
	 'alert reason',
	 'intervention on public roads',
	 'floor',
	 'location of the event',
	 'longitude intervention',
	 'latitude intervention',
	 'rescaled longitude intervention',
	 'rescaled latitude intervention',
	 'time day',
	 'time week',
	 'time year']].head(1)
	intervention_infos_json = intervention_infos.apply(lambda x: x.to_json(), axis=1)
	str(intervention_infos_json.to_list()[0])

	units = X_test.loc[X_test['intervention'] == intervention][['emergency vehicle selection',
	 'emergency vehicle',
	 'emergency vehicle type',
	 'rescue center',
	 'selection time',
	 'delta status preceding selection-selection',
	 'departed from its rescue center',
	 'longitude before departure',
	 'latitude before departure',
	 'delta position gps previous departure-departure',
	 'routing engine estimated distance',
	 'routing engine estimated duration',
	 'routing engine estimated distance from last observed GPS position',
	 'routing engine estimated duration from last observed GPS position',
	 'time elapsed between selection and last observed GPS position',
	 'updated routing engine estimated duration',
	 'departure center',
	 'GPS tracks count',
	 'estimated speed',
	 'estimated duration from speed',
	 'estimated time factor',
	 'estimated duration from time',
	 'intervention count',
	 'rescaled longitude before departure',
	 'rescaled latitude before departure']]

	units_json = units.apply(lambda x: x.to_json(), axis=1)
	
	command = 'curl -H "Content-type: application/json" -X POST http://127.0.0.1:5000/api -d \'{"intervention": '+str(intervention_infos_json.to_list()[0])+', "units":['+','.join(units_json)+']}\'' 


	print("\n*** INTERVENTION DETAILS FOR PREDICTION ***")
	for index, row in intervention_infos.iterrows():
		print('intervention:', int(row['intervention']))
		print('alert reason category:', int(row['alert reason category']))
		print('alert reason:', int(row['alert reason']))
		print('intervention on public roads:', int(row['intervention on public roads']))
		print('floor:', int(row['floor']))
		print('location of the event:', int(row['location of the event']))
		print('longitude intervention:', row['longitude intervention'])
		print('latitude intervention:', row['latitude intervention'])
		print('rescaled longitude intervention:', row['rescaled longitude intervention'])
		print('rescaled latitude intervention:', row['rescaled latitude intervention'])
		print('time day:', row['time day'])
		print('time week:', row['time week'])
		print('time year:', row['time year'],"\n")

	msg = 'See the ' + str(len(units)) + ' units details?'
	shall = input("%s (y/N) " % msg).lower() == 'y'

	if shall:
		print("\n*** UNITS DETAILS FOR PREDICTION ***")
		i = 1
		for index, row in units.iterrows():
			print('UNIT', i)
			print('emergency vehicle selection:', int(row['emergency vehicle selection']))
			print('emergency vehicle:', int(row['emergency vehicle']))
			print('emergency vehicle type:', int(row['emergency vehicle type']))
			print('rescue center:', int(row['rescue center']))
			print('selection time:', int(row['selection time']))
			print('delta status preceding selection-selection:', int(row['delta status preceding selection-selection']))
			print('departed from its rescue center:', row['departed from its rescue center'])
			print('longitude before departure:', row['longitude before departure'])
			print('latitude before departure:', row['latitude before departure'])
			print('delta position gps previous departure-departure:', int(row['delta position gps previous departure-departure']))
			print('routing engine estimated distance:', row['routing engine estimated distance'])
			print('routing engine estimated duration:', row['routing engine estimated duration'])
			print('routing engine estimated distance from last observed GPS position:', row['routing engine estimated distance from last observed GPS position'])
			print('routing engine estimated duration from last observed GPS position:', row['routing engine estimated duration from last observed GPS position'])
			print('time elapsed between selection and last observed GPS position:', row['time elapsed between selection and last observed GPS position'])
			print('updated routing engine estimated duration:', row['updated routing engine estimated duration'])
			print('departure center:', int(row['departure center']))
			print('GPS tracks count:', int(row['GPS tracks count']))
			print('estimated speed:', row['estimated speed'])
			print('estimated duration from speed:', row['estimated duration from speed'])
			print('estimated time factor:', row['estimated time factor'])
			print('estimated duration from time:', row['estimated duration from time'])
			print('intervention count:', int(row['intervention count']))
			print('rescaled longitude before departure:', row['rescaled longitude before departure'])
			print('rescaled latitude before departure:', row['rescaled latitude before departure'], '\n')
			i += 1
		#for index, row in units.iterrows():
			# print(row['intervention'], row['alert reason category'], row['alert reason'], row['intervention on public roads'], row['floor'], row['location of the event'], row['longitude intervention'], row['latitude intervention'], row['rescaled longitude intervention'], row['rescaled latitude intervention'], row['time day'], row['time week'], row['time year'],"\n")

	msg = 'See JSON request for prediction?'
	shall = input("%s (y/N) " % msg).lower() == 'y'
	if shall:
		print("\n*** JSON REQUEST FOR PREDICTION ***")
		print(command,"\n")

	msg = 'Before sending the request should we display the response under a table format?'
	shall = input("%s (y/N) " % msg).lower() == 'y'

	print("\n",strftime('%H:%M:%S'),"- Sending the request...")
	print("API response:")
	start_time = time()
	result = os.popen(command).read()
	results = result.split("\n")

	output = pd.DataFrame()
	for row in results[:-1]:
	    output = output.append(json.loads(row), ignore_index=True)

	for col in list(output):
	    output[col] = output[col].astype(int)
	    
	output.set_index('emergency vehicle selection', inplace=True)
	output = output[['delta selection-departure', 'delta departure-presentation', 'delta selection-presentation']]
	print(output.to_string())
	print("Response received in", utils.time_me(time() - start_time), "\n")
