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
	 'time year']].head(1).apply(lambda x: x.to_json(), axis=1)
	str(intervention_infos.to_list()[0])

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
	 'rescaled latitude before departure']].apply(lambda x: x.to_json(), axis=1)
	
	command = 'curl -H "Content-type: application/json" -X POST http://127.0.0.1:5000/api -d \'{"intervention": '+str(intervention_infos.to_list()[0])+', "units":['+','.join(units)+']}\'' 
	
	print("\nRequest for prediction:")
	print(command,"\n")

	print(strftime('%H:%M:%S'),"- Send the request")
	print("API response:")
	start_time = time()
	os.system(command)
	print("Response received in", utils.time_me(time() - start_time), "\n")
