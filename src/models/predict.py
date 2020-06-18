"""Function set to make a prediction with given trained models (linear & lgbm)

Standalone execution
--------------------
	# Load your Python environment
	# Add to the PYTHONPATH variable the project root directory
	export PYTHONPATH=$PYTHONPATH:$(pwd)
	 
	# Call the __main__ function to make a prediction
	python src/models/predict.py

"""

# Authors: Wenqi Shu-Quartier-dit-Maire
#          Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com
# License: MIT License

import os
import sys
import pandas as pd
import numpy as np
import time
import pickle
import optuna
from time import time, strftime

from src.config import *
from src.utils import utils
from src.features import features


def load_study_and_return_best_params(optuna_study_name,optuna_storage):
	"""Load (or create if do not exist) an Optuna study (https://optuna.readthedocs.io)

	Parameters
	----------
	optuna_study_name :
		Study’s name. Each study has a unique name as an identifier.

	optuna_storage :
		Database URL such as sqlite:///example.db. Please see also the documentation of create_study() for further details.

	Returns
	-------
	params : object
		Such as
		{
			"data": {
				"center_decay": 0.14281578186170577,
				"use_cyclical": True,
				"vehicle_decay": 0.17590059703294494,
			},
			"model": {
				"boosting_type": "gbdt",
				"colsample_bytree": 0.5279207022532362,
				"learning_rate": 0.012081577123096265,
				"min_child_samples": 45,
				"min_child_weight": 0.007084184412851127,
				"n_estimators": 568,
				"num_leaves": 483,
				"reg_alpha": 0.10389662610302736,
				"reg_lambda": 0.026121337399318097,
				"subsample": 0.9076986626277991,
				"subsample_freq": 0,
			},
		} 
	"""
	if __debug__: print("In load_study_and_return_best_params()")
	start_time = time()

	# Create a study if do not exist
	study = optuna.load_study(
					study_name=optuna_study_name, 
					storage=optuna_storage)
	print("Optuna study loaded in", utils.time_me(time() - start_time), "\n")

	# Retrieve best parameters
	trial = study.best_trial
	params = utils.sample_params(optuna.trial.FixedTrial(trial.params))
	
	return params


def load_linear_model(linear_model_file):
	"""Load a linear model from the given file

	Parameters
	----------
	linear_model_file : string
		Path to a pickle file

	Returns
	-------
	linear_model : sklearn.linear_model.LinearRegression
		Trained linear model
	"""
	if __debug__: print("In load_linear_model()")
	print(strftime('%H:%M:%S'), "- Load the linear model")
	start_time = time()
	linear_model = pickle.load(open(linear_model_file, 'rb'))

	print("Linar model loaded in", utils.time_me(time() - start_time), "\n")

	return linear_model


def load_lgbm_model(lgbm_model_file):
	"""Load a multiouput gradient boosting model from the given file

	Parameters
	----------
	lgbm_model_file : string
		Path to a multioutput gradient boosting model pickle file

	Returns
	-------
	lgbm_model : object
		Multioutput gradient boosting model.
	"""
	if __debug__: print("In load_lgbm_model()")
	print(strftime('%H:%M:%S'), "- Load the lgbm model")
	start_time = time()
	lgbm_model = pickle.load(open(lgbm_model_file, 'rb'))

	print("Lgbm model loaded in", utils.time_me(time() - start_time), "\n")

	return lgbm_model

def make_prediction(X_test,linear_model,lgbm_model):
	"""Make a prediction for the given dataset based on the given linear and lgbm models

	Parameters
	----------
	X_test : Pandas dataframe
		Input testing data.

	linear_model : object
		Linear model
	
	lgbm_model : object
		Multiouput gradient boosting model

	Returns
	-------
	lgbm_model : object
		Multioutput gradient boosting model.

	"""
	if __debug__: print("In make_prediction()")

	print(strftime('%H:%M:%S'),"- Start the computation for a first features set")
	start_time = time()
	X_test = features.compute_feature_set_one(X_test, linear_model)
	print("First features set computed in", utils.time_me(time() - start_time), "\n")

	params = {}
	if PARAMS_FROM_BEST_OPTUNA_STUDY_IN_DB == True:
		params = load_study_and_return_best_params(OPTUNA_STUDY_NAME,OPTUNA_STORAGE)
		
	else:
		params = PARAMS

	print(strftime('%H:%M:%S'), "- Start the computation of another feature set")
	start_time = time()
	X_test = features.compute_feature_set_two(X_test, ID, **params["data"])
	print("Features computed in", utils.time_me(time() - start_time), "\n")

	print("Drop useless parameters")
	X_test.drop(IGNORED, axis=1, inplace=True, errors='ignore')
	print(len(list(X_test)),"parameters left: ",list(X_test))

	print("Load the model\n")

	if WITH_CROSS_VALIDATION == True:
				
		print(strftime('%H:%M:%S'), "- Start computing the predictions with cross validation")
		start_time = time()
		y_cv = [loaded_model.predict(X_test) for loaded_model in lgbm_model["estimator"]]
		print("Predictions computed in", utils.time_me(time() - start_time),"for", np.shape(y_cv)[0], "entries\n")

		print("Cross validation test scores:")
		for item in lgbm_model["test_score"]:
			print(item)
		print("")
		
		y_cv = np.mean(y_cv, axis=0)

		df = pd.DataFrame(
			data=y_cv,
			index=X_test.index,
			columns=["delta selection-departure","delta departure-presentation","delta selection-presentation"]
		)
		return df

	else:
	
		print(strftime('%H:%M:%S'), "- Start computing the predictions")
		start_time = time()
		y = lgbm_model.predict(X_test)
		print("Predictions with no cross validation computed in", utils.time_me(time() - start_time),"for", np.shape(y)[0], "entries\n")

		df = pd.DataFrame(data=y,
			index=X_test.index,
			columns=["delta selection-departure","delta departure-presentation","delta selection-presentation"])
		return df


def make_prediction_and_save(X_test,linear_model_file,lgbm_model_file,y_predicted_file):
	"""Make a prediction for the given dataset based on the given linear and lgbm models and save the result to a CSV file

	Parameters
	----------
	X_test : Pandas dataframe
		Input testing data.

	linear_model_file : string
		Path to a linear model pickle file
	
	lgbm_model_file : string
		Path to a multioutput gradient boosting model pickle file

	y_predicted_file : string
		CSV data file where to save the predictions 

	"""
	if __debug__: print("In make_prediction_and_save()")

	linear_model = load_linear_model(linear_model_file)
	lgbm_model = load_lgbm_model(lgbm_model_file)
	df = make_prediction(X_test,linear_model,lgbm_model)
	print(strftime('%H:%M:%S'), "- Saving the prediction under", y_predicted_file)
	start_time = time()
	df.to_csv(y_predicted_file, sep=',', encoding='utf-8')
	print("Prediction saved in", utils.time_me(time() - start_time), "\n")


# For a shell standalone execution
if __name__ == '__main__':
	
	script_start_time = time()

	print("\n*** START MAIN FUNCTION FROM", os.path.dirname(os.path.abspath(__file__)) + os.sep + os.path.basename(__file__), "***\n")

	print("*******************************************************************")
	print("*                                                                 *")
	print("*  This script predict units response times for the given         *")  
	print("*  input file, trained linear and lgbm models.                    *")
	print("*                                                                 *")
	print("*******************************************************************\n")

	if (len(sys.argv) == 5):
		x_test_file = sys.argv[1]
		linear_model_file = sys.argv[2]
		lgbm_model_file = sys.argv[3]
		y_predicted_file = sys.argv[4]
	else:
		x_test_file = X_TEST_PREPROCESSED_FILE
		linear_model_file = LINEAR_MODEL_FILE
		if WITH_CROSS_VALIDATION == True:
			lgbm_model_file = LGBM_MODEL_WITH_CV_FILE
			y_predicted_file = Y_PREDICTED_WITH_CV_FILE
		else:
			lgbm_model_file = LGBM_MODEL_WITHOUT_CV_FILE
			y_predicted_file = Y_PREDICTED_WITHOUT_CV_FILE

	if (os.path.isfile(x_test_file) and os.path.isfile(linear_model_file) and os.path.isfile(lgbm_model_file) and os.path.isfile(y_predicted_file)) == False:
		print("USAGE: python src/models/predict.py [X_TEST_PREPROCESSED_FILE] [LINEAR_MODEL_FILE] [LGBM_MODEL_FILE] [Y_PREDICTED_FILE]")
		sys.exit()

	print("*** EQUIVALENT LAUNCHED COMMAND ***")
	print(sys.executable, sys.argv[0],x_test_file,lgbm_model_file,y_predicted_file,"\n")

	print("*** INPUT FILES ***")
	print("- Input testing data file:", x_test_file)
	print("- Linear model used:", linear_model_file)
	print("- Lgbm model used:", lgbm_model_file)
	print("- Output prediction data file:", y_predicted_file)
	print("")
	
	if WITH_CROSS_VALIDATION == True:
		print("The training will be done with cross validation\n")
	else:
		print("The training will be done without cross validation\n")

	print("*** START DATA PROCESSING ***")
	print(strftime('%H:%M:%S'),"- Start loading testing data at")
	start_time = time()
	X_test = pd.read_csv(x_test_file, index_col=ID)
	print("Testing data loaded in", utils.time_me(time() - start_time), "\n")

	print(len(list(X_test)),"input parameters: ",list(X_test), "\n")

	if DEV == False:
		print("The full dataset will be used for testing purpose.")
	else:
		print("Only", NUMBER_OF_ROWS_USED_FOR_DEV, "rows will be used for testing purpose.")
		X_test = X_test.head(NUMBER_OF_ROWS_USED_FOR_DEV)
	print("")

	make_prediction_and_save(X_test,linear_model_file,lgbm_model_file,y_predicted_file)

	print("*** SCRIPT COMPLETED IN", utils.time_me(time() - script_start_time), "***")



