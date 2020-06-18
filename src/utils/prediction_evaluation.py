"""For prediction evaluation

Standalone execution
--------------------
	# Load your Python environment
	# Add to the PYTHONPATH variable the project root directory
	export PYTHONPATH=$PYTHONPATH:$(pwd)
	 
	# Call the __main__ function to evaluate a prediction
	python src/utils/prediction_evaluation.py

"""

# Authors: Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com


import sys
from time import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import mlflow

from src.utils import utils
from src.config import *


def prediction_evaluation(observed, predicted):
	"""Make a prediction evaluation for the given observed and predicted datasets

	Parameters
	----------
	observed : Pandas dataframe
		Output observed data.

	predicted : Pandas dataframe
		Output predicted data.

	Returns
	-------
	selection_departure_r2_score : float
		Selection departure R² score 

	departure_presentation_r2_score : float
		Departure presentation R² score 

	selection_presentation_r2_score : float
		Selection presentation R² score 

	mean_r2_scores : float
		Mean for the 3 previous R² score 

	rmsle[0] : float
		Root Mean Squared Logarithmic Error

	root_mean_squared_error : float
		Root Mean Squared Error

	median_error : float
		Median error

	mean_error : float
		Mean error

	"""
	if __debug__: print("In prediction_evaluation()")

	selection_departure_r2_score = r2_score(observed["delta selection-departure"],predicted["delta selection-departure"])
	print("delta selection-departure R² score:", selection_departure_r2_score)
	departure_presentation_r2_score = r2_score(observed["delta departure-presentation"],predicted["delta departure-presentation"])
	print("delta departure-presentation R² score:", departure_presentation_r2_score)
	selection_presentation_r2_score = r2_score(observed["delta selection-presentation"],predicted["delta selection-presentation"])
	print("delta selection-presentation R² score:", selection_presentation_r2_score)

	mean_r2_scores = (selection_departure_r2_score + departure_presentation_r2_score + selection_presentation_r2_score)/3
	print("Mean on those 3 R² scores:", mean_r2_scores, "\n")

	rmsle = utils.rmsle(observed[["delta selection-presentation"]].values,predicted[["delta selection-presentation"]].values)
	print("RMSLE (Root mean squared logarithmic error):", rmsle[0])

	root_mean_squared_error = mean_squared_error(observed[["delta selection-presentation"]].values,predicted[["delta selection-presentation"]].values, squared=False)
	print("RMSE (Root mean squared error):", root_mean_squared_error, "seconds\n")

	median_error = abs(observed["delta selection-presentation"]-predicted["delta selection-presentation"]).median()
	print("Typical error:", median_error, "seconds")

	mean_error = abs(observed["delta selection-presentation"]-predicted["delta selection-presentation"]).mean()
	print("Mean error:", mean_error, "seconds")

	return (selection_departure_r2_score, departure_presentation_r2_score, selection_presentation_r2_score, mean_r2_scores, rmsle[0],root_mean_squared_error,median_error,mean_error)


# For a shell standalone execution
if __name__ == '__main__':

	script_start_time = time()

	print("\n*** START MAIN FUNCTION FROM", os.path.dirname(os.path.abspath(__file__)) + os.sep + os.path.basename(__file__), "***\n")

	print("*******************************************************************")
	print("*                                                                 *")
	print("*  This script evaluates the accuracy of a prediction compared    *")  
	print("*  to the given observed outputs.                                 *")
	print("*                                                                 *")
	print("*******************************************************************\n")

	if (len(sys.argv) == 3):
		y_test_file = sys.argv[1]
		y_predicted_file = sys.argv[2]
	else:
		y_test_file = Y_TEST_FILE
		y_predicted_file = Y_PREDICTED_WITH_CV_FILE

	if (os.path.isfile(y_test_file) and os.path.isfile(y_predicted_file)) == False:
		print("USAGE: python src/utils/prediction_evaluation.py [Y_TEST_FILE] [Y_PREDICTED_WITH_CV_FILE]")
		sys.exit()

	print("\n*** EQUIVALENT LAUNCHED COMMAND ***")
	print(sys.executable, sys.argv[0],y_test_file,y_predicted_file,"\n")


	print("*** INPUT FILES ***")
	print("- Output data file:", y_test_file)
	print("- Predicted output data file:", y_predicted_file)
	print("")

	y_test = pd.read_csv(y_test_file, index_col=ID)
	y_test_predicted = pd.read_csv(y_predicted_file, index_col=ID)

	selection_departure_r2_score, \
	departure_presentation_r2_score, \
	selection_presentation_r2_score, \
	mean_r2_scores, \
	rmsle, \
	root_mean_squared_error, \
	median_error, \
	mean_error = prediction_evaluation(y_test.loc[y_test_predicted.index,:], y_test_predicted)

	# ML Flow log param to record
	mlflow.log_param("DEV", DEV)
	mlflow.log_param("Amount of data used for testing", y_test.shape[0])
	mlflow.log_param("RUN_OPTUNA_STUDY", RUN_OPTUNA_STUDY)
	mlflow.log_param("PARAMS_FROM_BEST_OPTUNA_STUDY_IN_DB", PARAMS_FROM_BEST_OPTUNA_STUDY_IN_DB)
	mlflow.log_param("WITH_CROSS_VALIDATION", WITH_CROSS_VALIDATION)
	mlflow.log_param("LINEAR_MODEL_FILE", LINEAR_MODEL_FILE)
	if WITH_CROSS_VALIDATION == True:
		mlflow.log_param("LGBM_MODEL_FILE", LGBM_MODEL_WITH_CV_FILE)
	else:
		mlflow.log_param("LGBM_MODEL_FILE", LGBM_MODEL_WITHOUT_CV_FILE)

	mlflow.log_param("X_TRAIN_PREPROCESSED_FILE", X_TRAIN_PREPROCESSED_FILE)
	mlflow.log_param("X_TEST_PREPROCESSED_FILE", X_TEST_PREPROCESSED_FILE)
	mlflow.log_param("Y_TRAIN_FILE", Y_TRAIN_FILE)
	mlflow.log_param("Y_TEST_FILE", Y_TEST_FILE)
	mlflow.log_param("Y_PREDICTED_WITHOUT_CV_FILE", Y_PREDICTED_WITHOUT_CV_FILE)
	mlflow.log_param("Y_PREDICTED_WITH_CV_FILE", Y_PREDICTED_WITH_CV_FILE)

	# ML Flow log metric to record
	mlflow.log_metric("Selection-departure R2 score", selection_departure_r2_score)
	mlflow.log_metric("Departure-presentation R2 score", departure_presentation_r2_score)
	mlflow.log_metric("Selection-presentation R2 score", selection_presentation_r2_score)
	mlflow.log_metric("Mean R2 scores", mean_r2_scores)
	mlflow.log_metric("Root mean squared logarithmic error", rmsle)
	mlflow.log_metric("Root mean squared error", root_mean_squared_error)
	mlflow.log_metric("Median error in seconds", median_error)
	mlflow.log_metric("Mean error in seconds", mean_error)

	print("*** SCRIPT COMPLETED IN", utils.time_me(time() - script_start_time), "***")
