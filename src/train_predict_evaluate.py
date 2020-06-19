"""Script to train a model, predict and evaluate the result of the prediction

Standalone execution
--------------------
	# Load your Python environment
	# Add to the PYTHONPATH variable the project root directory
	export PYTHONPATH=$PYTHONPATH:$(pwd)
	 
	# Call the __main__ function to launch a standalone gradient boosting model training
	python src/train_predict_evauate.py

"""

# Authors: Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com



import mlflow
import random

from src.models.train_lgbm import *
from src.models.predict import *
from src.utils.prediction_evaluation import *

from src.config import *

if __name__ == '__main__':

	script_start_time = time()

	print("\n*** START MAIN FUNCTION FROM", os.path.dirname(os.path.abspath(__file__)) + os.sep + os.path.basename(__file__), "***\n")

	print("*******************************************************************")
	print("*                                                                 *")
	print("*  From preprocessed data, this script trains a model, makes a    *")
	print("*  prediction and evaluates the result.				             *")
	print("*                                                                 *")
	print("*******************************************************************\n")

	if (len(sys.argv) == 8):
		x_train_file = sys.argv[1]
		y_train_file = sys.argv[2]
		x_test_file = sys.argv[3]
		y_test_file = sys.argv[4]
		linear_model_file = sys.argv[5]
		lgbm_model_file = sys.argv[6]
		y_predicted_file = sys.argv[7]
	else:
		# Default input
		x_train_file = X_TRAIN_PREPROCESSED_FILE
		y_train_file = Y_TRAIN_FILE
		x_test_file = X_TEST_PREPROCESSED_FILE
		y_test_file = Y_TEST_FILE
		linear_model_file = LINEAR_MODEL_FILE
		if WITH_CROSS_VALIDATION == True:
			lgbm_model_file = LGBM_MODEL_WITH_CV_FILE
			y_predicted_file = Y_PREDICTED_WITH_CV_FILE
		else:
			lgbm_model_file = LGBM_MODEL_WITHOUT_CV_FILE
			y_predicted_file = Y_PREDICTED_WITHOUT_CV_FILE

	if (os.path.isfile(x_train_file) \
		and os.path.isfile(y_train_file) \
		and os.path.isfile(x_test_file) \
		and os.path.isfile(y_test_file) \
		and os.path.isfile(linear_model_file) \
		and os.path.isfile(lgbm_model_file) \
		and os.path.isfile(y_predicted_file)) == False:
		print("USAGE: python src/models/train_linear.py [X_TRAIN_PREPROCESSED_FILE] [Y_TRAIN_FILE] [X_TEST_PREPROCESSED_FILE] [Y_TEST_FILE] [LINEAR_MODEL_FILE] [LGBM_MODEL_WITH_CV_FILE] [Y_PREDICTED_WITHOUT_CV_FILE]")
		sys.exit()

	print("*** EQUIVALENT LAUNCHED COMMAND ***")
	print(sys.executable \
		, sys.argv[0] \
		, x_train_file \
		, y_train_file \
		, x_test_file \
		, y_test_file \
		, linear_model_file
		, lgbm_model_file \
		, y_predicted_file \
		, "\n")

	print("*** INPUT FILES ***")
	print("- Input training data file:", x_train_file)
	print("- Output training data file:", y_train_file)
	print("- Input testing data file:", x_test_file)
	print("- Output testing data file:", y_test_file)
	print("- Linear model will be saved into:", linear_model_file)
	print("- Gradient boosting model will be saved into:", lgbm_model_file)
	print("- Predicted output data file:", y_predicted_file)
	print("")

	mlflow.log_param("x_train_file", x_train_file)
	mlflow.log_param("y_train_file", y_train_file)
	mlflow.log_param("x_test_file", x_test_file)
	mlflow.log_param("y_test_file", y_test_file)
	mlflow.log_param("linear_model_file", linear_model_file)
	mlflow.log_param("lgbm_model_file", lgbm_model_file)
	mlflow.log_param("y_predicted_file", y_predicted_file)
	
	print("*** START TRAINING PHASE ***")

	if WITH_CROSS_VALIDATION == True:
		print("The training will be done with cross validation\n")
	else:
		print("The training will be done without cross validation\n")

	print(strftime('%H:%M:%S'), "- Start loading training data")
	start_time = time()
	X_train = pd.read_csv(x_train_file, index_col=ID)

	if USE_PARAMETERS_SUBSET_FOR_ROBUSTNESS_TEST == True:
		random_number_of_subset_columns = random.randint(1,len(X_train.columns))
		print("For robustness testing, from", len(X_train.columns), "input columns we will use a random subset of",random_number_of_subset_columns,"columns:")
		random_subset_of_columns = random.sample(list(X_train.columns), random_number_of_subset_columns)
		print(random_subset_of_columns)
		X_train = X_train[random_subset_of_columns]
	else:
		print(len(X_train.columns),"input columns:")
		print(list(X_train.columns))

	y_train = pd.read_csv(y_train_file, index_col=ID)
	print("Training data loaded in", utils.time_me(time() - start_time), "\n")

	if DEV == False:
		print("The full dataset will be used for training purpose.")
	else:
		print("Only", NUMBER_OF_ROWS_USED_FOR_DEV, "rows will be used for training purpose.")
		X_train = X_train.head(NUMBER_OF_ROWS_USED_FOR_DEV)
		y_train = y_train.loc[X_train.index,:]

	model_parameters = train_and_save_model(X_train, y_train, linear_model_file, lgbm_model_file, RUN_OPTUNA_STUDY)

	print("*** START PREDICTION PHASE ***")

	print(strftime('%H:%M:%S'),"- Start loading testing data at")
	start_time = time()
	X_test = pd.read_csv(x_test_file, index_col=ID)

	if USE_PARAMETERS_SUBSET_FOR_ROBUSTNESS_TEST == True:
		X_test = X_test[random_subset_of_columns]
	print("Training data loaded in", utils.time_me(time() - start_time), "\n")

	if DEV == False:
		print("The full dataset will be used for testing purpose.")
	else:
		print("Only", NUMBER_OF_ROWS_USED_FOR_DEV, "rows will be used for testing purpose.")
		X_test = X_test.head(NUMBER_OF_ROWS_USED_FOR_DEV)
	print("")

	make_prediction_and_save(X_test,linear_model_file,lgbm_model_file,y_predicted_file)

	print("*** START EVALUATION PHASE ***")

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

	mlflow.log_param("Amount of data used for testing", X_test.shape[0])
	mlflow.log_param("Amount of data used for training", X_train.shape[0])
	
	mlflow.log_param("DEV", DEV)
	mlflow.log_param("NUMBER_OF_ROWS_USED_FOR_DEV", NUMBER_OF_ROWS_USED_FOR_DEV)
	mlflow.log_param("USE_PARAMETERS_SUBSET_FOR_ROBUSTNESS_TEST", USE_PARAMETERS_SUBSET_FOR_ROBUSTNESS_TEST)

	mlflow.log_param("RUN_OPTUNA_STUDY", RUN_OPTUNA_STUDY)
	mlflow.log_param("OPTUNA_STUDY_NAME",OPTUNA_STUDY_NAME)
	mlflow.log_param("OPTUNA_STORAGE",OPTUNA_STORAGE)
	mlflow.log_param("OPTUNA_OPTIMIZATION_DIRECTION",OPTUNA_OPTIMIZATION_DIRECTION)
	mlflow.log_param("OPTUNA_LOAD_STUDY_IF_EXIST",OPTUNA_LOAD_STUDY_IF_EXIST)
	mlflow.log_param("OPTUNA_OBJECTIVE_AGGREGATION_FUNCTION",OPTUNA_OBJECTIVE_AGGREGATION_FUNCTION)
	mlflow.log_param("OPTUNA_NUMBER_OF_TRIALS",OPTUNA_NUMBER_OF_TRIALS)
	mlflow.log_param("OPTUNA_NUMBER_OF_PARALLEL_JOBS",OPTUNA_NUMBER_OF_PARALLEL_JOBS)
	mlflow.log_param("PARAMS_FROM_BEST_OPTUNA_STUDY_IN_DB", PARAMS_FROM_BEST_OPTUNA_STUDY_IN_DB)

	mlflow.log_param("WITH_CROSS_VALIDATION", WITH_CROSS_VALIDATION)
	mlflow.log_param("Model parameters", model_parameters)
	mlflow.log_param("LINEAR_MODEL_FILE", LINEAR_MODEL_FILE)
	if WITH_CROSS_VALIDATION == True:
		mlflow.log_param("LGBM_MODEL_FILE", LGBM_MODEL_WITH_CV_FILE)
	else:
		mlflow.log_param("LGBM_MODEL_FILE", LGBM_MODEL_WITHOUT_CV_FILE)

	mlflow.log_param("X_TRAIN_PREPROCESSED_FILE", X_TRAIN_PREPROCESSED_FILE)
	mlflow.log_param("X_TEST_PREPROCESSED_FILE", X_TEST_PREPROCESSED_FILE)
	mlflow.log_param("Y_TRAIN_FILE", Y_TRAIN_FILE)
	mlflow.log_param("Y_TEST_FILE", Y_TEST_FILE)
	if WITH_CROSS_VALIDATION == True:
		mlflow.log_param("Y_PREDICTED_FILE", Y_PREDICTED_WITHOUT_CV_FILE)
	else:
		mlflow.log_param("Y_PREDICTED_FILE", Y_PREDICTED_WITH_CV_FILE)

	# Log a metric; metrics can be updated throughout the run
	mlflow.log_metric("Selection-departure R2 score", selection_departure_r2_score)
	mlflow.log_metric("Departure-presentation R2 score", departure_presentation_r2_score)
	mlflow.log_metric("Selection-presentation R2 score", selection_presentation_r2_score)
	mlflow.log_metric("Mean R2 scores", mean_r2_scores)
	mlflow.log_metric("Root mean squared logarithmic error", rmsle)
	mlflow.log_metric("Root mean squared error", root_mean_squared_error)
	mlflow.log_metric("Median error in seconds", median_error)
	mlflow.log_metric("Mean error in seconds", mean_error)

	print("\n*** SCRIPT COMPLETED IN", utils.time_me(time() - script_start_time), "***")
