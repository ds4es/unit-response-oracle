"""Function set for gradient boosting model training and hyperparameters optimization

Standalone execution
--------------------
	# Load your Python environment
	# Add to the PYTHONPATH variable the project root directory
	export PYTHONPATH=$PYTHONPATH:$(pwd)
	 
	# Call the __main__ function to launch a standalone gradient boosting model training
	python src/models/train_lgbm.py

"""

# Authors: Wenqi Shu-Quartier-dit-Maire
#          Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com
# License: MIT License

import pandas as pd
import os
from sklearn.model_selection import cross_validate
import lightgbm as lgbm
import numpy as np
import pickle
import optuna

from src.models.multioutput import MultiOutputRegressorWithNan
from src.models.train_linear import *

from src.config import *
from src.utils import utils
from src.features import features


def sample_params(trial):
	"""Sampling parameter function for Optuna hyperparameters optimmization

	Parameters
	----------
	trial : object
		Trial object for Optuna optimization.

	Returns
	-------
	 : object
		Sampled hyperparameters such as
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
	if __debug__: print("In sample_params()")
	
	return {
		"data": {
			"use_cyclical": trial.suggest_categorical("use_cyclical", [False, True]),
			"center_decay": trial.suggest_uniform("center_decay", 0, 1),
			"vehicle_decay": trial.suggest_uniform("vehicle_decay", 0, 1),
			# "kde_params": sample_kde_params(trial),
		},
		"model": {
			###
			# leave max_depth=-1
			###
			# "boosting_type": trial.suggest_categorical(
			#     "boosting_type", ["gbdt", "dart"]
			# ),
			"boosting_type": "gbdt",
			"num_leaves": trial.suggest_int("num_leaves", 2, 512),
			"min_child_samples": trial.suggest_int("min_data_in_leaf", 2, 200),
			# not available
			# "max_bin": trial.suggest_int("max_bin", 2, 1000),
			"learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1),
			"n_estimators": trial.suggest_int("num_iterations", 10, 1000),
			"min_child_weight": trial.suggest_loguniform(
				"min_sum_hessian_in_leaf", 1e-5, 1e-1
			),
			"subsample": 1 - trial.suggest_uniform("1 - bagging_fraction", 0, 1),
			"subsample_freq": trial.suggest_int("bagging_freq", 0, 7),
			"colsample_bytree": 1 - trial.suggest_uniform("1 - feature_fraction", 0, 1),
			"reg_alpha": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
			"reg_lambda": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
			# not available
			# "boost_from_average": trial.suggest_categorical(
			#     "boost_from_average", [False, True]
			# ),
		},
	}


def get_model(params):
	"""Instanciate a multiouput gradient boosting model based the given parameters

	Parameters
	----------
	params : object
		Sampled hyperparameters

	Returns
	-------
	 : object
		Multiouput gradient boosting model

	"""
	if __debug__: print("In get_model()")

	return MultiOutputRegressorWithNan(lgbm.LGBMRegressor(**params["model"]))


def objective(trial, agg, X_train, y_train):
	"""Objective function for Optuna hyperparameters optimization

	Parameters
	----------
	trial : object
		Trial object for Optuna optimization.

	agg : string
		"median", "mean" or "min" defined in config.py

	X_train : Pandas dataframe
		Input training data.

	y_train : Pandas dataframe
		Output training data.
	
	Returns
	-------
	params : object
		np.mean(scores) or np.median(scores) or np.min(scores)

	"""
	if __debug__: print("In objective()")

	params = sample_params(trial)

	scores = cross_validate(get_model(params), X_train, y_train, cv=5, scoring="r2")["test_score"]

	trial.set_user_attr("cv_scores", scores.tolist())

	agg = {"median": np.median, "mean": np.mean, "min": np.min}[agg]

	return agg(scores)


def run_optuna_study_and_return_best_params(X_train,y_train):
	"""Run an Optuna study and return the best hyperparameters

	Parameters
	----------
	X_train : Pandas dataframe
		Input training data.

	y_train : Pandas dataframe
		Output training data.
	
	Returns
	-------
	params : object
		Best hyperparameters from the Optuna study

	"""
	if __debug__: print("In run_optuna_study_and_return_best_params()")

	# Create a study if do not exist
	study = optuna.create_study(
					study_name=OPTUNA_STUDY_NAME, 
					storage=OPTUNA_STORAGE, 
					direction=OPTUNA_OPTIMIZATION_DIRECTION,
					load_if_exists=OPTUNA_LOAD_STUDY_IF_EXIST)
	print("")

	print(strftime('%H:%M:%S'), "- Start an Optuna hyperparameter optimization")
	start_time = time()
	# Optimize an objective function
	study.optimize(
		lambda trial: objective(trial, OPTUNA_OBJECTIVE_AGGREGATION_FUNCTION, X_train, y_train),
		n_trials=OPTUNA_NUMBER_OF_TRIALS,
		n_jobs=OPTUNA_NUMBER_OF_PARALLEL_JOBS)
	print("Optuna hyperparameter optimization done in", utils.time_me(time() - start_time), "\n")

	trial = study.best_trial

	return sample_params(optuna.trial.FixedTrial(trial.params))


def train_and_save_model(X_train, y_train, linear_model_file, lgbm_model_file, run_optuna_study=False):
	"""Train and save linear and gradient boosting models

	Parameters
	----------
	X_train : Pandas dataframe
		Input training data.

	y_train : Pandas dataframe
		Output training data.

	linear_model_file : string
		Path to a linear model pickle file
	
	lgbm_model_file : string
		Path to a multioutput gradient boosting model pickle file

	run_optuna_study : bool
		False (default) : To use hyperparameters defined in the config.py file
		True : To run an Optuna study to use the best hyperparameters found from it

	Returns
	-------
	params : object
		Used hyperparameters

	"""
	if __debug__: print("In train_and_save_model()")

	# train_and_save_linear_model
	if 'routing engine estimated duration' in X_train.columns:
		print(strftime('%H:%M:%S'), "- Start the linear model training")
		start_time = time()
		linear_model = train_and_save_linear_model(X_train[["routing engine estimated duration"]], y_train["delta selection-presentation"], linear_model_file)
		print("Linear model trained and saved in", utils.time_me(time() - start_time), "\n")
	elif __debug__: print("Do not train linear model")

	print(strftime('%H:%M:%S'), "- Start the computation for a first features set")
	start_time = time()
	if 'linear_model' in locals():
		X_train = features.compute_feature_set_one(X_train, linear_model)
	else:
		X_train = features.compute_feature_set_one(X_train)
	print("First features set computed in", utils.time_me(time() - start_time), "\n")

	params = {}

	if run_optuna_study == True:
		params = run_optuna_study_and_return_best_params(X_train,y_train)

	else:
		params = PARAMS

	print(strftime('%H:%M:%S'), "- Start the computation of another feature set")
	start_time = time()
	X_train = features.compute_feature_set_two(X_train, ID, **params["data"])
	print("Features computed in", utils.time_me(time() - start_time), "\n")

	print("Drop useless parameters\n")
	X_train.drop(IGNORED, axis=1, inplace=True, errors='ignore') 

	model = get_model(params)

	if WITH_CROSS_VALIDATION == True:
		print(strftime('%H:%M:%S'), "- Start the training with cross validation")
		cv = cross_validate(
			get_model(params), X_train, y_train, cv=5, scoring="r2", return_estimator=True
		)
		print("Training with cross validation completed in", utils.time_me(time() - start_time), "\n")

		utils.save_object(cv, lgbm_model_file)
		print("Model with cross validation saved in", lgbm_model_file, "\n")

	else:
		print(strftime('%H:%M:%S'), "- Start the training without cross validation")
		model.fit(X_train, y_train)
		print("Training without cross validation completed in", utils.time_me(time() - start_time), "\n")
		
		pickle.dump(model, open(lgbm_model_file, 'wb'))
		print("Model without cross validation saved in", lgbm_model_file, "\n")

	return params


# For a shell standalone execution
if __name__ == '__main__':
	
	script_start_time = time()

	print("\n*** START MAIN FUNCTION FROM", os.path.dirname(os.path.abspath(__file__)) + os.sep + os.path.basename(__file__), "***\n")

	print("*******************************************************************")
	print("*                                                                 *")
	print("*  This script trains a multioutput gradient boosting model on    *")  
	print("*  preprocessed data for service units response times prediction. *")
	print("*                                                                 *")
	print("*******************************************************************\n")

	if (len(sys.argv) == 4):
		x_train_file = sys.argv[1]
		y_train_file = sys.argv[2]
		linear_model_file = sys.argv[3]
		lgbm_model_file = sys.argv[4]
	else:
		x_train_file = X_TRAIN_PREPROCESSED_FILE
		y_train_file = Y_TRAIN_FILE
		linear_model_file = LINEAR_MODEL_FILE
		if WITH_CROSS_VALIDATION:
			lgbm_model_file = LGBM_MODEL_WITH_CV_FILE
		else:
			lgbm_model_file = LGBM_MODEL_WITHOUT_CV_FILE

	if (os.path.isfile(x_train_file) and os.path.isfile(y_train_file) and os.path.isfile(linear_model_file) and os.path.isfile(lgbm_model_file)) == False:
		print("USAGE: python src/models/train_lgbm.py [X_TRAIN_PREPROCESSED_FILE] [Y_TRAIN_FILE] [LINEAR_MODEL_FILE] [LGBM_MODEL_FILE]")
		sys.exit()

	print("*** EQUIVALENT LAUNCHED COMMAND ***")
	print(sys.executable, sys.argv[0],x_train_file,y_train_file,linear_model_file,lgbm_model_file,"\n")

	print("*** INPUT FILES ***")
	print("- Input training data file:", x_train_file)
	print("- Output training data file:", y_train_file)
	print("- Gradient boosting model will be saved into:", lgbm_model_file)
	print("")
	
	print("*** START DATA PROCESSING ***")
	print(strftime('%H:%M:%S'), "- Start loading training data")
	start_time = time()
	X_train = pd.read_csv(x_train_file, index_col=ID)
	y_train = pd.read_csv(y_train_file, index_col=ID)
	print("Training data loaded in", utils.time_me(time() - start_time), "\n")

	if DEV == False:
		print("The full dataset will be used for training purpose.")
	else:
		print("Only", NUMBER_OF_ROWS_USED_FOR_DEV, "rows will be used for training purpose.")
		X_train = X_train.head(NUMBER_OF_ROWS_USED_FOR_DEV)
		y_train = y_train.loc[X_train.index,:]

	train_and_save_model(X_train, y_train, linear_model_file, lgbm_model_file, RUN_OPTUNA_STUDY)

	print("*** SCRIPT COMPLETED IN", utils.time_me(time() - script_start_time), "***")
