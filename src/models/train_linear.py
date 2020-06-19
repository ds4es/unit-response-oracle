"""For linear model training

Standalone execution
--------------------
	# Load your Python environment
	# Add to the PYTHONPATH variable the project root directory
	export PYTHONPATH=$PYTHONPATH:$(pwd)
	 
	# Call the __main__ function to launch a standalone linear training
	python src/models/train_linear.py

"""

# Authors: Wenqi Shu-Quartier-dit-Maire
#          Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com
# License: MIT License

import os
import sys
from time import time, strftime
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

from src.config import *
from src.utils import utils

def train_and_save_linear_model(X_train,y_train,linear_model_file):
	"""Train and save a linear model

	Parameters
	----------
	X_train : Pandas dataframe
		Input training data.

	y_train : Pandas dataframe
		Output training data.

	linear_model_file : string
		Path to a linear model pickle file

	Returns
	-------
	linear_model : sklearn.linear_model.LinearRegression
		Trained linear model

	"""
	if __debug__: print("In train_and_save_linear_model()")

	linear_model = LinearRegression().fit(X_train,y_train)

	pickle.dump(linear_model, open(linear_model_file, 'wb'))

	return linear_model
	

# For a shell standalone execution
if __name__ == '__main__':

	script_start_time = time()

	print("\n*** START MAIN FUNCTION FROM", os.path.dirname(os.path.abspath(__file__)) + os.sep + os.path.basename(__file__), "***\n")

	print("*******************************************************************")
	print("*                                                                 *")
	print("*  This script trains and saves a linear model on preprocessed    *")  
	print("*  data for service units response times prediction.              *")
	print("*                                                                 *")
	print("*******************************************************************\n")

	if (len(sys.argv) == 4):
		x_train_file = sys.argv[1]
		y_train_file = sys.argv[2]
		linear_model_file = sys.argv[3]
	else:
		# Default input
		x_train_file = X_TRAIN_PREPROCESSED_FILE
		y_train_file = Y_TRAIN_FILE
		linear_model_file = LINEAR_MODEL_FILE

	if (os.path.isfile(x_train_file) and os.path.isfile(y_train_file) and os.path.isfile(linear_model_file)) == False:
		print("USAGE: python src/models/train_linear.py [X_TRAIN_PREPROCESSED_FILE] [Y_TRAIN_FILE] [LINEAR_MODEL_FILE]")
		sys.exit()

	print("*** EQUIVALENT LAUNCHED COMMAND ***")
	print(sys.executable, sys.argv[0],x_train_file,y_train_file,linear_model_file,"\n")
	
	print("*** INPUT FILES ***")
	print("- Input training data file:", x_train_file)
	print("- Output training data file:", y_train_file)
	print("- Linear model will be saved into:", linear_model_file)
	print("")
	
	print("*** START DATA PROCESSING ***")
	print("Start loading training data at", strftime('%H:%M:%S'))
	start_time = time()
	X_train = pd.read_csv(x_train_file, index_col=ID)
	y_train = pd.read_csv(y_train_file, index_col=ID)
	print("Training data loaded in", utils.time_me(time() - start_time), "\n")

	print("Start training linear regression model at", strftime('%H:%M:%S'))
	start_time = time()
	train_and_save_linear_model(X_train[["routing engine estimated duration"]], y_train["delta selection-presentation"], linear_model_file)
	print("Training completed and resulting model saved in", utils.time_me(time() - start_time))
	print("Model saved under", linear_model_file, "\n")

	print("*** SCRIPT COMPLETED IN", utils.time_me(time() - script_start_time), "***")


