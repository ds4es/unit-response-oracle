"""Config file for the different runnable scripts

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

import os

# To remove assert and __debug__-dependent statements (here designated as 'Debug mode OFF') add the -O flag
if __debug__:
    print('Debug mode ON')
else:    
    print('Debug mode OFF')
    # To avoid warnings as: ...loader.py:###: SyntaxWarning: "is" with a literal. Did you mean "=="?
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.getcwd()
RAW_DATA = "data/raw/"
PROCESSED_DATA = "data/processed/"
PREDICTED_DATA = "data/predicted/"
MODELS_DIR = 'models/'

# For developement purpose
DEV = False 
NUMBER_OF_ROWS_USED_FOR_DEV = 1000 # X_train number of rows used in training phase if DEV = True
USE_PARAMETERS_SUBSET_FOR_ROBUSTNESS_TEST = False # Set it to True to check if the training and prediction could still work with a random subset of input parameters 

# Optuna configuration parameters
RUN_OPTUNA_STUDY = False  # If 'False' the values from PARAMS (see below) )ill be used and if 'True' from the best study for OPTUNA_STUDY_NAME stored in OPTUNA_STORAGE
OPTUNA_STUDY_NAME = 'paris_fire_brigade_2018'
OPTUNA_STORAGE = 'sqlite:///trials.db'
OPTUNA_OPTIMIZATION_DIRECTION = 'maximize'
OPTUNA_LOAD_STUDY_IF_EXIST = True # For optuna.create_study load_if_exists parameter
OPTUNA_OBJECTIVE_AGGREGATION_FUNCTION = 'mean' # mean, median or min can be used
OPTUNA_NUMBER_OF_TRIALS = 1 # Number of consecutive trial in an Optuna study
OPTUNA_NUMBER_OF_PARALLEL_JOBS = -1 # If set to -1, the number used for n_jobs in study.optimize will be set to the number of CPU  
PARAMS_FROM_BEST_OPTUNA_STUDY_IN_DB = False # If 'True' will load (or create if do not exist) an Optuna study from OPTUNA_STORAGE and OPTUNA_STUDY_NAME

WITH_CROSS_VALIDATION = True # For training & prediction

SECONDS_IN_A_DAY = 60 * 60 * 24
SECONDS_IN_A_WEEK = SECONDS_IN_A_DAY * 7
SECONDS_IN_A_YEAR = SECONDS_IN_A_DAY * 365


##############
# Parameters specific to the dataset
##

LINEAR_MODEL_FILE = MODELS_DIR + 'paris_fire_brigade_2018/linear_model.pkl'
LGBM_MODEL_WITHOUT_CV_FILE = MODELS_DIR + 'paris_fire_brigade_2018/without_cv_lgbm_model.pkl'
LGBM_MODEL_WITH_CV_FILE = MODELS_DIR + 'paris_fire_brigade_2018/with_cv_lgbm_model.pkl'

X_TRAIN_PREPROCESSED_FILE = PROCESSED_DATA + "paris_fire_brigade_2018/x_train_preprocessed.zip"
X_TEST_PREPROCESSED_FILE = PROCESSED_DATA + "paris_fire_brigade_2018/x_test_preprocessed.zip"

Y_TRAIN_FILE = RAW_DATA + "paris_fire_brigade_2018/y_train.zip"
Y_TEST_FILE = RAW_DATA + "paris_fire_brigade_2018/y_test.zip"

Y_PREDICTED_WITHOUT_CV_FILE = PREDICTED_DATA + "paris_fire_brigade_2018/y_predicted_without_cv.csv"
Y_PREDICTED_WITH_CV_FILE = PREDICTED_DATA + "paris_fire_brigade_2018/y_predicted_with_cv.csv"

ID = "emergency vehicle selection"

# Paramaters coming from an Optuna study
PARAMS = {
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

# Geographic scope in which GPS data are and will be observed (used for normalization) 
# See utils.utils.rescale_minmax
LON_MIN = 2.101116
LAT_MIN = 48.65683
LON_MAX = 2.6470112
LAT_MAX = 49.043056


##############
# For another dataset the following arrays may require adjustment
##
OBJECTIVES = [
    "delta selection-departure",
    "delta departure-presentation",
    "delta selection-presentation",
]

CAT_FEATURES = [
    "alert reason category",
    "alert reason",
    "intervention on public roads",
    "location of the event",
    "emergency vehicle",
    "emergency vehicle type",
    "rescue center",
    "departed from its rescue center",
    "departure center",
]

NUM_FEATURES = [
    "floor",
    "routing engine estimated distance",
    "routing engine estimated duration",
    "delta position gps previous departure-departure",
    "routing engine estimated distance from last observed GPS position",
    "routing engine estimated duration from last observed GPS position",
    "time elapsed between selection and last observed GPS position",
    "updated routing engine estimated duration",
    "delta status preceding selection-selection",
    "GPS tracks count",
    "intervention count",
    "time day",
    "time week",
    "time year",
]

DROP_BEFORE_DUMP = [
    "date key sélection",
    "time key sélection",
    "GPS tracks departure-presentation",
    "GPS tracks datetime departure-presentation",
    "routing engine response",
    "routing engine estimate from last observed GPS position",
    "status preceding selection",
]

IGNORED = [
    "intervention",
    "selection time",
    "longitude before departure",
    "latitude before departure",
    "longitude intervention",
    "latitude intervention",
    "rescaled longitude before departure",
    "rescaled longitude intervention",
    "rescaled latitude before departure",
    "rescaled latitude intervention",
    "start time",
    "end time",
    "estimated speed",
    "estimated duration from speed",
    "estimated time factor",
    "estimated duration from time",
    "estimated intervention time",
    "estimated delta selection-presentation",
]


print(os.path.basename(__file__), "loaded!")

