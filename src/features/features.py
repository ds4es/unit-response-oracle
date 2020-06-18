"""Functions for feature sets computation
"""

# Authors: Wenqi Shu-Quartier-dit-Maire
#          Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com
# License: MIT License

import os
import sys
import numpy as np
import pandas as pd
from numba import njit
import pickle

from src.utils.utils import *


def make_linear_model_prediction(X_test,linear_model):
    """Make a prediction for the given dataset and linear model

    Parameters
    ----------
    X_test : Pandas dataframe
        Input testing data.

    linear_model : object
        Linear model

    Returns
    -------
     : array, shape (n_samples,)
        Linear model prediction.

    """
    if __debug__: print("In make_linear_model_prediction()")

    return linear_model.predict(X_test)

    
def compute_feature_set_one(X, linear_model=None):
    """Compute a first set of features

    Before computing each feature we check if for each parameter it depends on is available

    Parameters
    ----------
    X : Pandas dataframe
        Input data.

    linear_model : Trained linear model from sklearn.linear_model.LinearRegression

    Returns
    -------
    X : Pandas dataframe
        Input data + first set of additional computed features

    """
    if __debug__: print("In compute_feature_set_one()")

    if 'routing engine estimated duration' in X.columns:
        if __debug__: print("Compute feature 1 for feature set one")
        # Compute the "estimated delta selection-presentation" feature
        X["estimated delta selection-presentation"] = make_linear_model_prediction(X[["routing engine estimated duration"]], linear_model)
    elif __debug__: print("Do not compute feature 1 for feature set one")

    if all([col in X.columns for col in ['selection time','estimated delta selection-presentation']]):
        if __debug__: print("Compute feature 2 for feature set one")
        # Compute the "estimated intervention time" feature
        X["estimated intervention time"] = (X["selection time"] + X["estimated delta selection-presentation"])
    elif __debug__: print("Do not compute feature 2 for feature set one")

    if all([col in X.columns for col in ['rescaled longitude before departure','rescaled longitude intervention','rescaled latitude before departure','rescaled latitude intervention']]):
        if __debug__: print("Compute feature 3 for feature set one")
        # Compute the "distance" feature
        X["distance"] = np.sqrt((X["rescaled longitude before departure"] - X["rescaled longitude intervention"]) ** 2 + (X["rescaled latitude before departure"] - X["rescaled latitude intervention"]) ** 2)
    elif __debug__: print("Do not compute 3 for feature set one")

    return X


def compute_feature_set_two(X, ID, *, use_cyclical, center_decay, vehicle_decay, kde_params=None):
    """Compute a second set of features

    Before computing each feature we check if for each parameter it depends on is available

    Parameters
    ----------
    X : Pandas dataframe
        Input data.

    ID : int
        Input data identifier column name

    use_cyclical : bool
        hyperparameter resulting from an optuna study

    center_decay : float
        hyperparameter resulting from an optuna study

    vehicle_decay : float
        hyperparameter resulting from an optuna study

    kde_params :
        hyperparameter resulting from an optuna study

    Returns
    -------
    X : Pandas dataframe
        Input data + second set of additional computed features

    """
    if __debug__: print("In compute_feature_set_two()")


    # Compute cyclical features
    if all([col in X.columns for col in ["time day", "time week"]]):

        if (use_cyclical):
            if __debug__: print("Compute cyclical features for feature set two")

            CYCLICAL = ["time day", "time week"]

            for f in CYCLICAL:
                X = X.merge(cyclical(X[f]), on=ID)
            
            X.drop(CYCLICAL, axis=1, inplace=True, errors='ignore')

    elif __debug__: print("Do not compute cyclical features")


    # Compute the exponentially weighted moving average (EWMA)
    if all([col in X.columns for col in ['selection time','rescue center']]):
        if __debug__: print("Compute exponentially weighted moving average for feature set two")

        X["ewma rescue center"] = ewma(
            X["selection time"], X["rescue center"], center_decay
        )

        X["ewma departure center"] = ewma(
            X["selection time"], X["departure center"], center_decay
        )

        X["ewma emergency vehicle"] = ewma(
            X["selection time"], X["departure center"], vehicle_decay
        )

    elif __debug__: print("Do not compute exponentially weighted moving average")


    # Compute the kernel density estimation (KDE)
    if ((kde_params is not None) and all([col in X.columns for col in ["estimated intervention time","selection time","rescaled longitude before departure", "rescaled latitude before departure","rescaled longitude intervention", "rescaled latitude intervention"]])):
        if __debug__: print("Compute kernel density estimation for feature set two")

        start_pos = X[["rescaled longitude before departure", "rescaled latitude before departure"]]
        end_pos = X[["rescaled longitude intervention", "rescaled latitude intervention"]]
        start_time = X["selection time"][:, None] * kde_params["time_distance_factor"]
        end_time = X["estimated intervention time"][:, None] * kde_params["time_distance_factor"]

        distributions = {
            "requests": np.hstack((end_pos, start_time)),
            "cars_start": np.hstack((start_pos, start_time)),
            "cars_end": np.hstack((end_pos, end_time)),
        }
        distributions["cars_start_end"] = np.vstack(
            (distributions["cars_start"], distributions["cars_end"])
        )

        n_distributions = len(distributions)
        from tqdm import tqdm

        for name_train, distrib_train in distributions.items():
            bandwidth = kde_bandwidth(distrib_train) * kde_params["bandwidth"]
            kde = KernelDensity(kernel=kde_params["kernel"], bandwidth=bandwidth)

            kde.fit(distrib_train)
            for name_eval, distrib_eval in tqdm(distributions.items()):
                X[f"kde {name_train} {name_eval}"] = kde.score_samples(distrib_eval)

    elif __debug__: print("Do not compute kernel density estimation")

    return X

