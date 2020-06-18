import os
import sys
import numpy as np
import pandas as pd
from numba import njit
import pickle

def time_me(elapsed_time):
    microseconds_time = elapsed_time * 1000000

    microseconds = int((microseconds_time) % 1000) ;
    milliseconds = int((microseconds_time / 1000) % 1000) ;
    seconds = int((microseconds_time / 1000000) % 60) ;
    minutes = int(((microseconds_time / (1000000*60)) % 60));
    hours   = int(((microseconds_time / (1000000*60*60)) % 24));

    if(hours != 0):
        return ("%d hours %d min %d sec" % (hours, minutes, seconds)) 
    elif(minutes != 0):
        return ("%d min %d sec" % (minutes, seconds))    
    elif(seconds != 0):
        return ("%d sec %d ms" % (seconds, milliseconds))
    else :
        return ("%d ms %d µs" % (milliseconds, microseconds))


# def rescale(df,mapping):
#     if __debug__: print("In rescale()")

#     min_ = min(df[f].min() for f in mapping)
#     max_ = max(df[f].max() for f in mapping)
#     for f, nf in mapping.items():
#         df[nf] = (df[f] - min_) / (max_ - min_)

def rescale_minmax(df,mapping,min,max):
    if __debug__: print("In rescale_lon_lat()")

    for f, nf in mapping.items():
        df[nf] = (df[f] - min) / (max - min)


def linreg(x, y):
    if __debug__: print("In linreg()")

    nan = np.isnan(x) | np.isnan(y)
    x = x[~nan]
    y = y[~nan]
    xm = x.mean()
    ym = y.mean()
    x = x - xm
    y = y - ym
    alpha = x.dot(y) / x.dot(x)
    beta = ym - alpha * xm
    return alpha, beta


def linreg_transform(x, y, return_coefs=False):
    if __debug__: print("In linreg_transform()")

    a, b = linreg(x, y)
    if return_coefs:
        return x * a + b, a, b
    else:
        return x * a + b


def to_int_keys_pd(series):
    """
    Handle nan correctly
    """
    if __debug__: print("In to_int_keys_pd()")
    
    if series.isnull().any():
        index = {v: i for i, v in enumerate(set(series[~series.isnull()]), 1)}
        return (
            series.swifter.progress_bar(False)
            .apply(lambda x: index[x] if not pd.isnull(x) else 0)
            .astype("int32")
        )
    else:
        index = {v: i for i, v in enumerate(set(series))}
        return (
            series.swifter.progress_bar(False).apply(lambda x: index[x]).astype("int32")
        )


def cyclical(series):
    if __debug__: print("In cyclical()")

    name = series.name
    return pd.DataFrame(
        {f"{name} cos": np.cos(series), f"{name} sin": np.sin(series)},
        index=series.index,
    )


# @njit(cache=True)
def _ewma(n_cat, time, category, decay):
    """
    Optimized function to compute the ewma per category
    """
    if __debug__: print("In _ewma()")

    n = len(time)
    order = np.argsort(time)

    out = np.empty(n)
    last_time = np.full(n_cat, time.min())
    value = np.zeros(n_cat)

    for i in range(n):
        t = order[i]
        cat = category[t]
        time_delta = time[t] - last_time[cat]
        out[i] = value[cat] = 1 + value[cat] * decay ** time_delta
        last_time[cat] = time[t]

    return out


def ewma(time, category, decay):
    if __debug__: print("In ewma()")

    assert 0 <= decay < 1

    if isinstance(time, pd.Series):
        time = time.values
    if isinstance(category, pd.Series):
        category = category.values

    uniq, category = np.unique(category, return_inverse=True)
    n_cat = len(uniq)

    # rule of thumb assuming uniform rate and uniform categories
    scale = (time.max() - time.min()) / len(time) * n_cat
    time = (time - time.min()) / scale

    return _ewma(n_cat, time, category, decay)


def sample_params(trial):
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


def save_object(obj, filename): 
    if __debug__: print("In save_object()")

    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# Root mean squared logarithmic error 
def rmsle(real, predicted):
    if __debug__: print("In rmsle()")

    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


print(os.path.basename(__file__), "loaded!")
