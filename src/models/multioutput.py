"""Modified MultiOutputRegressor class to allow NaNs
"""

# Authors: Wenqi Shu-Quartier-dit-Maire
#          Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com
# License: MIT License

from sklearn.multioutput import (
    np,
    Parallel,
    delayed,
    check_X_y,
    check_array,
    is_classifier,
    _fit_estimator,
    check_is_fitted,
    MultiOutputRegressor,
)
#from sklearn.linear_model import ElasticNet

class MultiOutputRegressorWithNan(MultiOutputRegressor):
    """Modified MultiOutputRegressor class to allow NaNs
    """

    def fit(self, X, y, sample_weight=None):    
        """Fit the model to data X and the multioutput target y.

        from https://github.com/scikit-learn/scikit-learn/blob/7e85a6d1f/sklearn/multioutput.py#L124

        Parameters
        ----------
        X :
            Input training data.

        y :
            Multioutput training data.

        """
        if __debug__: print("In MultiOutputRegressorWithNan fit()")

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement" " a fit method")

        # This line is modified to allow NaNs
        X, y = check_X_y(
            X, y, multi_output=True, accept_sparse=True, force_all_finite="allow-nan"
        )

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError("Underlying estimator does not support" " sample weights.")

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(self.estimator, X, y[:, i], sample_weight)
            for i in range(y.shape[1])
        )

        return self

    def predict(self, X):
        if __debug__: print("In MultiOutputRegressorWithNan predict()")

        """Predict multi-output variable using a model trained for each target variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Input testing data.

        Returns
        -------
        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.

        """
        check_is_fitted(self)
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement" " a predict method")

    
        X = check_array(X, accept_sparse=True, force_all_finite="allow-nan")
        
        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X) for e in self.estimators_
        )

        return np.asarray(y).T

