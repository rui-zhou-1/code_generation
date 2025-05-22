"""
LinearRegressionModel
A class for implementing and training a linear regression model.
"""

import numpy as np

class LinearRegressionModel:
    """
    Linear regression model class.

    Methods:
        fit(X, y): Fit the model according to the given training data.
        predict(X): Predict using the linear model.
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
            X (np.ndarray): Training vector of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).

        Returns:
            self (object): Trained LinearRegressionModel instance.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.coef_ = theta_best[1:]
        self.intercept_ = theta_best[0]

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters:
            X (np.ndarray): Samples of shape (n_samples, n_features).

        Returns:
            y_pred (np.ndarray): Predicted values for samples in X.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coef_) + self.intercept_