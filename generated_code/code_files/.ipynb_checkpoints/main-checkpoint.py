"""
Main entry point for running experiments with the linear regression model.
"""

import linear_regression_model
import utils

def run_exp():
    """
    Run an experiment with the linear regression model.

    This function loads the dataset, trains the model, and evaluates its performance.
    """
    X, y = utils.load_data('data.csv')

    model = linear_regression_model.LinearRegressionModel()
    model.fit(X, y)

    # Example prediction (not implemented)
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

def eval():
    """
    Evaluate the model's performance on a separate test set.
    """
    # Placeholder for evaluation logic
    pass

if __name__ == "__main__":
    run_exp()