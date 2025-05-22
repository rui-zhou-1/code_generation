"""
Utility functions for data processing and evaluation.
"""

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    """
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y