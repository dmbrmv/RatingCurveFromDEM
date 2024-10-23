import numpy as np


def rmse(y_true, y_pred):
    """Calculate the Root Mean Squared Error (RMSE) for two given arrays of values.

    Parameters
    ----------
    y_true : array-like
        The true values.
    y_pred : array-like
        The predicted values.

    Returns
    -------
    rmse : float
        The Root Mean Squared Error between ``y_true`` and ``y_pred``.
    """
    return np.sqrt(np.nanmean((np.array(y_true) - np.array(y_pred)) ** 2))
