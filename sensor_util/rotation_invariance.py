
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
import numpy as np

'''
Preprocessing functions for introducing rotation invariance by
projecting onto principal components (global pcs or sliding window)
'''

def project_onto_global_pc(x, y, z, num_pcs=1):
    """Projects data from the three input channels onto principal component(s)

    Parameters
    ----------
    x : array-like
        1-D array of time series data
    y : array-like
        1-D array of time series data (from another orientation)
    z : array-like
        1-D array of time series data (from another orientation)
    num_pcs : int, optional.
        Number of principal components.  Should be between 1 and 3, inclusive.  Default is 1.

    Returns
    -------
    principalComponents: ndarray
        A 2D array of dimensions (length of x, y, or z) x num_pcs

    Notes
    ------
    x, y, and z should be of equal length
    """

    data = np.column_stack([x, y, z])
    data = StandardScaler(with_std=False).fit_transform(
        data
    )  
    pca = PCA(n_components=num_pcs)
    principalComponents = pca.fit_transform(data)
    return principalComponents


def project_onto_sliding_pc(x, y, z, window_size, num_pcs=1):
    """Projects data from the three input channels onto local principal component(s), calculated based on a sliding window.

    Parameters
    ----------
    x : array-like
        1-D array of time series data
    y : array-like
        1-D array of time series data (from another orientation)
    z : array-like
        1-D array of time series data (from another orientation)
    window_size: int
        For each point, the principal component(s) is/are calculated based on a window of this size centerd around the point.
    num_pcs : int, optional
        Number of principal components.  Should be between 1 and 3, inclusive.  Default is 1.

    Returns
    -------
    principalComponents: ndarray
        A 2D array of dimensions (length of x, y, or z) x num_pcs

    Notes
    ------
    x, y, and z should be of equal length
    """
    data = np.column_stack([x, y, z])
    data = StandardScaler(with_std=False).fit_transform(data)
    pca = PCA(n_components=num_pcs)
    projected_path = np.zeros((x.size, num_pcs))
    for t in range(
        math.floor(window_size / 2 + 1), x.size - math.floor(window_size / 2 + 1)
    ):
        windowed_data = data[
            t - math.floor(window_size / 2) - 1: t + math.floor(window_size / 2), :
        ]
        pca = pca.fit(windowed_data)
        principalComponents = pca.transform(data[t, :].reshape(1, -1))
        projected_path[t, :] = principalComponents[0, :]
    return projected_path