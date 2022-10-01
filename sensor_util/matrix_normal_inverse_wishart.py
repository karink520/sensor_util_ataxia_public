import scipy.stats as stats

"""
Functions for sampling from matrix-normal-inverse-wishart distributions
"""


def sample_matrix_normal_inverse_wishart(nu, S, M, K):
    """ Returns samples from an inverse wishart and a matrix normal distribution

    Parameters
    ----------
    nu : int
        Degrees of freedom for inverse wishart (greater than or equal to scale matrix)
    S : array_like
        Scale matrix for inverse wishart (symmetric, positive definite)
    M: array_like
        Mean of matrix normal distribution
    K: array_like
        Column covariance for matrix normal distribution.  Symmetric and positive definite,
        with each dimension equal to the number of columns of tbe mean matrix M

    Returns
    -------
    A : ndarray
        All but the last column of a random matrix drawn from the matrix normal distribution 
        with row covariance of Sigma, column covariance of K, and mean of M.  Should be almost
        the same shape as M, but with one fewer columns.
    b : ndarray
        The last column of a random matrix drawn from the matrix normal distribution 
        with row covariance of Sigma, column covariance of K, and mean of M. Has one column and
        the same number of rows as M.
    Sigma :
        A random matrix drawn from an inverse wishart distribution with scale S and degrees of 
        freedom nu

    Notes:
    -------
    Note the draw from the matrix normal distribution gives a concatentation of A and b.
    """

    Sigma = stats.invwishart.rvs(df=nu, scale=S)
    Ab = stats.matrix_normal.rvs(mean=M, rowcov=Sigma, colcov=K)
    b = Ab[:, Ab.shape[1] - 1]
    A = Ab[:, 0: Ab.shape[1] - 1]
    return A, b, Sigma


def sample_matrix_normal_inverse_wishart_omit_b(nu, S, M, K):
    """ Returns samples from an inverse wishart and a matrix normal distribution

    Parameters
    ----------
    nu : int
        Degrees of freedom for inverse wishart (greater than or equal to scale matrix)
    S : array_like
        Scale matrix for inverse wishart (symmetric, positive definite)
    M: array_like
        Mean of matrix normal distribution
    K: array_like
        Column covariance for matrix normal distribution.  Symmetric and positive definite,
        with each dimension equal to the number of columns of tbe mean matrix M

    Returns
    -------
    A : ndarray
        A random matrix drawn from the matrix normal distribution with row covariance of Sigma,
        column covariance of K, and mean of M. 
    Sigma :
        A random matrix drawn from an inverse wishart distribution with scale S and degrees of 
        freedom nu

    Notes:
    -------
    The 'omit b' in the function title is a reference to the notatation of an AR-HMM where A is the
    dynamics and b is a bias term that, if included, could for convenient be concatenated with A in
    the matrix normal draw.
    """
    
    Sigma = stats.invwishart.rvs(df=nu, scale=S)
    A = stats.matrix_normal.rvs(mean=M, rowcov=Sigma, colcov=K)
    return A, Sigma
