from sensor_util.matrix_normal_inverse_wishart import (
    sample_matrix_normal_inverse_wishart,
)
from sensor_util.matrix_normal_inverse_wishart import (
    sample_matrix_normal_inverse_wishart_omit_b,
)

import numpy as np
import scipy.stats as stats


def test_sample_matrix_normal_inverse_wishart_omit_b():
    dim_y = 2
    num_lags = 3
    nu_0 = 5
    S_0 = 1 / 100 * np.eye(dim_y)
    M_0 = 0.25 * np.ones((dim_y, dim_y * num_lags))
    k = stats.invgamma.rvs(a=1, scale=25, size=num_lags * dim_y)
    K_0 = np.diag(k)
    A, Sigma = sample_matrix_normal_inverse_wishart_omit_b(nu_0, S_0, M_0, K_0)
    assert A.shape == M_0.shape
    assert A.shape == ((dim_y, dim_y * num_lags))
    assert Sigma.shape == ((dim_y, dim_y))
    assert Sigma.shape == S_0.shape


def test_sample_matrix_normal_inverse_wishart():
    dim_y = 2
    num_lags = 3
    nu_0 = 5
    S_0 = 1 / 100 * np.eye(dim_y)
    M_0 = 0.25 * np.ones((dim_y, dim_y * num_lags))
    k = stats.invgamma.rvs(a=1, scale=25, size=num_lags * dim_y)
    K_0 = np.diag(k)
    A, b, Sigma = sample_matrix_normal_inverse_wishart(nu_0, S_0, M_0, K_0)
    assert A.shape[0] == M_0.shape[0]
    assert A.shape[1] == M_0.shape[1] - 1
    assert A.shape == ((dim_y, dim_y * num_lags - 1))
    assert Sigma.shape == ((dim_y, dim_y))
    assert Sigma.shape == S_0.shape
    assert b.shape == (M_0.shape[0],)
