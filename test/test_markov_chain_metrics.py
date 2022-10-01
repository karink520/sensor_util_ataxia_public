from sensor_util.markov_chain_metrics import (
    entropy_rate_for_markov_chain,
    stationary_distribution_for_markov_chain,
)

import numpy as np


def test_markov_chain_metrics():
    tol = 0.0001
    transition_matrix = np.array([[0.1, 0.6, 0.3], [0.5, 0.3, 0.2], [0.2, 0.1, 0.7]])
    entropy_rate = entropy_rate_for_markov_chain(transition_matrix)
    stationary_distribution = stationary_distribution_for_markov_chain(
        transition_matrix
    )

    assert (
        np.absolute(
            stationary_distribution[0]
            - stationary_distribution.dot(transition_matrix)[0]
        )
        < tol
    )
    assert (
        np.absolute(
            stationary_distribution[1]
            - stationary_distribution.dot(transition_matrix)[1]
        )
        < tol
    )
    assert (
        np.absolute(
            stationary_distribution[2]
            - stationary_distribution.dot(transition_matrix)[2]
        )
        < tol
    )
    assert np.absolute(entropy_rate - 0.89237937) < tol
