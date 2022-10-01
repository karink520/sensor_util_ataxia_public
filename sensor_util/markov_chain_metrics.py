# documented

import numpy as np
import scipy.stats as stats

"""
Calculates some quantities associated with markov chains of a given transition
matrix (entropy rate and stationary distribution)
"""


def entropy_rate_for_markov_chain(transition_matrix):
    """Calculates the entropy rate for a Markov chain from its transition matrix

    Parameters
    ----------
    transition_matrix : array_like
        2-D array, where the (i,j)th element is the probability fo transitioning from state i to state j

    Returns
    -------
    entropy_rate
        The entropy rate, scalar value in units of nats per time-step
    """

    num_states = transition_matrix.shape[0]
    stationery_distribution = stationary_distribution_for_markov_chain(
        transition_matrix
    )
    entropy_rate = 0
    for i in range(num_states):
        entropy_rate += stationery_distribution[i] * stats.entropy(
            transition_matrix[i, :]
        )
    return entropy_rate


def stationary_distribution_for_markov_chain(transition_matrix):
    """Calculates the stationary distribution for a Markov chain from its transition matrix

    Parameters
    ----------
    transition_matrix : array_like
        2-D array, where the (i,j)th element is the probability fo transitioning from state i to state j

    Returns
    -------
    stationary distribution: ndarray
        The stationary distribuion of the Markov chain, where the ith element gives the stationary distribution
        for the ith state.
    """

    eig_vals, eig_vectors = np.linalg.eig(transition_matrix)
    ind = np.argmax(eig_vals)
    stationary_distribution = np.absolute(
        np.linalg.inv(eig_vectors)[ind, :]
    )  
    stationary_distribution /= np.sum(stationary_distribution)
    return stationary_distribution
