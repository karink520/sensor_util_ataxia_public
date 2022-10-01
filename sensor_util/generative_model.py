import numpy as np
import scipy.stats as stats
from .matrix_normal_inverse_wishart import sample_matrix_normal_inverse_wishart_omit_b, sample_matrix_normal_inverse_wishart


def set_dimension_parameters(num_states=3, dim_y=2, T=1000, num_lags=2):
    '''
    Returns a dictionary with the specified parameters.

    num_states is the number of hmm states
    num_lags is the number of lags in the AR process
    T is the total length of the signal (in number of time bins)
    dim_y is the number of channels in the signal (e.g. 6 if we have 3 acc channels and 3 gyro channels)
    '''
    return {"num_states": num_states, "num_lags": num_lags, "T": T, "dim_y": dim_y}


def set_hyperparameters_default(dimensions, include_bias_term=False):
    '''
    Returns a dictionary with some default hyperparameter values

    '''

    num_lags, dim_y = dimensions["num_lags"], dimensions["dim_y"]
    # set kappa (stickiness), and note that it will not be learned
    kappa = 100

    # Autoregressive parameters
    # set nu0 (scalar) as in demo of pyhsmm_autoregressive)
    nu_0 = 5  #
    # set S0 (dim_y x dim_y)
    S_0 = 1 / 100 * np.eye(dim_y)
    # set M0
    M_0 = 0.25 * np.ones((dim_y, dim_y * num_lags))
    if include_bias_term:
        M_0 = np.hstack((M_0, np.zeros((dim_y, 1)))) # mean zero for the bias (b) term

    # set K0 (nlags*dim_y+1 x nlags*dim_y+1) the ARD prior, a diagonal matrix with entries drawn from invGamma(1/25, 1/25)
    if include_bias_term:
        k = stats.invgamma.rvs(a=1, scale=25, size=num_lags * dim_y + 1)
    else:
        k = stats.invgamma.rvs(a=1, scale=25, size=num_lags * dim_y)
    # k[num_lags*dim_y+1] = 0 #  set last element to zero, so as not to shrink the affine contribution of b
    K_0 = np.diag(k)
    K_0_inv = np.linalg.inv(K_0)

    # Parameters that probably won't be learned
    # set gamma (scalar), or sample it from Gamma(1, 1/100). (Should this be learned?)
    gamma = stats.gamma.rvs(a=1, scale=100)
    # set alpha (scalar), or sample it from Gamma(1, 1/100). (Should this be learned?)
    alpha = stats.gamma.rvs(a=1, scale=100)

    true_hyperparameters = {
        "kappa": kappa,
        "gamma": gamma,
        "alpha": alpha,
        "S_0": S_0,
        "M_0": M_0,
        "K_0": K_0,
        "K_0_inv": K_0_inv,
        "nu_0": nu_0,
    }

    return true_hyperparameters


def set_parameters_from_model(dimensions, hyperparameters, include_bias_term=False):
    '''
    Draw samples
    '''


    num_states, dim_y, num_lags = (
        dimensions["num_states"],
        dimensions["dim_y"],
        dimensions["num_lags"],
    )
    alpha, gamma, kappa = (
        hyperparameters["alpha"],
        hyperparameters["gamma"],
        hyperparameters["kappa"],
    )
    nu_0, S_0, M_0, K_0 = (
        hyperparameters["nu_0"],
        hyperparameters["S_0"],
        hyperparameters["M_0"],
        hyperparameters["K_0"],
    )

    # set beta (num_states x 1) , needed for the mean of the transition matrix (creates sharing
    # between the transition probabilities from different states)
    beta = np.random.dirichlet(alpha=np.ones(num_states) * gamma / num_states)

    # set pi (num_states x num_states) , the transition matrix for states
    pi = np.zeros((num_states, num_states))
    for i in range(num_states):
        kappa_i = np.zeros((num_states,))
        kappa_i[i] = kappa
        pi[i] = np.random.dirichlet(alpha=alpha * beta + kappa_i)

    # Autoregressive parameters
    # set A, b, Sigma
    #print("calculating A, b, Sigma")
    A = np.zeros((dim_y, dim_y * num_lags, num_states))
    if include_bias_term:
        b=[]
    Sigma = np.zeros((dim_y, dim_y, num_states))
    # A's taken from pyhsmm-autoregressive example
    As = [
        0.99 * np.hstack((-np.eye(2), 2 * np.eye(2))),
        np.array(
            [
                [np.cos(np.pi / 6), -np.sin(np.pi / 6)],
                [np.sin(np.pi / 6), np.cos(np.pi / 6)],
            ]
        ).dot(np.hstack((-np.eye(2), np.eye(2))))
        + np.hstack((np.zeros((2, 2)), np.eye(2))),
        np.array(
            [
                [np.cos(-np.pi / 6), -np.sin(-np.pi / 6)],
                [np.sin(-np.pi / 6), np.cos(-np.pi / 6)],
            ]
        ).dot(np.hstack((-np.eye(2), np.eye(2))))
        + np.hstack((np.zeros((2, 2)), np.eye(2))),
    ]

    for i in range(num_states):
        if include_bias_term:
            _, b0, Sigma0 = sample_matrix_normal_inverse_wishart(nu_0, S_0, M_0, K_0)
            b.append(b0)
        else:
            A0, Sigma0 = sample_matrix_normal_inverse_wishart_omit_b(nu_0, S_0, M_0, K_0)
        #A[:, :, i] = As[i]
        A[:,:,i] = A0
        Sigma[:, :, i] = Sigma0

    parameters = {"beta": beta, "pi": pi, "A": A, "Sigma": Sigma}
    if include_bias_term:
        parameters["b"] = b
    return parameters


def simulate_states(pi, T, initial_state=0):
    # returns a vector of states, and counts of each transition
    x = [initial_state]
    num_states = pi.shape[0]
    transition_counts = np.zeros(
        (num_states, num_states)
    )  # transition_counts(k,j) represents counts of transitionfs from state k to j
    for t in range(1, T):
        current_state = x[t - 1]
        next_state = np.random.choice(
            num_states, 1, replace=False, p=pi[current_state].flatten()
        )[0]
        x.append(next_state)
        transition_counts[current_state, next_state] += 1

    return x, transition_counts.astype(int)


def simulate_from_generative_model(
    dimensions, hyperparameters, parameters, initial_state=0
):
    '''
    Simulate hmm states and AR signals, given the autoregressive parameters and markov transition matrix pi

    Returns
    -------
    x:
        state sequence
    y: 
        AR-based emissions

    Notes
    -----
    If parameters has a key 'b', then it will be used as a bias term. (As in, y_t+1 = Ay_t + b).
    If parameters has no key 'b', no bias term is used (or b=0 is used)
    '''

    pi, A, Sigma = parameters["pi"], parameters["A"], parameters["Sigma"]
    dim_y, num_lags, T, num_states = dimensions["dim_y"], dimensions["num_lags"], dimensions["T"], dimensions["num_states"]
    
    # if the bias term is included, get it, otherwise set it to all zeros
    b = parameters.get("b", [np.zeros((1, dim_y))] * num_states)

    # sample x, the vector of hidden states (length T), each one an integer in 0,..,L-1
    x, transition_counts = simulate_states(pi, T, initial_state=initial_state)

    # Observations
    # initialize first few y's
    y = np.empty((dim_y, T))
    for t in range(num_lags):
        y[:, t] = stats.norm.rvs(size=dim_y)

    for t in range(num_lags, T):
        # TODO: memoize
        current_state = x[t]
        mean_y = np.matmul(
            A[:, :, current_state],
            (y[:, t - num_lags:t]).reshape(dim_y * num_lags, 1, order="F"),
        ).flatten() # + b[current_state].flatten() # MODIFIED TO INCLUDE B
        ynew = stats.multivariate_normal.rvs(
            mean=mean_y, cov=Sigma[:, :, current_state]
        )
        y[:, t] = ynew

    return x, y, transition_counts


def show_samples_from_dynamics(
    A, Sigma, dimensions, hyperparameters, b=None, length_of_trace=10, num_traces=5, state=0
):
    num_states = dimensions["num_states"]
    dimensions["T"] = length_of_trace
    parameters = {"pi": np.identity(num_states), "A": A, "Sigma": Sigma}
    if b is not None:
        parameters["b"] = b
    x, y, _ = simulate_from_generative_model(
        dimensions, hyperparameters, parameters, initial_state=state
    )
    return x, y
