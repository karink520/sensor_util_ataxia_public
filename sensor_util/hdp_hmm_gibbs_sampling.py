import scipy.stats as stats
import numpy as np
from .matrix_normal_inverse_wishart import sample_matrix_normal_inverse_wishart_omit_b, sample_matrix_normal_inverse_wishart


def sample_auxiliary_state_variables(
    beta, transition_counts, hyperparameters, dimensions
):
    kappa = hyperparameters["kappa"]
    alpha = hyperparameters["alpha"]
    # returns m | beta
    num_states = dimensions["num_states"]
    # in CRF metaphor, m_kj is how many tables in restaurant k considered j
    m = np.zeros((num_states, num_states))
    for k in range(num_states):
        for j in range(num_states):
           # p = (alpha * beta / (alpha * beta + num_states) ) * (alpha * beta + kappa) / (alpha * beta + transition_counts + kappa)
            for l in range(transition_counts[k, j]):  # TODO: VECTORIZE
                if k == j:
                    p = (alpha * beta[j] / (alpha * beta[j] + num_states)) * (
                        (alpha * beta[j] + kappa) / (alpha * beta[j] + l + kappa)
                    )
                else:
                    p = (alpha * beta[j] / (alpha * beta[j] + num_states)) * (
                        (alpha * beta[j]) / (alpha * beta[j] + l)
                    )
                m[k, j] += stats.bernoulli.rvs(p=p)
    return m


def sample_state_parameters(states, transition_counts, m, hyperparameters, dimensions):
    num_states = dimensions["num_states"]
    alpha, gamma, kappa = (
        hyperparameters["alpha"],
        hyperparameters["gamma"],
        hyperparameters["kappa"],
    )
    # transition_counts: the j_kth entry is the number of transitions from state j to k
    kappa_i = kappa * np.eye(num_states)

    beta = np.random.dirichlet(alpha=gamma / num_states + m.sum(axis=1))  # CHECK AXIS!

    pi = np.zeros((num_states, num_states))
    for state_label in range(num_states):
        pi[state_label] = np.random.dirichlet(
            alpha=alpha * beta
            + transition_counts[state_label, :]
            + kappa_i[state_label]
        )

    return beta, pi


def sample_autoregressive_parameters(
    states_sample, observations_trimmed, observations_lags, hyperparameters, dimensions, include_bias_term=False
):
    num_states, dim_y, num_lags, T = (
        dimensions["num_states"],
        dimensions["dim_y"],
        dimensions["num_lags"],
        dimensions["T"],
    )
    nu_0, _, K_0_inv, M_0, S_0 = (
        hyperparameters["nu_0"],
        hyperparameters["K_0"],
        hyperparameters["K_0_inv"],
        hyperparameters["M_0"],
        hyperparameters["S_0"],
    )

    n = np.bincount(states_sample)
    if n.size < num_states:  # TODO: make sure states get counted in correct order
        n = np.concatenate((n, np.zeros(num_states - n.size)))
    nu_n = n + nu_0

    # initializing with the correct sizes
    A = np.zeros((dim_y, dim_y * num_lags, num_states))
    Sigma = np.zeros((dim_y, dim_y, num_states))
    b = np.zeros((dim_y, num_states))

    for state_label in range(num_states):

        obs_state = observations_trimmed[:, np.array(states_sample) == state_label] # dim_y x (number of states with state equal to state_label)?
        obs_lags_state = observations_lags[:, np.array(states_sample) == state_label] # (dim_y * num_lags) x (number of states with state equal to state_label)?

        if include_bias_term:
            obs_lags_state = np.vstack((obs_lags_state, np.ones((1, obs_lags_state.shape[1])))) #TODO: check this

        outer_products_obs_lags = obs_state @ obs_lags_state.T
        outer_products_lags_lags = obs_lags_state @ obs_lags_state.T
        outer_products_obs_obs = obs_state @ obs_state.T


        K_n_inv = K_0_inv + outer_products_lags_lags
        K_n = np.linalg.inv(K_n_inv)
        M_n = (outer_products_obs_lags + M_0 @ K_0_inv) @ K_n
        S_n = (
            S_0 + outer_products_obs_obs + M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T
        )
        if include_bias_term:
            A0, b0, Sigma0 = sample_matrix_normal_inverse_wishart(
                nu_n[state_label], S_n, M_n, K_n
            )
            b[:, state_label] = b0
        else:
            A0, Sigma0 = sample_matrix_normal_inverse_wishart_omit_b(
                nu_n[state_label], S_n, M_n, K_n
            )
        A[:, :, state_label] = A0
        Sigma[:, :, state_label] = Sigma0

    return A, b, Sigma


def sample_states(pi, A, b, Sigma, observations_trimmed, observations_lags, dimensions):
    # NEED TO TEST REINCLUSION OF B
    num_states = dimensions["num_states"]
    T_trimmed = observations_trimmed.shape[1]

    # compute the conditional probalities p(observation_(t+1) | x_(t+1)=j, A, b, Sigma) for each state j
    # for use in both forward and backward calculations
    obs_conditional_prob = np.zeros((num_states, T_trimmed))
    obs_conditional_prob_log = np.zeros((num_states, T_trimmed))
    for t in range(T_trimmed - 1, 0 - 1, -1):
        for j in range(num_states):
            obs_conditional_prob_log[j, t] = stats.multivariate_normal.logpdf(
                observations_trimmed[:, t],
                mean=(A[:, :, j] @ observations_lags[:, t]), # + b[:, j],
                cov=Sigma[:, :, j],
            )
            obs_conditional_prob[j,t] = np.exp(obs_conditional_prob[j, t])

    # backward_messages is matrix of backward-passed messages
    # backward_messages(k, t) is prob of obs(t+t:T) given state(t) and params pi, A, b, Sigma
    backward_messages = np.ones((num_states, T_trimmed))

    # Pass messages back. Leave backward_messages(:,T) as ones
    for t in range(T_trimmed - 2, 0 - 1, -1):  # this was T-2 and lags before were t+1
        log_scaling_factor = np.amax(obs_conditional_prob[:, t + 1])
        for k in range(num_states):
            # backward_messages[k,t] = sum(pi[k,:] * backward_messages[:,t+1] * obs_conditional_prob[:,t+1]) / scaling_factor[t] 
            # elementwise products, scalar division
            backward_messages[k, t] = sum(
                pi[k, :]
                * backward_messages[:, t + 1]
                * np.exp(obs_conditional_prob_log[:, t + 1] - log_scaling_factor)
            )
        #print(backward_messages[:, t].sum())    
        backward_messages[:, t] /= backward_messages[:, t].sum()

    # Sample states forward
    states = np.zeros((T_trimmed))
    transition_counts = np.zeros((num_states, num_states))
    for t in range(1, T_trimmed):
        current_state = int(states[t - 1])
        log_transition_probs = (
            np.log(pi[current_state, :])
            + np.log(obs_conditional_prob[:, t])
            + np.log(backward_messages[:, t])
        )
        pi[current_state, :] * obs_conditional_prob[:, t] * backward_messages[:, t]
        next_state = int(sample_discrete_from_log(log_transition_probs))
        states[t] = next_state
        transition_counts[current_state, next_state] += 1

    return states.astype(int), transition_counts.astype(int)


# sample_discrete_from_log from mattjj
def sample_discrete_from_log(p_log, axis=0, dtype=np.int32):
    "samples log probability array along specified axis"
    cumvals = np.exp(p_log - np.expand_dims(p_log.max(axis), axis)).cumsum(
        axis
    )  # cumlogaddexp
    thesize = np.array(p_log.shape)
    thesize[axis] = 1
    randvals = np.random.random(size=thesize) * np.reshape(
        cumvals[[slice(None) if i is not axis else -1 for i in range(p_log.ndim)]],
        thesize,
    )
    return np.sum(randvals > cumvals, axis=axis, dtype=dtype)


def sample_with_fixed_dynamics(
    A_fixed,
    Sigma_fixed,
    pi_initial,
    beta_initial,
    observations_trimmed,
    observations_lags,
    dimensions,
    true_hyperparameters,
    n_samples,
    verbose=1,
):
    pi_samp = pi_initial
    beta_samp = beta_initial

    T = dimensions["T"]
    num_lags = dimensions["num_lags"]
    num_states = dimensions["num_states"]
    states_samples_temp = np.zeros((T - num_lags, n_samples))
    m_samples_temp = np.zeros((num_states, num_states, n_samples))
    pi_samples_temp = np.zeros((num_states, num_states, n_samples))
    beta_samples_temp = np.zeros((num_states, n_samples))
    b_samp = []

    for i in range(n_samples):
        if i % 100 == 0 and verbose and i != 0:
            print("samples complete:", i)
        states_sample, transition_counts = sample_states(
            pi_samp,
            A_fixed,
            b_samp,
            Sigma_fixed,
            observations_trimmed,
            observations_lags,
            dimensions,
        )
        states_samples_temp[:, i] = np.array(states_sample)

        m_samp = sample_auxiliary_state_variables(
            beta_samp, transition_counts, true_hyperparameters, dimensions
        )
        m_samples_temp[:, :, i] = m_samp

        beta_samp, pi_samp = sample_state_parameters(
            states_sample, transition_counts, m_samp, true_hyperparameters, dimensions
        )
        beta_samples_temp[:, i] = beta_samp
        pi_samples_temp[:, :, i] = pi_samp

    return {
        "states_samples": states_samples_temp,
        "beta_samples": beta_samples_temp,
        "pi_samples": pi_samples_temp,
    }
