import glob
import pprint
import math
import h5py
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import scipy
from statsmodels.tsa.tsatools import lagmat
from sklearn.datasets import make_spd_matrix
from imblearn.ensemble import BalancedRandomForestClassifier

from .generative_model import set_dimension_parameters
from .generative_model import set_hyperparameters_default
from .generative_model import set_parameters_from_model
from .generative_model import simulate_from_generative_model
from .generative_model import simulate_states
from .generative_model import show_samples_from_dynamics

from .filters import wavelet_denoising, lowpass_filter, highpass_filter
from .hdp_hmm_gibbs_sampling import *
from .load_sensor_data_utils import load_IMUtask_data_from_mat, get_task_inds_simplistic, get_paths_to_files_with_sessions_after
from .sensor_data_plotting import plot_IMUtask_sensor_data, hdp_hmm_basic_plot
from .rotation_invariance import project_onto_global_pc, project_onto_sliding_pc
from .markov_chain_metrics import entropy_rate_for_markov_chain, stationary_distribution_for_markov_chain


'''
Uses pre-estimated AR dynamics (loaded from specified .h5 file) to calculate hmm states and their associated features from files.
Saves features in a csv.
'''

def find_state_related_features_arr_hmm(filelist, path_to_previously_learned_params, save_features_to_new_file, file_num_end, file_num_start=0):
    '''
    Uses pre-estimated AR dynamics to estimate hmm states from data, calculate and save their associated features

    Parameters
    ----------
    filelist: array-like
        List of strings representing paths to files of IMU data to be processed
    
    path_to_previously_learned_params: str
        Path to h5 file containing previously sampled vector-autoregressive parameters.  We assume the h5 file has variables:
            A_samples: matrices controlling forward evolution of AR process for each state (will use its mean as fixed AR dynamics)
            Sigma_samples: samples of noise matrices for AR process for each state (will use its mean as fixed AR dynamics)
            pi_samples: transition probabilities for hmm (will use mean as inital value for pi to be sampled from data)
            pi_samples: transition probabilities for hmm (will use mean as inital value for pi to be sampled from data)

    save_features_to_new_file: bool
        If True, create a new file to save to.  If False, update existing file of features for sessions by including additional sessions

    file_num_end: int
        Index of last file in file_list to be processed

    file_num_start: int, optional
        Default=0.   Index of first file in file_list to be processed 


    Returns
    ---------
    None

    Notes
    --------
    The first set of parameters are adjustable, but should match what was used in the model
    '''

    # Load previously learned parameters
    f = h5py.File(path_to_previously_learned_params,'r')

    # Set parameters
    dimensions = {}
    dimensions['T'] = f['T'][0]
    dimensions['num_lags'] = f['num_lags'][0]
    dimensions['num_states'] = f['num_states'][0]
    dimensions['dim_y'] = f['dim_y'][0]
    true_hyperparameters = set_hyperparameters_default(dimensions)
    true_hyperparameters['kappa'] = f['kappa'][0]
    true_hyperparameters['gamma'] = f['gamma'][0]
    true_hyperparameters['alpha'] = f['alpha'][0]
    true_hyperparameters['K_0'] = f['K_0'][:]
    true_hyperparameters['K_0_inv'] = f['K_0_inv'][:]

    A_samples = f['A_samples'][:]
    pi_samples = f['pi_samples'][:]
    beta_samples = f['beta_samples'][:]
    # states_samples = f['states_samples'][:] Not needed
    Sigma_samples = f['Sigma_samples'][:]
    f.close()

    # Set parameters for state sampling
    n_samples = 250
    burn_in_for_dynamics=1000
    burn_in=50
    A_fixed = A_samples[:,:,:,burn_in_for_dynamics:].mean(axis=-1)
    Sigma_fixed = Sigma_samples[:,:,:,burn_in_for_dynamics:].mean(axis=-1)
    pi_initial = pi_samples[:,:,burn_in_for_dynamics:].mean(axis=2)
    beta_initial = beta_samples[:,burn_in_for_dynamics:].mean(axis=1)

    # Do the sampling
    state_sequences_arr = np.empty((len(filelist), dimensions['T']-dimensions['num_lags']), dtype=np.int32)
    features_dict = {}
    file_num=0
    for filename in filelist[file_num_start:file_num_end]:
        session = filename.rsplit("/", 1)[-1][0:16]  
        print(file_num)
        print(session)
        
        #Load data and transform it with the same preprocessing/sampling as data used for training
        try:
            sensor_data_df = load_IMUtask_data_from_mat(filename=filename)
        except:
            print("failed to load ", session, "file: ", filename)
            continue
        projected_gyro_temp, projected_acc_temp = preprocessing_updated(sensor_data_df)
        observations_temp=np.vstack([projected_gyro_temp.T, projected_acc_temp.T])
        dimensions['T'] = observations_temp.shape[1]
        T = dimensions['T']
        num_states = dimensions['num_states']
        observations_lags, observations_trimmed = lagmat(np.fliplr(observations_temp.T.copy()), maxlag = dimensions['num_lags'], trim = "both", original = 'sep')
        observations_lags = np.fliplr(observations_lags).T
        observations_trimmed = np.fliplr(observations_trimmed).T
        
        #Sample states and transition probs, keeping other parameters fixed
        samples = sample_with_fixed_dynamics(A_fixed, Sigma_fixed, pi_initial, beta_initial, observations_trimmed, observations_lags, dimensions, true_hyperparameters, n_samples=n_samples, verbose=1)

        #Put mode of states (post-burn-in) into a dataframe
        modes, _ = stats.mode(samples['states_samples'].astype(int)[:,burn_in:],axis=1)
        state_sequences_arr[file_num,:] = modes.astype(int).flatten()
        
        #Calculate posterior mean of transition probabilities and corresponding markov chain stationary distribution
        posterior_mean_pi =  samples['pi_samples'][:,:,burn_in:].mean(axis=2)
        stationary = stationary_distribution_for_markov_chain(posterior_mean_pi)
        
        #Calculate features and add them to a dictionary for this row (session)
        row_dict = {}
        row_dict['entropy_rate'] = entropy_rate_for_markov_chain(posterior_mean_pi)
        
        #Calculate frequency of mode state assignment in state samples for each time point
        overall_frequency_of_mode = (samples['states_samples'][:,burn_in:] == modes).sum(axis = 1) / (n_samples - burn_in)
        row_dict['overall_frequency'] = overall_frequency_of_mode.mean()
        
        #record length of same-state runs in state sequence (i.e. [0 0 0 1 1 0]) contains runs of 0s of length 3, 1
        state_runs={}
        for state in range(num_states):
            state_runs[state] = [] 
        for state, group in itertools.groupby(list(modes.flatten())):
            length_of_group = len(list(group))
            state_runs[state].append(length_of_group)
     
        for state in range(num_states):
            
            key = 'stationary_' + str(state)
            row_dict[key] = stationary[state]
            
            key = 'self_transition_' + str(state)
            row_dict[key] = posterior_mean_pi[state,state]
            
            key = 'empirical_frequency_' + str(state)
            if state >= len(np.bincount(modes.flatten())):
                #if this state has a label number that's greater than that of any state that was actually observed, it will fall outside
                #of the length of the bincount; in this case we want to set the empirical frequency to 0
                row_dict[key] = 0
            else:
                row_dict[key] = np.bincount(modes.flatten())[state] / modes.flatten().size
            
            key = 'state_certainty_' + str(state)
            row_dict[key] = overall_frequency_of_mode[modes.flatten() == state].mean()
            
            key = 'state_runs_length_mean_' + str(state)
            row_dict[key] = abs(np.array(state_runs[state]).mean())
            
            key = 'state_runs_length_stdev_' + str(state)
            row_dict[key] = abs(np.array(state_runs[state]).std())
        
        #Add this row to full sessions/features data dictionary
        features_dict[session] = row_dict
        
        #fig = hdp_hmm_basic_plot(projected_gyro_temp, projected_acc_temp, observations_trimmed, burn_in=burn_in, states_samples=samples['states_samples'], num_states=num_states, num_lags=num_lags, overall_frequency_of_mode=overall_frequency_of_mode, cmap=cmap, connect_adjacent_dots=True)
        #figs.append(fig)
        file_num +=1

        if file_num % 10 == 0:
            if not save_features_to_new_file:
                features_df = pd.read_csv(path_to_features_df_to_update)
                features_df.set_index(['Unnamed: 0'],inplace=True)
                features_dict_prev = features_df.to_dict('index')
            else:
                features_dict_prev = {}
            features_dict_prev.update(features_dict)
            features_df = pd.DataFrame.from_dict(features_dict_prev, orient='index')
            features_df.to_csv(path_to_save_features_to)

    if not save_features_to_new_file:
        features_df = pd.read_csv(path_to_features_df_to_update)
        features_df.set_index(['Unnamed: 0'],inplace=True)
        features_dict_prev = features_df.to_dict('index')
    else:
        features_dict_prev = {}
    features_dict_prev.update(features_dict)
    features_df = pd.DataFrame.from_dict(features_dict_prev, orient='index')
    features_df.to_csv(path_to_save_features_to)


def preprocessing_updated(df, start=100, stop=5000):
    '''
    Preprocesses gyroscope an acceleration data, fixed PCA isntead of sliding
    
    Parameters:
    df: DataFrame
        DataFrame with acceleration and gyroscope data with column names like 'gyro_task_x' and 'acc_task_y'
    
    start: int
        Trim to remove indices 0:start within the estimated task portion of the data

    stop: int
        Trim to remove indices stop:end within the estimated task portion of the data

    Returns
    --------
    preprocessed_gyro : ndarray
        preprocessed gyroscope data

    preprocessed_acc : ndarray
        preprocessed accelerometer data


    Notes
    -------
    For each of acceleration and gyroscoping 3-channel signals:
    Project onto the first PC
    Perform wavelet denoising
    Extract the task indices and clip the data to bins start:stop within those task indices
    Downsample by a factor of 10
    '''
    df.reset_index(inplace=True)
    task_inds, rest_inds = get_task_inds_simplistic(df['gyro_task_x'], df['gyro_task_y'], df['gyro_task_z'])

    df = df - df.mean(axis =0)
    df = df.iloc[task_inds]

    projected_gyro_temp = project_onto_global_pc(df['gyro_task_x'], df['gyro_task_y'], df['gyro_task_z'], num_pcs=1)
    projected_gyro_temp = np.expand_dims(wavelet_denoising(projected_gyro_temp),axis=1) 
    projected_gyro_temp = projected_gyro_temp[start:stop,:]
    
    projected_acc_temp = project_onto_global_pc(df['acc_task_x'], df['acc_task_y'], df['acc_task_z'], num_pcs=1)
    projected_acc_temp = np.expand_dims(wavelet_denoising(projected_acc_temp),axis=1)
    projected_acc_temp = projected_acc_temp[start:stop,:]

    preprocessed_gyro = scipy.signal.resample(projected_gyro_temp, int((stop-start)/10) )
    preprocessed_acc  = scipy.signal.resample(projected_acc_temp,  int((stop-start)/10) )
    return preprocessed_gyro, preprocessed_acc

