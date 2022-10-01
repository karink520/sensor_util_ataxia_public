import collections
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sn
import matplotlib.pyplot as plt


def plot_IMUtask_sensor_data(df, sampling_frequency=128):
    '''
    Plots IMU task sensor data for acc, gyro, magno, and orientation against time (4 subplots)

    Parameters
    ----------
    df : DataFrame
        with columns for the x, z, z components of acc, gyro, and magno, and 4 orientation components
        assumes columsn named like acc_task_x, magno_task_y, orient_1, orient_2, etc.
    sampling_frequency : float or int
        sampling frequency in Hz, used to scale the horizontal time axis to display labels in seconds.
        Default value is 128Hz (sampling frequency for APDM IMU data)

    Notes
    ----------
    Time labels for the horizontal axis are displayed in seconds, with the start of the df at time 0.
    No return value - function displays plots as a side-effect.
    '''

    fig, ax = plt.subplots(4, 1)
    t = np.linspace(0, df.shape[0]/sampling_frequency, df.shape[0])
    ax[0].plot(t, df[["acc_task_x", "acc_task_y", "acc_task_z"]])
    ax[0].title.set_text("acc")
    ax[1].plot(t, df[["gyro_task_x", "gyro_task_y", "gyro_task_z"]])
    ax[1].title.set_text("gyro")
    ax[2].plot(t,df[["magno_task_x", "magno_task_y", "magno_task_z"]])
    ax[2].title.set_text("magno")
    ax[3].plot(t,df[["orient_1", "orient_2", "orient_3", "orient_4"]])
    ax[3].title.set_text("orient")

    plt.xlabel("time (s)")
    fig.set_size_inches(15, 10)


def plot_IMUtask_sensor_data_psd(df, freq, col_name_suffix="_w_psd"):
    """
    Plots the power spectral density as a semilog-y line plot for IMU data (subplots for acc, gyro, magno, orient)
    given a DataFrame containing the PSDs

    Parameters
    ----------
    df : DataFrame
        With columns for the power spectral density of each of each component of the IMU data.
        Columns are assumed to be named like acc_task_x + col_name_suffix, gyro_task_y + col_name_suffix, etc.
    freq : array_like
        array/list of frequencies represented in the psd
    col_name_suffix: str
        Suffix added to the original names of the IMU task variable (so if col_name_suffix is _w_psd, columns have 
        names like acc_task_x_w_psd)
        
    """
    fig, ax = plt.subplots(4, 1)

    ax[0].semilogy(
        freq,
        df[
            [
                "acc_task_x" + col_name_suffix,
                "acc_task_y" + col_name_suffix,
                "acc_task_z" + col_name_suffix,
            ]
        ],
    )
    ax[0].set_xlim(right=30)
    ax[0].title.set_text("acc")
    ax[1].semilogy(
        freq,
        df[
            [
                "gyro_task_x" + col_name_suffix,
                "gyro_task_y" + col_name_suffix,
                "gyro_task_z" + col_name_suffix,
            ]
        ],
    )
    ax[1].title.set_text("gyro")
    ax[1].set_xlim(right=30)
    ax[2].semilogy(
        freq,
        df[
            [
                "magno_task_x" + col_name_suffix,
                "magno_task_y" + col_name_suffix,
                "magno_task_z" + col_name_suffix,
            ]
        ],
    )
    ax[2].set_xlim(right=30)
    ax[2].title.set_text("magno")
    ax[3].semilogy(
        freq,
        df[
            [
                "orient_1" + col_name_suffix,
                "orient_2" + col_name_suffix,
                "orient_3" + col_name_suffix,
                "orient_4" + col_name_suffix,
            ]
        ],
    )
    ax[3].title.set_text("orient")
    ax[3].set_xlim(right=30)

    plt.xlabel("frequency")
    fig.set_size_inches(10, 15)


def hdp_hmm_basic_plot(
    observations_trimmed,
    burn_in,
    states_samples,
    num_states,
    time_step_size = 1/128.0 * 10,
    plot_overall_frequency_of_mode=False,
    cmap=None,
    connect_adjacent_dots=False
):
    '''
    Creates a plot of the the (1d, preprocessed) gyroscope and accelerometer signals, with each time point
    color coded to the state estimated by the ar-hmm model (the mode of the sampled states for that time point)

    Parameters
    -----------
    observations_trimmed: ndarray
        Dimensions are 2 x T.  Will plot each of the two rows in a separate graph.  (Most likely the first
        is preprocessed acceleration, the second preprocessed gyroscope)
    burn_in: int
        State modes will be calculated from the samples remaining after the first burn_in state samples are discarded
    states_samples: ndarray
        Array of state samples. Dimensions are T x n_samples where n_samples is number of samples in the Gibbs sampler
    Color coding for a time point is based on the mode of all the state samples for that 
        time point (excluding the first burn_in samples)
    num_states: int
        Number of states of the hmm in the ar-hmm model
    time_step_size: float   
        The time duration (in seconds) of one time bin in the gyro and acc data.  Used for labeling the time axis correctly
    plot_overall_frequency_of_mode: bool, optional
        Default: False.  If true, include a third plot that visualizes the certainty associated with each state estimate.
        State estimates come from the *mode* of the state_samples produced by the Gibbs sampler.  The frequency of mode
        is the fraction (between 0 and 1) of state samples at a given time point that are equal to the mode state sample
        at that point.
    cmap: cm.cmap, optional
        Default: None.  The colormap for the color coding of points by state.  If None, gist_rainbow cmap is used
    connect_adjacent_dots: bool, optional
        Default: False.  If True, adjacent time points that have the same state will be connected by a line whose color
        matches the color for the state
    Returns:
    ----------
    fig
        A matplotlib figure with separate axes for the accelerometer and gyroscope data.
    Notes
    ----------
    A side-effects is displaying the figure
    '''

    # Create axes
    if plot_overall_frequency_of_mode:
        num_axes = 3
    else:
        num_axes = 2
    _, axs = plt.subplots(num_axes, sharex=True)

    # Calculate mode states
    modes, _ = stats.mode(states_samples.astype(int)[:, burn_in:], axis=1)
    modes = np.array([x[0] for x in modes])

    if cmap is None:
        cmap = plt.cm.get_cmap("gist_rainbow", num_states+1)

    time = np.arange(observations_trimmed.shape[1])*time_step_size
    time =np.array(time).flatten()
  
    # Color coded scatter plots
    axs[0].plot(time[:], observations_trimmed[0, :], lw=0.5, c="lightgray")
    axs[0].scatter(
        time,
        observations_trimmed[0, :],
        s=6,
        c=[cmap(m) for m in modes]
    )

    axs[1].plot(time[:], observations_trimmed[1, :], lw=0.5, c="lightgray")
    axs[1].scatter(
        time,
        observations_trimmed[1, :],
        s=4,
        c=[cmap(m) for m in modes]
    )

    # Connect adjacent dots in the scatter plot if they are the same state
    if connect_adjacent_dots:
        for idx, m in enumerate(modes[:-1]):
            if m == modes[idx+1]:
                axs[0].plot(time[idx:idx+2], observations_trimmed[0, idx:idx+2], lw=0.5, c=cmap(m))
                axs[1].plot(time[idx:idx+2], observations_trimmed[1, idx:idx+2], lw=0.5, c=cmap(m))


    if plot_overall_frequency_of_mode:
        n_samples = states_samples.shape[1]
        overall_frequency_of_mode = (states_samples[:,burn_in:] == np.expand_dims(modes, 1)).sum(axis = 1) / (n_samples - burn_in)
        axs[2].scatter(
            time,
            overall_frequency_of_mode,
            s=2,
            c="gray",
        )
    else:
        axs[2].plot(states_samples[:, burn_in:].mean(axis=1))

    plt.xlabel('time (s)')
    plt.tight_layout()
    fig = plt.gcf()
    plt.gcf().set_size_inches(18, 6)
    plt.show()
    return fig


def class_probs_reliability(df):
    '''
    One way of visualizing binary class probalities for ataxia vs. control for subjects with
    more than one session, to see variance in model classification scores and model errors

    Parameters
    -----------
    df: DataFrame
        Must have columns 'ID', 'gen_diagnosis_num' with values 3 and 1 (control and ataxia), 
        'spec_diagnosis', 'session' and 'predict_probs_ataxia'.  

    Returns
    -----------
    fig
        A matplotlib figure.  Each line corresponds to one subject, with dots on the x-axis
        corresponding to the classification score given to each session for that subject by the model 
        (larger is more likely ataxic).  Misclassifications (using a classification threshold of 0.5) 
        are marked with red Xs.  Each line is labeled with its ID and specific diagnosis
    
    Notes
    -------
    Side effect: displays the picture. Prints the number of individuals with repeat visits Could use some clean-up.
    '''

    repeatIDs = [ID for ID, count in collections.Counter(df['ID']).items() if count > 1] # count > 1 to see subjects with repeat sessions; count > 0 to see all
    print(len(repeatIDs), "individuals with repeat visits")

    fig, ax = plt.subplots()
    i = 1000
    for ID in repeatIDs:
        this_df = df[df['ID'] == ID] # A small dataframe with only rows corresponding to a particular ID
        is_correct = this_df[((this_df['gen_diagnosis_num'] == 3) & (this_df['predict_probs_ataxia'] < 0.5)) | ((this_df['gen_diagnosis_num'] == 1) & (this_df['predict_probs_ataxia'] > 0.5))] 
        is_not_correct = this_df[~(((this_df['gen_diagnosis_num'] == 3) & (this_df['predict_probs_ataxia'] < 0.5)) | ((this_df['gen_diagnosis_num'] == 1) & (this_df['predict_probs_ataxia'] > 0.5)))] 

        ax.scatter(is_correct['predict_probs_ataxia'], i*np.ones_like((is_correct['predict_probs_ataxia'])), marker = 'o')
        ax.scatter(is_not_correct['predict_probs_ataxia'], i*np.ones_like((is_not_correct['predict_probs_ataxia'])), marker = 'x',c='red')
        for k in range(this_df.shape[0]):
            ax.text(this_df['predict_probs_ataxia'].iloc[k] + 0.01, i - 5 + 10*(k % 2), str(this_df['session'].iloc[k])[6:], fontsize=10  )
       
        ax.text(1.1, i, str(this_df['spec_diagnosis'].iloc[0]) + " , " + str(this_df['ID'].iloc[0]), fontsize=10)
        i -= 20
    ax.axes.get_yaxis().set_visible(False)
    ax.axvline(x=0.5, color='gray', linestyle='--')
    fig.set_size_inches(10,20)
    fig.show()

    return fig


def reliability_scatter(df, reliability_var='predict_probs_ataxia', true_diffs_var=None, show_plots=False, color_by_diagnosis=False,fontsize=12):
    '''
    Plots test vs. retest scores and calculates the correlations.

    Parameters
    ------------
    df: DataFrame
        Has columns 'ID', 'session', and a column with a name that is the same as reliability_var.  If
        true_diffs_var is not None, it must be a column of df.  If color_by_diagnisis is True, df
        must have a gen_diagnosis_num column
    reliability_var: str
        A column name of df.  The variable of interest for which we will plot and compute the test-retest
        scores.
    true_diffs_var: str, optional
        Default: None.  If not None, must be a column of df.  Should be something like a severity score where
        the differences between consecutive visits would be of interest.
    show_plots: bool, optional
        default: False.  If true, show the plots
    color_by_diagnosis: bool, optional
        default: False.  Color dots in scatter by gen_diagnosis_num value.  See colors variable below for
        the color coding

    Returns
    ------------
    fig
        A scatter plot with a point for every consecutive in time pair of points for a subject with
        more than one session.  (So a subject with 3 visits would have two corresponding points, 
        one comparing visits 1-2, the other 2-3). The more recent score is on the x-axis, the less-recent
        score of each pair is the y-axis.
    score_df: DataFrame
        Has a row for each pair of consecutive visits by a subject.
        Has columns 'score_recent', 'score_less_recent' and 'score_gap' corresponding to the reliability_var
        scores for two consecutive visits and the difference between them (most recent - less recent).
        Also has columns 'most_recent_session' and 'less_recent_session' to give the sessions for each pair
        If a true_diffs_var was provided, also includes a column 'true_diffs_gap' for the difference between
        the value of true_diff_vars at the two visits (most recent - less recent)
    r: float
        Pearson's correlation coefficient for test/retest values of reliability_var
    '''
    
    
    repeatIDs = [ID for ID, count in collections.Counter(df['ID']).items() if count > 1]
    num_repeat_visits = len(repeatIDs)
    print(len(repeatIDs), "individuals with repeat visits")
    scores_dicts = []
    fig, ax = plt.subplots()
    colors = ['','red', 'green', 'blue','seagreen','orange','','yellow']
    
    for ID in repeatIDs:
        this_df = df[df['ID'] == ID]
        #sort by date of session
        this_df['date'] = this_df['session'].str[6:16]
        this_df.sort_values(by='date', ascending=False, inplace=True)
        for row_idx in range(this_df.shape[0]-1):      
            score_recent = this_df[reliability_var].iloc[row_idx] # most recent score
            score_recent_2 = this_df[reliability_var].iloc[row_idx + 1] # second most recent score
            score_gap = score_recent - score_recent_2
            scores_dicts.append({'score_recent':score_recent, 'score_less_recent': score_recent_2, 'score_gap': score_gap, 'most_recent_session': this_df['session'].iloc[0], 'less_recent_session': this_df['session'].iloc[1] })
            if true_diffs_var is not None:
                true_diff = this_df[true_diffs_var].iloc[0] - this_df[true_diffs_var].iloc[1] # most recent minus second_most recent
                scores_dicts[-1]['true_diffs_gap'] = true_diff
            else:
                scores_dicts[-1]['true_diffs_gap'] = None
            if color_by_diagnosis:
                scores_dicts[-1]['color'] = colors[this_df['gen_diagnosis_num'].iloc[0]]
        
    scores_df = pd.DataFrame(scores_dicts)
    scores_df.sort_values(by='score_gap', inplace=True)
    r = np.corrcoef(scores_df['score_recent'], scores_df['score_less_recent'])[0,1]

    if color_by_diagnosis:
        ax.scatter(scores_df['score_recent'], scores_df['score_less_recent'], c=scores_df['color'])
    else: 
        ax.scatter(scores_df['score_recent'], scores_df['score_less_recent'])

    ax.set_xlabel('most recent session', fontsize=fontsize)
    ax.set_ylabel('less_recent_session', fontsize=fontsize)
    if show_plots:
        max_val = max([np.max(scores_df['score_recent']), np.max(scores_df['score_less_recent'])])
        plt.plot([0, max_val], [0, max_val], '--', color = 'k')
        plt.ylabel('most recent session', fontsize=fontsize)
        plt.xlabel('less recent session', fontsize=fontsize)
        fig.set_size_inches(6,6)
        fig.show()
    
    return fig, scores_df, r, num_repeat_visits 

    
def plot_task_sensor_data_psd(df, freq, col_name_suffix="_w_psd"):
    fig, ax = plt.subplots(4, 1)

    ax[0].semilogy(
        freq,
        df[
            [
                "acc_task_x" + col_name_suffix,
                "acc_task_y" + col_name_suffix,
                "acc_task_z" + col_name_suffix,
            ]
        ],
    )
    ax[0].set_xlim(right=30)
    ax[0].title.set_text("acc")
    ax[1].semilogy(
        freq,
        df[
            [
                "gyro_task_x" + col_name_suffix,
                "gyro_task_y" + col_name_suffix,
                "gyro_task_z" + col_name_suffix,
            ]
        ],
    )
    ax[1].title.set_text("gyro")
    ax[1].set_xlim(right=30)
    ax[2].semilogy(
        freq,
        df[
            [
                "magno_task_x" + col_name_suffix,
                "magno_task_y" + col_name_suffix,
                "magno_task_z" + col_name_suffix,
            ]
        ],
    )
    ax[2].set_xlim(right=30)
    ax[2].title.set_text("magno")
    ax[3].semilogy(
        freq,
        df[
            [
                "orient_1" + col_name_suffix,
                "orient_2" + col_name_suffix,
                "orient_3" + col_name_suffix,
                "orient_4" + col_name_suffix,
            ]
        ],
    )
    ax[3].title.set_text("orient")
    ax[3].set_xlim(right=30)

    plt.xlabel("frequency")
    fig.set_size_inches(10, 15)


def plot_state_frequencies_by_severity(df, T, num_states, target, task_labels, task_labels_for_display):

    '''
    Display bar plot showing the frequency of each hmm-state, with separate bars based on specified target (e.g. 'bars_arm_R').
    Will show a separate plot for each task_label

    Parameters
    -------------
    df: DataFrame
        Has column name like empirical_frequency_4_heelshin, where the number is a state, and the last
        portion is the task name
    T: int
        Number of time-steps, each of which is assigned a state
    num_states: int
        Number of distinct states in the hmm model
    target: str
        The name of a column of df.  Separate bars will be drawing for different values of target

    Returns
    ---------
    fig

    Notes
    ------
    Currently configured for bars_arm

    '''

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    bars_labels = ['Control', 'BARS: 0-0.5', 'BARS: 1-1.5', 'BARS: 2+']
    num_tasks = len(task_labels)
    total_state_counts = np.zeros((num_tasks,len(bars_labels), num_states))

    for _, row in df.iterrows():
        if row['gen_diagnosis_num'] == 3:
            bars_score_index = 0
        elif row[target] == 0.5 or row[target] == 0:
            bars_score_index = 1       
        elif row[target] == 1 or row[target] == 1.5:
            bars_score_index = 2   
        elif row[target] >= 2:
            bars_score_index = 3
        else:
            continue
        
        for state in range(num_states):
            for task_index in range(num_tasks):
                col_name = 'empirical_frequency_' + str(state) + '_' + task_labels[task_index]
                state_count = row[col_name] * T
                if math.isnan(state_count):
                    state_count = 0
                total_state_counts[task_index, bars_score_index, state] +=  state_count

    # bin specifically for fnf/lb/heelshin
    
    width = 0.2
    cmap = plt.cm.get_cmap("bwr", num_states+1)
    colors = [cmap(x+2) for x in range(len(bars_labels))] 
    for task_index in range(2):
        for bars_score_index in range(len(bars_labels)):
                ax[task_index].bar(np.arange(0,num_states) - 0.5 + bars_score_index * width, total_state_counts[task_index, bars_score_index, :] / total_state_counts[task_index,bars_score_index,:].sum(), width, color=colors[bars_score_index], label=bars_labels[bars_score_index])
                ax[task_index].spines["top"].set_visible(False)
                ax[task_index].spines["right"].set_visible(False)
                ax[task_index].spines["left"].set_visible(False)
                ax[task_index].spines["bottom"].set_visible(False)
                ax[task_index].set_xticks(np.arange(num_states))
        ax[task_index].set_title(task_labels_for_display[task_index])

    target = 'bars_leg_R'
    for _, row in df.iterrows():
        if row['gen_diagnosis_num'] == 3:
            bars_score_index = 0
        elif row[target] == 0.5 or row[target] == 0:
            bars_score_index = 1       
        elif row[target] == 1 or row[target] == 1.5:
            bars_score_index = 2   
        elif row[target] >= 2:
            bars_score_index = 3
        else:
            continue
        
        for state in range(num_states):
            task_index = 2
            col_name = 'empirical_frequency_' + str(state) + '_' + task_labels[task_index]
            state_count = row[col_name] * T
            if math.isnan(state_count):
                state_count = 0
            total_state_counts[task_index, bars_score_index, state] +=  state_count

    task_index = 2
    for bars_score_index in range(len(bars_labels)):
        ax[task_index].bar(np.arange(0,num_states) - 0.5 + bars_score_index * width, total_state_counts[task_index, bars_score_index, :] / total_state_counts[task_index,bars_score_index,:].sum(), width, color=colors[bars_score_index], label=bars_labels[bars_score_index])
        ax[task_index].spines["top"].set_visible(False)
        ax[task_index].spines["right"].set_visible(False)
        ax[task_index].spines["left"].set_visible(False)
        ax[task_index].spines["bottom"].set_visible(False)
        ax[task_index].set_xticks(np.arange(num_states))
    ax[task_index].set_title(task_labels_for_display[task_index])

    ax[1].legend(prop={'size':12})
    fig.set_size_inches(10,3)
    return fig


def feature_importances_bar_chart(feature_imps):
    '''
    Plot a horizontal bar chart for feature importances

    Parameters
    ----------
    feature_imps: Series
        Index is name of feature, value is the importance
        
    Returns
    --------
    fig
    '''
    fig, ax = plt.subplots()
    y_pos = np.arange(len(feature_imps))
    y_pos = np.flip(y_pos)
    y_pos = [2*i for i in y_pos]
    ax.barh(y_pos, [feature for feature in feature_imps],height=1.5,color='crimson')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(feature_imps.index))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.set_size_inches(5,0.5*len(y_pos))
    return fig

def pretty_confusion_matrix(confusion_matrix, class_labels_list, normalize_with_true_total=False, font_size=24):
    '''
    Displayes a given confusion matrix with class labels, annotations and background colors, optionally normalized

    Parameters
    ----------
    confusion_matrix: ndarray
        A confusion matrix (can be 2-class or multi-class)
    class_labels_list: array-like
        Ordered list of class labels as strings
    normalize_with_true_total: bool, optional
        Default: False.  If true, normalize so that each row adds to one.  That is, present results as fraction
        of popultaions with a certain true diagnosis class getting classified one way vs. another.
    
    Returns
    --------
    fig

    Notes
    ------
    sklearn's confusion_matrix is oriented so that c_i,j is the number known to be in i and predicted to be in
    j, so that each ROW is a true diagnosis.

    '''
    fig, ax = plt.subplots(figsize=(4,4))
    sn.set(font_scale=2)
    if normalize_with_true_total:
        confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1)[:,None]  #the [:,None] is for broadcasting
    df_cm = pd.DataFrame(confusion_matrix, class_labels_list,
              class_labels_list)
    sn.heatmap(df_cm, annot=True, cmap="OrRd",annot_kws={"size": 30}, cbar=False, square=True)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = font_size, rotation=0)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = font_size, va='center')

    return fig