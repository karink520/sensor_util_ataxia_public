from scipy.io import loadmat
import pandas as pd
import numpy as np
import math


"""
Functionality for loading IMU data and time-frequency transformed data from matlab,
and calculating basic time-frequency features.

Also includes some utilities that give extra functionality for preprocessing if
desired, such as averaging values from left and right sensors, and setting
severity scores for control subjects to 0, and recoding diagnoses, and comparing
dates of sessions based on file names.
"""


def load_IMUtask_data_from_mat(session=None, sensor=None, task_name=None, filename=None):
    '''
    Loads IMU task data from a .mat file and puts it in a pandas DataFrame

    Parameters:
    ----------
    session: str, optional
        E.g. in IDIDX_YYYY_MM_DD format.  If filename is not provided, then this function will 
        load a file with path task_name/session_APDM_sensor_task_name.mat
    sensor: str, optional
        E.g. in RUE or LUE.  If filename is not provided, then this function will 
        load a file with path task_name/session_APDM_sensor_task_name.mat
    task_name: str, optional
        E.g. FNF2 or Mirroring. If filename is not provided, then this function will 
        load a file with path task_name/session_APDM_sensor_task_name.mat
    filename: str, optional
        Path to file

    Returns
    ---------
    df: DataFrame   
        A pandas DataFrame with rows corresponding to values of ts_task and columns like 'acc_task_x',
        'magno_task_z' and 'orient_task_4'.  Thirteen columns: three each for acc, gyro, magno and 4
        for orient.  Index comes from ts_task

    Notes
    --------
    Must provide either (session, sensor, and task_name) OR filename.  If both are provided,
    filename will be used.
    The matlab file to be loaded is assumed to have variables ts_task, acc_task, gyro_task, magno_task, and orient_task
    '''

    if filename is None:
        filename = (
            task_name 
            + "/"
            + session
            + "_APDM"
            + "_"
            + sensor
            + "_"
            + task_name
            + ".mat"
        )
    matlabdata = loadmat(filename)

    acc_column_names = ["acc_task_x", "acc_task_y", "acc_task_z"]
    gyro_column_names = ["gyro_task_x", "gyro_task_y", "gyro_task_z"]
    magno_column_names = ["magno_task_x", "magno_task_y", "magno_task_z"]
    orient_column_names = ["orient_1", "orient_2", "orient_3", "orient_4"]

    df_acc = pd.DataFrame(
        data=matlabdata["acc_task"],
        index=matlabdata["ts_task"][:, 0],
        columns=acc_column_names,
    )
    df_gyro = pd.DataFrame(
        data=matlabdata["gyro_task"],
        index=matlabdata["ts_task"][:, 0],
        columns=gyro_column_names,
    )
    df_magno = pd.DataFrame(
        data=matlabdata["magno_task"],
        index=matlabdata["ts_task"][:, 0],
        columns=magno_column_names,
    )
    df_orient = pd.DataFrame(
        data=matlabdata["orient_task"],
        index=matlabdata["ts_task"][:, 0],
        columns=orient_column_names,
    )

    df = df_acc.merge(df_gyro, how="inner", left_index=True, right_index=True)
    df = df.merge(df_magno, how="inner", left_index=True, right_index=True)
    df = df.merge(df_orient, how="inner", left_index=True, right_index=True)

    return df


def set_zeros_for_control_subjects(df, column_names_to_zero_for_control_subjects):
    '''
    Used to set specified disease-related scores to zero for control subjects (i.e. to replace NaNs)

    Parameters
    ------------
    df: DataFrame
        Needs to contain all of the columns in column_names_to_zero_for_control_subjects, and
        column 'gen_diagnosis_num', where controls are coded with a gen_diagnosis_num of 3
    column_names_to_zero_for_control_subjects: array-like
        A list of strings, each of which is a column-name of df that will be set to zero for
        control subjects.  These should be columns that specify disease scores like bars_total

    Returns
    ------------
    df: DataFrame
        Modified input DataFrame where each row corresponding to a control subject has the value of
        each column in column_names_to_zero_for_control_subjects set to 0

    Notes
    ------------
    This function assumes that no control subject has a valid non-zero disease severity score!
    '''

    for column_name in column_names_to_zero_for_control_subjects:
        if column_name in df.columns:
            df.loc[df.gen_diagnosis_num == 3, column_name] = 0
    return df


def average_left_and_right_features(df):
    '''
    Takes a dataframe with features that correspond to right and left side, and averages corresponding features from opposite sides
    
    Parameters
    ----------
    df: DataFrame
        Assumes corresponding left and right hand features are labeled with identical name except for the inclusion of RUE/LUE or RLE/LLE
    
    Returns
    -------
    new_df: DataFrame
        The new dataframe, with corresponding LUE and RUE features averaged, and LLE and RLE features averaged
        That is, something like the entropy for the gyro signal for the LUE would be averaged with
        the entropy for the gyro signal for the RUE.

    new_column_list
        A list of columns with their new names
    '''

    new_df = pd.DataFrame()
    new_column_list = []
    new_df = pd.concat([new_df, df['ID']], axis = 1, sort=False)
    for col_name in df.columns:
        if 'RUE' in col_name:
            col_name_split = col_name.split('_RUE')
            left_col_name = col_name_split[0] + '_LUE' + col_name_split[1]
            reduced_col_name = col_name_split[0] + col_name_split[1]
            new_df[reduced_col_name] = np.nanmean(df[[col_name,left_col_name]], axis=1)
            new_column_list.append(reduced_col_name)
        elif 'RLE' in col_name:
            col_name_split = col_name.split('_RLE')
            left_col_name = col_name_split[0] + '_LLE' + col_name_split[1]
            reduced_col_name = col_name_split[0] + col_name_split[1]
            new_df[reduced_col_name] = np.nanmean(df[[col_name,left_col_name]], axis=1)
            new_column_list.append(reduced_col_name)
        else:
            if 'LUE' not in col_name and 'LLE' not in col_name:
                new_df = pd.concat([new_df, df[col_name]], axis = 1)
        new_df.set_index(df.index)
    return new_df, new_column_list


def filter_and_recode_gen_diagnosis_num(df, gen_diagnoses_to_consider, combine_ataxia_and_AT=False):
    ''' 
    Recodes the gen_diagnosis_column to zeros and 1s, for comparisons of (1,3), (1,2), or (2,3)

    Parameters
    ---------
    df: DataFrame
        Assumed to have a column gen_diagnosis_num

    gen_diagnosis_to_consider: set
        Set of values of gen_diagnosis_num that will be included in the output DataFrame.  Rows 
        of df corresponding to other values of gen_diagnosis_num will be deleted.  If gen_diagnosis_to_consider
        is one of the following, then gen_diagnosis_num will also be recoded to a 0-1 value as follows:
        #(1,3) -> (1,0)
        #(2,3) -> (1,0)
        #(7, 3) -> (1,0)
        #(11, 3) -> (1,0)
        #(10, 3) -> (1,0)
        #(1,2) -> (1,0)

    combine_ataxia_and_AT: bool, optional 
        If true, set the gen_diagnosis_num for AT AND ataxia to be 1.  Default=False

    Returns
    ---------
    df2: DataFrame
        The DataFrame with data from gen_diagnosis_num not in gen_diagnosis_to_consider removed,
        and the remaining gen_diagnosis_to_consider recoded
        
    Notes
    --------  
    Useful in the notebook exploration
    '''


    df2  = df.copy()
    if combine_ataxia_and_AT:
         df.loc[df.gen_diagnosis_num == 7,'gen_diagnosis_num'] = 1

    df2 = df[df['gen_diagnosis_num'].isin(gen_diagnoses_to_consider)]
    print(len(gen_diagnoses_to_consider))
    if len(gen_diagnoses_to_consider) == 2:
        if 1 in gen_diagnoses_to_consider and 3 in gen_diagnoses_to_consider: #(1,3) -> (1,0)
            df2.loc[df2.gen_diagnosis_num == 3,'gen_diagnosis_num'] = 0
        elif 2 in gen_diagnoses_to_consider and 3 in gen_diagnoses_to_consider:  #(2,3) -> (1,0)
            df2.loc[df2.gen_diagnosis_num == 3,'gen_diagnosis_num'] = 0
            df2.loc[df2.gen_diagnosis_num == 2,'gen_diagnosis_num'] = 1
        elif 1 in gen_diagnoses_to_consider and 2 in gen_diagnoses_to_consider: #(1,2) -> (1,0)
            df2.loc[df2.gen_diagnosis_num == 2,'gen_diagnosis_num'] = 0
        elif 11 in gen_diagnoses_to_consider and 3 in gen_diagnoses_to_consider: #(11, 3) -> (1,0)
             df2.loc[df2.gen_diagnosis_num == 3,'gen_diagnosis_num'] = 0
             df2.loc[df2.gen_diagnosis_num == 11,'gen_diagnosis_num'] = 1
        elif 10 in gen_diagnoses_to_consider and 3 in gen_diagnoses_to_consider:  #(10, 3) -> (1,0)
             df2.loc[df2.gen_diagnosis_num == 3,'gen_diagnosis_num'] = 0
             df2.loc[df2.gen_diagnosis_num == 10,'gen_diagnosis_num'] = 1
        elif 7 in gen_diagnoses_to_consider and 3 in gen_diagnoses_to_consider: #(7, 3) -> (1,0)
            df2.loc[df2.gen_diagnosis_num == 3,'gen_diagnosis_num'] = 0
            df2.loc[df2.gen_diagnosis_num == 7,'gen_diagnosis_num'] = 1
    print(np.bincount(df2['gen_diagnosis_num']))
    return df2


def is_first_session_date_earlier(session1, session2):
    '''
    Utility for comparing dates of two session strings in IDIDX_YYYY_MM_DD format

    Parameters
    -----------
    session1: str
        represents a session in IDIDX_YYYY_MM_DD format
    session2: str
        represents a sesssion in IDIDX_YYYY_MM_DD format

    Returns
    ----------
    bool
        True if session1 happened before session2, False otherwise
    '''

    year1 = int(session1.split("_")[-3])
    month1 = int(session1.split("_")[-2])
    day1 = int(session1.split("_")[-1])
    year2 = int(session2[6:10])
    month2 = int(session2[11:13])
    day2 = int(session2[14:16])

    if year2 < year1:
        return False
    elif year2 == year1 and month2 < month1:
        return False
    elif year2 == year1 and month2 == month1 and day2 < day1:
        return False
    else:
        return True


def get_paths_to_files_with_sessions_after(filelist, month=1, day=1, year=2019):
    '''
    Takes list of paths to file and returns list of paths to files after a certain comparison date

    Parameters
    ----------
    filelist: array-like
        List of strings, each the path to a filename, with format '*/IDIDX_YYYY_MM_DD*'
    month: int
        Month (1-12) of comparison date
    day:
        Day (1-31) of comparison date
    year:
        Four-digit year of comparison date

    Returns
    ---------
    filtered_files: array_like
        List of strings, each the path to a filename in filelist, but only those whose date is after
        the comparison date provided by year, month, date

    '''
    filtered_files = []
    comparison_date = str(year) + "_" + str(month) + "_" + str(day)
    for filename in filelist:
        session = filename.rsplit("/")[-1][0:16] # extract session from filename
        if is_first_session_date_earlier(comparison_date, session):
            continue
        else:
            filtered_files.append(filename)
    return filtered_files


def load_time_freq_task_data_from_mat(session, sensor, task_name, filename=None):
    ''''
    Loads data from a matlab file for the wavelet transformed or sst-ed data 

    Parameters
    --------
    session : str
        Session string- e.g. in the format IDIDX_YYYY_MM_DD
    sensor : str
        Sensor location, e.g. 'LUE' or 'RLE' 
    task_name : str
    filename : str (optional)
        If included, this is the path of the file to be loaded. If none, file will be loaded from a file with path
        task_name + "_pc_wt/" + session + "_APDM" + "_" + sensor + "_" + task_name + "_pc_wt" + ".mat"

    
    Returns
    --------
    dict
        "f_wt" : frequency vector for wavelet tranform
        "f_sst" frequency vector for wavelet tranform
        "t" : vector of timestamps
        "wt": dict of wavelet transformed data
        "sst": dict of wavelet synchrosqueezing transformed data
        "names": keys for the wt and sst dictionary (like 'acc_gyro_x' or 'magno_pc_2')

    Notes
    --------
    The matlab .mat file is assumed to include variables named: f_sst, f_wt, ts_task, sst_transforms, and wt_transforms
    '''
    if filename is None:
        filename = (
            task_name
            + "_pc_wt/"
            + session
            + "_APDM"
            + "_"
            + sensor
            + "_"
            + task_name  
            + "_pc_wt" 
            + ".mat"
        )
    matlabdata = loadmat(filename)
    f_sst = pd.Series(matlabdata["f_sst"].flatten())
    f_wt = pd.Series(matlabdata["f_wt"].flatten())
    t = pd.Series(matlabdata["ts_task"].flatten())
    wt, sst = {}, {}
    names = [str(name) for name in matlabdata['sst_transforms'].dtype.fields.keys()] #names like 'acc_task_x' or 'acc_pc_1'
    for i in range(9):
        wt[names[i]] = matlabdata["wt_transforms"][0][0][i]
        sst[names[i]] = matlabdata["sst_transforms"][0][0][i]
    return {"f_wt": f_wt, "f_sst": f_sst, "t": t, "wt": wt, "sst": sst, "names": names}


def get_task_inds_simplistic(gyro_x, gyro_y, gyro_z):
    '''
    Get indices for the active portion of the task for this sensor, selecting either the first half
    temporally or the second half, depending on which one has more power in the gyroscope series.

    Parameters
    ----------
    gyro_x: Series
        1D array giving first channel of gyro data
    
    gyro_y: Series
        1D array giving first channel of gyro data
    
    gyro_z: Series
        1D array giving first channel of gyro data

    Returns:
    ----------
    inds_task: array-like
        List of indices estimated to correspond to the task portion
        (either the first half or the second half of the indices in the supplied gyro data)

    inds_task: array-like
        List of indices estimated to correspond to the rest portion
        (either the first half or the second half of the indices in the supplied gyro data)

    Notes
    ---------
    This is simplistic in that 1) it uses total power to distinguish between task and rest and
    2) It splits the list of indices exactly in half at the midpoint and categorizes one half
    as task and the other as rest
    '''

    num_timebins = gyro_x.shape[-1]
    t_halfway = math.floor(num_timebins / 2)
    total_power = abs(gyro_x) ** 2 + abs(gyro_y) ** 2 + abs(gyro_z) ** 2
    if total_power[t_halfway:].sum() > total_power[0:t_halfway].sum():
        inds_task = np.arange(t_halfway, num_timebins)
        inds_rest = np.arange(0, t_halfway)
        task_is_first = False # This variable included for quality control purposes - can return it to be compared to data visually
    else:
        inds_task = np.arange(0, t_halfway)
        inds_rest = np.arange(t_halfway, num_timebins)
        task_is_first = True

    if isinstance(gyro_x, pd.Series):
        if total_power.iloc[t_halfway:].sum() > total_power.iloc[0:t_halfway].sum():
            inds_task = np.arange(t_halfway, num_timebins)
            inds_rest = np.arange(0, t_halfway)
            task_is_first = False
        else:
            inds_task = np.arange(0, t_halfway)
            inds_rest = np.arange(t_halfway, num_timebins)
            task_is_first = True
        inds_rest = gyro_x.index[inds_rest]
        inds_task = gyro_x.index[inds_task]

    return inds_task, inds_rest


def create_basic_features_from_time_freq(
    file_list,
    task_name,
    sensor_type,
    sensor_locations,
    task_name_abbreviated=None, #if task name includes "pedi", this is a version without it
    high_v_low_freq_cutoff=2,
    avg_left_and_right=False,
    use_first_pcs=True,
    num_pcs=1,
    wt_or_sst="sst"
):
    """
    Creates a set of hand-designed fatures from the wavelet transformed or synchrosqueezed data

    file_list: array-like
        List of strings, each of which is the path to a .mat file containing wt and sst data (as output by
        sst and wt matlab script)
    task_name: str
        e.g. FNF2 or Heel_Shin
    sensor_type: str
        e.g. 'acc' or 'gyro'
    sensor_location: set
    Set of locations, each a string 'LUE' or 'RLE'
    task_name_abbreviated: str, optional
        If task name includes "pedi", this is a version without it - used to treat task pairs like FNF2 and FNF2_Pedi as the same task
    high_v_low_freq_cutoff: float, optional
        Default: 2.0.  The frequency, in Hz, that will be used to divide high frequency from low in features that are
        calculated based on separate high and low frequency values
    avg_left_and_right: bool, optional
        Default: False.  If true, features will be computed separately for LUE vs RUE, and LLE vs. RLE  If False, features from
        both sides will be computed by loading two corresponding .mat files, and the features returned by this function will
        all be averaged over rigth and left.
    use_first_pcs: bool, optional
        Default: False.  If true, will expect the .mat files from which the time-frequency data is loaded to have variables like
        gyro_pc_1, and  the computed features will have names like task_power_RUE_gyro_pc_1.  If False, the .mat file will be expected to have
        variables named things like gyro_task_x, and features will named like task_power_RUE_gyro_x.
    num_pcs: int, optional
        Default: 1. Number of principal components to use, if use_first_pcs is True.
    wt_or_sst: {'sst', 'wt'}. optional
        Default: 'sst'

    Returns
    --------
    df: DataFrame 
        A dataFrame with columns that are features, and indices that are sessions in IDIDX_YYYY_MM_DD function

    Notes
    ---------
    The function make_basic_features() creates the features; see its documentation for which features are computed
    """

    total_power_pcs = {}
    list_of_all_features = []
    sessions = []

    if task_name_abbreviated is None:
        task_name_abbreviated = task_name

    for filename in file_list:
        all_features_for_one_file = {}
        features = [] # will be a list of dictionaries (one for each sensor location if we are keeping left and right separate)
        print(filename)
        session = filename.rsplit("/", 1)[-1][0:16]
        sessions.append(session)
        ID = session[0:5]
        for sensor in sensor_locations:
            time_freq = load_time_freq_task_data_from_mat(session, sensor, task_name)

            # Get breakpoint between task and rest using gyro data
            if wt_or_sst == "sst":
                f = time_freq["f_sst"]
                is_f_ascending = True
            elif wt_or_sst == "wt":
                f = time_freq["f_wt"]
                is_f_ascending = False
            else:
                print("invalid parameter wt_or_sst, must be one of 'sst' or 'wt'")

            # Calculate task and rest indices using gyroscope power
            if use_first_pcs:
                x = time_freq[wt_or_sst]["gyro_pc_1"].sum(axis=0)
                y = time_freq[wt_or_sst]["gyro_pc_2"].sum(axis=0)
                z = time_freq[wt_or_sst]["gyro_pc_3"].sum(axis=0)
            else:
                x = time_freq[wt_or_sst]["gyro_task_x"].sum(axis=0)
                y = time_freq[wt_or_sst]["gyro_task_y"].sum(axis=0)
                z = time_freq[wt_or_sst]["gyro_task_z"].sum(axis=0)
            inds_task, inds_rest = get_task_inds_simplistic(x, y, z) 
            
            if use_first_pcs:
                features = []
                features_pcs = {}
                #pc_component_names = ["_task_x", "_task_y", "_task_z"] 
                pc_component_names = ["_pc_1", "_pc_2", "_pc_3"]
                for i in range(num_pcs):
                    if avg_left_and_right:
                        feature_name_suffix = "_pc" + str(i+1) + "_" + task_name_abbreviated #we start numbering pcs with 1
                    else:
                        feature_name_suffix = "_" + sensor + "_pc" + str(i+1) + "_" + task_name_abbreviated 
                    total_power_pcs[i] = time_freq[wt_or_sst][
                        sensor_type + pc_component_names[i] # Because wt and sst structs were saved from matlab with fiedes like "acc_pc_1" or "magno_pc_3"
                    ]
                    # features[sensor + "_pc" + str(i)] = make_basic_features(
                    features_pcs.update(make_basic_features(
                        total_power_pcs[i],
                        inds_task,
                        inds_rest,
                        f,
                        high_v_low_freq_cutoff,
                        is_f_ascending, 
                        feature_name_suffix = feature_name_suffix
                    ))
                features.append(features_pcs)

            else:
                x = time_freq[wt_or_sst][sensor_type + "_task_x"]
                y = time_freq[wt_or_sst][sensor_type + "_task_y"]
                z = time_freq[wt_or_sst][sensor_type + "_task_z"]
                total_power = abs(x) ** 2 + abs(y) ** 2 + abs(z) ** 2
                
                features.append(make_basic_features(
                    total_power,
                    inds_task,
                    inds_rest,
                    f,
                    high_v_low_freq_cutoff,
                    is_f_ascending,
                    feature_name_suffix=  "_" + sensor + "_" + task_name_abbreviated
                ))
                
            if avg_left_and_right:
                for key in features[0].keys():
                    new_key = key.replace('_' + sensor_locations[0] + '_', '_')
                    new_key = new_key.replace('_' + sensor_locations[0], '')
                    all_features_for_one_file[new_key] = (features[0][key] + features[0][key])/2
            else:
                all_features_for_one_file.update(features[0])
           
            all_features_for_one_file['ID'] = ID
            all_features_for_one_file['session'] = session       

        list_of_all_features.append(all_features_for_one_file)  
              
    df = pd.DataFrame.from_records(list_of_all_features, index = sessions)
    df.drop(['session'], axis=1, inplace=True)
    return df


def get_average_cosine_similarity(total_power, inds_task):
    '''
    Calculates the average cosine similarity of the vector of frequency powers between adjacent time-bins 

    Parameters
    ----------
    total_power: ndarray
        A 2D array of power values swith dimensions given by frequency x time
    inds_task: ndarray
        A vector of integer indices used to identify which bins of total_power correspond to the
        'task' (i.e in FNF for the RUE sensor, this would be the portion of time where of the right hand is performing the
        task, and the left is at rest)

    Returns
    ----------
    float
        The mean of the cosine similarities between each adjacent bin of the task portion
        of total_power, where what constitutes the task portion is given by inds_task

    '''
    task_power = total_power[:, inds_task]
    cosine_sims = (task_power[:,0:-2] * task_power[:,1:-1]).sum(axis=0) / (np.linalg.norm(task_power[:,0:-2], axis=0) *  np.linalg.norm(task_power[:,1:-1], axis=0))
    return cosine_sims.mean()


def make_basic_features(
    total_power, inds_task, inds_rest, f, high_v_low_freq_cutoff, is_f_ascending, feature_name_suffix=""
):
    '''
    Creates the basic features from one channel of time-frequency data (e.g. from acc_pc_1 or gyro_z)

    Parameters:
    total_power: ndarray
        A 2D array of power with dimensions given by frequency x time
    inds_task: ndarray
        A vector of integer indices used to identify which bins of total_power correspond to the
        'task' (i.e in FNF for the RUE sensor, this would be the portion of time where of the right hand is performing the
        task, and the left is at rest)
    inds_rest
        A vector of integer indices used to identify which bins of total_power correspond to the
        rest (i.e in FNF for the RUE sensor, this would be the portion of time where of the left hand is performing the
        task, and the right is at rest)
    f: array-like
        Vector of frequencies, each one corresponding to a row of total_power
    high_v_low_freq_cutoff: float
        The frequency, in Hz, that will be used to divide high frequency from low in features that are
        calculated based on separate high and low frequency values
    is_f_ascending: bool
    If True, the frequencies in total_power are assumed to be in ascending order.  Otherwise, assumed to be in decending order
    feature_name_suffix: str, optional
        The suffix to be added to each of the basic feature names to indicate what channel it came from.  E.g. "_LUE_gyro_x" would
        produce features like 'task_power_LUE_gyro_x'

    Returns
    --------
    features: dict
        Keys are feature numes (strings), and values are feature values (float).  Features are:
        task_power, rest_power,task_to_rest_power_ratio,task_low_freq_power, rest_low_freq_power,
        task_high_freq_power,rest_high_freq_power, task_low_to_high_freq_power_ratio, rest_low_to_high_freq_power_ratio,
        task_center_freq, rest_center_freq, task_spread_freq, rest_spread_freq, task_center_freq_of_low_freq,
        task_center_freq_of_high_freq, task_cosine_sim_adjacent_timebins
    '''

    # Get power during task and at rest
    task_power = (total_power[:, inds_task]).mean().sum()
    rest_power = (total_power[:, inds_rest]).mean().sum()

    # Get ratio of power at rest and during task
    task_to_rest_power_ratio = task_power / rest_power

    # Get amount of high and low frequency power
    ind_high_v_low_freq_cutoff = np.abs(f - high_v_low_freq_cutoff).idxmin()

    if is_f_ascending:
        task_low_freq_power = total_power[
            0:ind_high_v_low_freq_cutoff, inds_task
        ].mean()
        rest_low_freq_power = total_power[
            0:ind_high_v_low_freq_cutoff, inds_rest
        ].mean()

        task_high_freq_power = total_power[
            ind_high_v_low_freq_cutoff:, inds_task
        ].mean()
        rest_high_freq_power = total_power[
            ind_high_v_low_freq_cutoff:, inds_rest
        ].mean()

    else:
        task_high_freq_power = total_power[
            0:ind_high_v_low_freq_cutoff, inds_task
        ].mean()
        rest_high_freq_power = total_power[
            0:ind_high_v_low_freq_cutoff, inds_rest
        ].mean()

        task_low_freq_power = total_power[ind_high_v_low_freq_cutoff:, inds_task].mean()
        rest_low_freq_power = total_power[ind_high_v_low_freq_cutoff:, inds_rest].mean()

    task_low_to_high_freq_power_ratio = task_low_freq_power / task_high_freq_power
    rest_low_to_high_freq_power_ratio = rest_low_freq_power / rest_high_freq_power

    # Get mean mean frequency at rest and during task
    task_center_freq = (total_power[:, inds_task].sum(axis=1) * f).sum() / (
        (total_power[:, inds_task]).sum(axis=1).sum()
    )
    rest_center_freq = (total_power[:, inds_rest].sum(axis=1) * f).sum() / (
        (total_power[:, inds_rest]).sum(axis=1).sum()
    )

    # Center of low freq
    if is_f_ascending:
        task_center_freq_of_low_freq = (
            total_power[:ind_high_v_low_freq_cutoff, inds_task].sum(axis=1)
            * f[:ind_high_v_low_freq_cutoff]
        ).sum() / (
            (total_power[:ind_high_v_low_freq_cutoff, inds_task]).sum(axis=1).sum()
        )
        task_center_freq_of_high_freq = (
            total_power[ind_high_v_low_freq_cutoff:, inds_task].sum(axis=1)
            * f[ind_high_v_low_freq_cutoff:]
        ).sum() / (
            (total_power[ind_high_v_low_freq_cutoff:, inds_task]).sum(axis=1).sum()
        )
    else:
        task_center_freq_of_low_freq = (
            total_power[ind_high_v_low_freq_cutoff:, inds_task].sum(axis=1)
            * f[ind_high_v_low_freq_cutoff:]
        ).sum() / (
            (total_power[ind_high_v_low_freq_cutoff:, inds_task]).sum(axis=1).sum()
        )
        task_center_freq_of_high_freq = (
            total_power[:ind_high_v_low_freq_cutoff, inds_task].sum(axis=1)
            * f[:ind_high_v_low_freq_cutoff]
        ).sum() / (
            (total_power[:ind_high_v_low_freq_cutoff, inds_task]).sum(axis=1).sum()
        )

    # Get measure of spread of frequency during task and at rest
    task_spread_freq = (
        total_power[:, inds_task].sum(axis=1) * (f - task_center_freq) ** 2
    ).sum() / ((total_power[:, inds_task]).sum(axis=1).sum())
    rest_spread_freq = (
        total_power[:, inds_rest].sum(axis=1) * (f - rest_center_freq) ** 2
    ).sum() / ((total_power[:, inds_rest]).sum(axis=1).sum())

    features = {
        "task_power": task_power,
        "rest_power": rest_power,
        "task_to_rest_power_ratio": task_to_rest_power_ratio,
        "task_low_freq_power": task_low_freq_power,
        "rest_low_freq_power": rest_low_freq_power,
        "task_high_freq_power": task_high_freq_power,
        "rest_high_freq_power": rest_high_freq_power,
        "task_low_to_high_freq_power_ratio": task_low_to_high_freq_power_ratio,
        "rest_low_to_high_freq_power_ratio": rest_low_to_high_freq_power_ratio,
        "task_center_freq": task_center_freq,
        "rest_center_freq": rest_center_freq,
        "task_spread_freq": task_spread_freq,
        "rest_spread_freq": rest_spread_freq,
        "task_center_freq_of_low_freq": task_center_freq_of_low_freq,
        "task_center_freq_of_high_freq": task_center_freq_of_high_freq,
        "task_cosine_sim_adjacent_timebins": get_average_cosine_similarity(total_power, inds_task)
    }

    features = {str(key) + feature_name_suffix : value for key, value in features.items()}

    return features
