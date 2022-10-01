
import scipy
import numpy as np
from .rotation_invariance import project_onto_global_pc #, project_onto_sliding_pc
from .filters import wavelet_denoising
from .load_sensor_data_utils import get_task_inds_simplistic


def preprocess_for_ar_hmm(df, start=100, stop=5000):
    '''
    Preprocesses gyroscope and acceleration data (using fixed PCA window & wavelet denosing),
    followed by downsampling
    
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
    For acceleration, additionally subtract off the signal mean.
    '''
    df.reset_index(inplace=True)
    task_inds, rest_inds = get_task_inds_simplistic(df['gyro_task_x'], df['gyro_task_y'], df['gyro_task_z'])
    acc_means = df[ ['acc_task_x', 'acc_task_y', 'acc_task_z']].iloc[rest_inds].mean(axis=0)
    
    projected_gyro_temp = project_onto_global_pc(df['gyro_task_x'], df['gyro_task_y'], df['gyro_task_z'], num_pcs=1)
    projected_gyro_temp = np.expand_dims(wavelet_denoising(projected_gyro_temp),axis=1) 
    projected_gyro_temp = projected_gyro_temp[task_inds,:]
    projected_gyro_temp = projected_gyro_temp[start:stop,:]
    
    projected_acc_temp = project_onto_global_pc(df['acc_task_x'], df['acc_task_y'], df['acc_task_z'], num_pcs=1)
    projected_acc_temp = np.expand_dims(wavelet_denoising(projected_acc_temp),axis=1)
    projected_acc_temp = projected_acc_temp[task_inds,:]
    projected_acc_temp = projected_acc_temp[start:stop,:]

    preprocessed_gyro = scipy.signal.resample(projected_gyro_temp, int((stop-start)/10) )
    preprocessed_acc  = scipy.signal.resample(projected_acc_temp,  int((stop-start)/10) )
    print(preprocessed_acc.mean())
    preprocessed_acc -= preprocessed_acc.mean()

    return preprocessed_gyro, preprocessed_acc

