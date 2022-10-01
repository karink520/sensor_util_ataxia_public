This repository shares code resources for fitting an AR-HMM model to wearable sensor IMU data, extracting features from the fitted AR-HMM model, and calculating an alternative set of features using a synchrosqueezing-based time frequency approach. It also includes utilities for visual quality control, preprocessing, plotting, and fitting classification and regression models for disease severity using the derived features.

The general pipeline was applied to our data is as follows:

# 0. Visual QC on the sensor files
The function `MATLAB/IMU_visual_quality_control.m` allows the user to iterate through plots of session data, for visual quality control purposes.
Run this interactively, entering notes as necessary.

For interactively plotting the raw data from a specific IMU in Python, the utility function:
`sensor_plotting_data.plot_IMUtask_sensor_data` is a useful helper.

*Note:* To load raw IMU data from .mat files into Python for downstream processing, `load_sensor_data_utils.load_IMUtask_data_from_mat` is a helpful utility function.  

# 1. Time-frequency feature extraction

## Map sensor accelerometer and gyroscope data to time-frequency domain with synchrosqueezing transform and save new files (includes preprcessing by projecting 3-dimensional signal onto first principal component).

Run the script `MATLAB/sst_and_wt_sensor_data_export.m` for each chosen task (update directories to point to data for the desired task and sensors). Transformed (synchrosqueezed) task data will be saved to `{parent_dir_name}/{task}_pc_wt/` (or`{parent_dir_name}/{task}_wt/` if not preprocessing with PCA). 

## Construct time-frequency features from time-frequency representation
Run `run_create_time_freq_features.py` for each desired task. This script will load the `.mat` files corresponding to the synchrosqueezed transform using and create features from the data with the function
`load_sensor_data_utils.create_basic_features_from_time_freq`, then save features for each session/sensor-modality/sensor-location in a csv within `time_frequency_features/` directory. Script that does this:

*Note:* To load time-frequency .mat files (output by `MATLAB/sst_and_wt_sensor_data_export.m`) into a pandas dataframe for alternative downstream processing, `load_sensor_data_utils.load_time_freq_task_data_from_mat` is a helpful utility function.

## Merge time-frequency features from different tasks and combine with subject data
Merge the csvs corresponding to accel. and gyro. for each task that were created in the `time_frequency_features/` directory in the previous step, using e.g. pandas merge functionality.
Further combine the result with subject data about severity scores diagnosis etc. and save the result as a csv.

# 2. AR-HMM feature extraction

## Preprocessing for AR-HMM
Use `load_sensor_data_utils.load_IMUtask_data_from_mat` to load `.mat` IMU data into a pandas DataDrame, followed by `preprocessingfor_ar_hmm/preprocess_for_ar_hmm`, which  includes a wavelet denoising, a dimensionality reduction with PCA, and a downsampling step.

## Use a subset of the data to learn autoregressive parameters in AR-HMM
Fit an AR-HMM by Gibbs sampling with `hdp_hmm_gibbs_sampling`. Because it may be computationally intensive to learn the autoregressive parameters by fitting the entire dataset, we use just part of the data to find autoregressive parameters, and then fix the autoregressive parameters to the learned posterior mean for the next step where we inferring the states for each time point and other model parameters for each sensor recording. Tools for inspecting sample trajectories associated with the learned parameters are in `generative_model.py`.

## Use learned AR parameters to estimate state-sequences and extract state-space related features
Fixing the autoregressive parameters, we can learn state sequences and parameters for each sensor with `find_states_ar_hmm.find_state_related_features_arr_hmm`.

# 3. Disease classification and severity prediction
Utilities for balanced random forest classification and random forest regression are in `classification_regression_utilities.py`.


