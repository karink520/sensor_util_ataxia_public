import glob
from .load_sensor_data_utils import create_basic_features_from_time_freq

if __name__== '__main__':
    # SET THESE BASED ON TASK
    # Can make filelist based on files in folder, or manually set to a list of strings with filenames, e.g. if you only want to compute features for a few files
    file_list = glob.glob('Light_Bulb_pc_wt/*RUE_Light_Bulb_pc_wt.mat') # we'll use RUE to set the filelist, but will create features for RUE and LUE
    task_name = 'Light_Bulb'
    out_acc_filename = 'time_freq_features/sst_features_LUE_RUE_acc_light_bulb_pc.csv'
    out_gyro_filename = 'time_freq_features/sst_featuers_LUE_RUE_gyro_light_bulb_pc.csv'
    sensor_lcations = ('RUE','LUE') # use ('RUE','LUE') for upper extremity tasks (lightbulb, fnf, mirroring), # use ('RLE','LLE') for lower extremity tasks (lightbulb, fnf, mirroring)
    high_v_low_freq_cutoff = 3 # Cutoff in Hz for what gets counted as "high" vs. "low" frequency in feature construction - used 3 for lightbulb, 2 for others

    # GENERALLY ASSUMED TO BE CONSTANT ACROSS TASK
    avg_LUE_and_RUE = False # don't average features from the right and left side
    num_pcs = 1 # How many principal components to use for feature extraction
    wt_or_sst = 'sst' # Synchrosqeezing ('sst') or wavelet transform ('wt')

    # CREATE FEATURES AND OUTPUT TO CSVS - one for acc and one for gyro
    df_acc_pc = create_basic_features_from_time_freq(file_list, task_name, sensor_type='acc', sensor_locations=sensor_locations, high_v_low_freq_cutoff=high_v_low_freq_cutoff, avg_left_and_right=avg_LUE_and_RUE, use_first_pcs=True, num_pcs=num_pcs, wt_or_sst=wt_or_sst)
    df_acc_pc.to_csv(out_acc_filename)
    df_gyro_pc = create_basic_features_from_time_freq(file_list, task_name, sensor_type='gyro', sensor_locations=sensor_locations, high_v_low_freq_cutoff=high_v_low_freq_cutoff, avg_left_and_right=avg_LUE_and_RUE, use_first_pcs=True, num_pcs=num_pcs, wt_or_sst=wt_or_sst)
    df_gyro_pc.to_csv(out_gyro_filename)