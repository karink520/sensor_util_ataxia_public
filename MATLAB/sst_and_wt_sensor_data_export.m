% Calculate and save wavelet-synchrosqueezed power of the principal components for a series of sessions, given
% signal acc_task and timestamps ts_task

Fs = 128;
top_freq = 15;
project_to_pcs = true; % whether to transform pcs (true) or original signal (false)

task = ''; % Task name
task_path = ''; % Path to directory that contains sessions;
parent_dir_name = '.'; % Transformed task data will be saved to {parent_dir_name}/{task}_pc_wt/ or if we're not projecting onto pc: {parent_dir_name}/{task}_wt/
files = dir(''); % Regex that captures the recordings within the task folder for the relevant extremity (UE or LE)

%% OPTIONAL SECTION, DEPENDING ON TASK
% If the task is divided into task and task_Pedi (e.g. FNF), this portion
% of the script will
% create a list of paths to the relevant files (`filenames`) by
% iterating through sessions, and find left and right recordings for the
% given task and the left and right upper extremity,
% checking both the task and task_Pedi directories
% You can also use it to create a list of paths to files whose sessions are
% in a cell_array new_IMU_sessions = {'IDIDX_YYYY_MM_DD',
% 'IDIDY_YYYY_MM_DD'}, a useful alternative if you just want to process a
% few files (e.g. new sessions)

add_sessions_by_session_name = false;
if add_sessions_by_session_name
    new_IMU_sessions = {}; %CHANGE ME

    for i=1:2:2*length(new_IMU_sessions)
        file_missing = false;
        current_session = new_IMU_sessions{floor((i+1)/2)};
        file_path_R = strcat(task_path,'/', current_session, '_APDM_RUE_',task,'.mat');
        disp(file_path_R)
        file_path_R_pedi = strcat(task_path,'_Pedi','/',current_session,'_APDM_RUE_', task ,'_Pedi.mat');
        disp(file_path_R_pedi)
        if isfile(file_path_R)
            file_path_L = strcat(task_path,'/',current_session, '_APDM_LUE_',task,'.mat');
            task_directory_name = task; % usually the same as task, but might be task_pedi
        elseif isfile(file_path_R_pedi)
            file_path_R = file_path_R_pedi;
            file_path_L = strcat(task_path,'_Pedi','/',current_session,'_APDM_LUE_', task ,'_Pedi.mat');
            task_directory_name = strcat(task, '_Pedi');
        else
            file_missing = true;
            %missing_list(end) = file_path_R;
            disp("file not found")
        end
        if not(file_missing)
            files(i).name = file_path_R;
            files(i).task = task_directory_name;
            files(i+1).name = file_path_L;
            files(i+1).task = task_directory_name;
        end
    end
end

%% Iterate through the selected session filenames in `files` variable, load each one, project to the first 
% princial component(s), calculate the sst and wavelet transform and save.

file_num_start = 1; % can change this and next line if don't want to iterate through ALL files
file_num_end = length(files); %

for i=file_num_start:file_num_end
  i % print for progress tracking purposes
  filename = strcat(task_path,files(i).name);
  load(filename);
  signals = {acc_task, gyro_task, magno_task};
  
  if project_to_pcs
       names = {'acc_pc_1', 'acc_pc_2', 'acc_pc_3', 'gyro_pc_1', 'gyro_pc_2', 'gyro_pc_3', 'magno_pc_1', 'magno_pc_2', 'magno_pc_3'};
       for j = 1:3 % iterate through acc, gyro, and magno
           signal = signals{j};
           [coeff, score, latent] = pca(signal);
           for k=1:3 %iterate through pcs
                x = score(:,k);
                [wt,f_wt] = cwt(x,'bump',Fs);
                [sst, f_sst] = wsst(x,Fs,'bump');
                wt = wt(f_wt<top_freq,:);
                f_wt = f_wt(f_wt < top_freq);
                sst = sst(f_sst<top_freq,:);
                f_sst = f_sst(f_sst < top_freq);
                wt_transforms.(names{(j-1)*3 + k}) = abs(wt);
                sst_transforms.(names{(j-1)*3 + k}) = abs(sst);
           end
       end
       tf_filename = extractBefore(filename,'.mat');
       tf_filename = extractAfter(tf_filename,'/');
       tf_filename = string(strcat(parent_dir_name,'/', task,'_pc_wt/', tf_filename, "_pc_wt.mat")); % save the wavelet transformed data here
       tf_filename = erase(tf_filename, 'Pedi_') % Don't want to include Pedi in the filename
       save(tf_filename, 'wt_transforms', 'f_wt', 'sst_transforms', 'f_sst', 'ts_task');
       %save(tf_filename, 'sst_transforms', 'f_sst', 'ts_task'); %Can use
       %this if want to only save ssts (not also the wts)

      
  else
      names = {'acc_task_x', 'acc_task_y', 'acc_task_z', 'gyro_task_x', 'gyro_task_y', 'gyro_task_z', 'magno_task_x', 'magno_task_y', 'magno_task_z'};
      for j = 1:9
        x = signals{ceil(j / 3)}(:, mod(j,3)+1);
        [wt,f_wt] = cwt(x,'bump',Fs);
        [sst, f_sst] = wsst(x,Fs,'bump');

        wt = wt(f_wt<top_freq,:);
        f_wt = f_wt(f_wt < top_freq);
        sst = sst(f_sst<top_freq,:);
        f_sst = f_sst(f_sst < top_freq);
        wt_transforms.(names{j}) = abs(wt);
        sst_transforms.(names{j}) = abs(sst);

      end  
      
      tf_filename = strcat(extractBefore(filename,'.mat'),'_wt.mat');
      tf_filename = extractAfter(tf_filename,'/');
      tf_filename = extractAfter(tf_filename,'/');
      tf_filename = string(strcat(parent_dir_name, task,'_wt/', tf_filename)); % save the wavelet transformed data here
      tf_filename = erase(tf_filename,'Pedi_')
      save(tf_filename, 'wt_transforms', 'f_wt', 'sst_transforms', 'f_sst', 'ts_task');
      %save(tf_filename, 'sst_transforms', 'f_sst', 'ts_task'); %Can use
      %this if want to only save ssts (not also the wts)
  end
end
