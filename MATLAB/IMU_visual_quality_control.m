%% Display some sample plots for each session to get a visual check on
% a) time alignment b) whether sensors are positioned correctly on body
% To use, run step 1, then repeatedly alternate between step 2 (showing 
% plots) and step 3 (entering notes)

% Assumes the relevant figures for each task are in IMU_All/{task}/Figures/
%% STEP 1: set sessions to iterate through
IMU_sessions = {}; % SET THIS List of sessions strings (IDIDX_YYYY_MM__DD)

% or, get sessions from ID column of subject_data
path_to_subect_data = ''; %SET THIS IF DESIRED
t = readtable(path_to_subject_data);
IMU_sessions = t.ID;

session_index = 0;
notes = {}; % empty cell array, to be filled in with notes about observed session quality

%% STEP 2: View figures
close all % close figures

%advance to next session
session_index = 1 %CHANGE THIS AT EACH ITERATION
current_session = string(IMU_sessions(session_index))

% Show upper extremity plots for the mirroring task
task = 'Mirroring3';
task_path = strcat('IMU_All/',task,'/Figures/');
fig_path_R = task_path + current_session + '_APDM_RUE_' + task + '.fig';
fig_path_L = task_path + current_session + '_APDM_LUE_' + task + '.fig';
f_mirroringR = openfig(fig_path_R);
f_mirroringL = openfig(fig_path_L);

% Use mirroring2 to check for pedi vs. adult
task = 'Mirroring2';
task_path = strcat('IMU_All/',task,'/Figures/');
mirroring2_fig_path_R = task_path + current_session + '_APDM_RUE_' + task + '.fig';
if isfile(fig_path_R) | isfile(mirroring2_fig_path_R) %check if the image file exists at this location before loading
    is_pedi = false;
else   %look for the file in the _pedi task folder
    is_pedi = true;
end

% Show upper extremity plots for the FNF task
if is_pedi
    task = 'FNF2_Pedi';
else
    task = 'FNF2';
end

task_path = strcat('IMU_All/',task,'/Figures/');
fig_path_R = task_path + current_session + '_APDM_RUE_' + task + '.fig';
fig_path_L = task_path + current_session + '_APDM_LUE_' + task + '.fig';
f3 = openfig(fig_path_R);
f4 = openfig(fig_path_L);


%Show upper extremity plots for Lightbulb task
task = 'Light_Bulb';
task_path = 'IMU_All/Light_Bulb/Figures/';
fig_path_R = task_path + current_session + '_APDM_RUE_' + task + '.fig';
fig_path_L = task_path + current_session + '_APDM_LUE_' + task + '.fig';
f1 = openfig(fig_path_R);
f2 = openfig(fig_path_L);


% Show lower extremity plots for the heel-shin task
task = 'Heel_Shin2';
task_path = 'IMU_All/Heel_Shin2/Figures/';
fig_path_R = task_path + current_session + '_APDM_RLE_' + task + '.fig';
fig_path_L = task_path + current_session + '_APDM_LLE_' + task + '.fig';
f5 = openfig(fig_path_R);
f6 = openfig(fig_path_L);

% Spread the plots out on the screen
figs = [f1, f2, f3, f4, f5, f6, f_mirroringR, f_mirroringL];   %as many as needed
nfig = length(figs);
frac = 1/nfig;
for K = 1 : nfig
  old_pos = get(figs(K), 'Position');
  width = 0.25 * old_pos(3);
  set(figs(K), 'Position', [(K-1)*width, old_pos(2), width, old_pos(4)]);
end

%% STEP 3: Enter notes after viewing each plot

%edit after viewing each set of figures
Light_Bulb_notes = 'Light_Bulb notes here';
FNF_notes = 'FNF2 notes here'; 
Heel_Shin_notes ='Heel_Shin2 notes here';
notes{session_index, 1} = char(current_session);
notes{session_index, 2} = FNF_notes;
notes{session_index, 3} = Light_Bulb_notes;
notes{session_index, 4} = Heel_Shin_notes;

% print notes if desired
notes

%% STEP 4: Save notes
% EDIT DATE or name of output file as desireds
% save('IMU_QC_DATE.mat', 'notes')
