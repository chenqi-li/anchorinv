# Derived from https://github.com/zqiao11/TSCIL/blob/main/data/grabmyo.py

import scipy.io as scio
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from tqdm import tqdm
import re
from collections import defaultdict
import time
np.random.seed(3)

# Perform Z-standardization using training population scalar mean and std
# matdata_dir is .mat data converted using the provided script physionet.org/files/grabmyo/1.1.0/grabmyo_convert_wfdb_to_mat.py
# Running this file: (nhie) chenqi@CTG-Workstation:~/Desktop/eeg-conformer-chenqi$ 
# python grabmdataset_preprocessing_traintest_psz.py > "/data/quee4692/GRABMdataset/GRABMdataset_filtered_256hz_5s_psz_tt_dataloader/dataset_summary.txt"

# 17 classes, 16 targeting motions and 1 for rest. We only use the 16 classes with motions.
# each recording is 5 seconds in duration, sampled at 2048 hz. So each has 10240 time steps. We downsample to 256Hz.
# each activity performs 7 trials. We use all the trials.
# N_samples per class: 43 * 7 * n_windows. We do not split into windows, so n_windows = 1.


#--------------
matdata_dir = "/data/quee4692/GRABMdataset/physionet.org/files/grabmyo/1.1.0/Output BM"
preprocessed_dir = "/data/quee4692/GRABMdataset/GRABMdataset_filtered_256hz_5s"
dataloader_dir = "/data/quee4692/GRABMdataset/GRABMdataset_filtered_256hz_5s_psz_tt_dataloader"

preprocessing = True # load from matdata_dir and save to preprocessed_dir
splitting = True # load from preprocessed_dir and save to dataloader_dir
resample_length = 256 * 5  # downsample to 256 hz
input_channels_grabmyo = 28
N_subjects = 43
N_classes = 16
N_trials = 7
N_sessions = 3
#--------------

# Make directory for saving the files in npy format
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)
if not os.path.exists(dataloader_dir):
    os.makedirs(dataloader_dir)

# Preprocess and save to preprocessed directory
if preprocessing:
    print("Preprocessing to deal with NaN and downsample...")
    for session in range(1, N_sessions+1): # loop through all the sessions
        for subject in range(1, N_subjects+1): # loop through all the subjects
            # Load the mat file
            filepath = os.path.join(matdata_dir, f'Session{session}_converted', f'session{session}_participant{subject}.mat')
            mat = scio.loadmat(filepath)
            collections_forearm = mat['DATA_FOREARM']
            collections_wrist = mat['DATA_WRIST']
            for trial in range(0, N_trials): # loop through all the trials
                for clas in range(0, N_classes):  # loop through 16 gestures, and ignore the last (17th) gesture which is rest
                    # Concatenate wrist and forearm
                    record = np.concatenate((collections_forearm[trial][clas],collections_wrist[trial][clas]), axis=1)
                    record = np.nan_to_num(record)
                    record = resample(record, resample_length)
                    output_file = f'session{session}_participant{str(subject).zfill(2)}_gesture{str(clas).zfill(2)}_trial{trial}.npy'
                    np.save(os.path.join(preprocessed_dir, output_file), record)
    print("Preprocessing completed.")

# Load training file (session 1 and session 2) to compute mean and standard deviation 
if splitting:
    train_data = [] #np.empty([0,input_channels_grabmyo])
    for file in tqdm(os.listdir(os.path.join(preprocessed_dir)), desc='Gathering all training data for computing mean and std'):
        if file[0:8] in ['session1', 'session2']:
            data = np.load(os.path.join(preprocessed_dir,file)) # [1280, 28]
            train_data.append(data)
    train_data = np.concatenate(train_data, axis=0)
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    print(f'Training Data size: {train_data.shape}')
    print(f'Scalar Mean of Training Data: {train_mean}')
    print(f'Scalar StD of Training Data: {train_std}')

    # Normalize using training statistics and save file in npy format
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)
    for file in tqdm(os.listdir(preprocessed_dir), desc='Saving segments'):
        # Load the data and normalize it
        data = np.load(os.path.join(preprocessed_dir,file))
        data = (data-train_mean)/train_std
        
        # Save the data
        gesture = str(int(file[30:32])) # get the gesture class
        filepath = os.path.join(dataloader_dir, file)
        if file[0:8] in ['session1', 'session2']:
            train_dict[gesture].append(file)
        elif file[0:8] in ['session3']:
            test_dict[gesture].append(file)
        np.save(filepath, data.squeeze()) # [1280, 28]
    train_dict = dict(train_dict)
    test_dict = dict(test_dict)
    time.sleep(1)
    np.savez(os.path.join(dataloader_dir, 'train_set.npz'), **train_dict)
    np.savez(os.path.join(dataloader_dir, 'test_set.npz'), **test_dict)

    print(f'Training set: {train_dict}')
    print(f'Test set: {test_dict}')