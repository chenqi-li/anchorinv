import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.signal import butter, lfilter, freqz
from tqdm import tqdm
import pickle
import shutil
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict
import time
np.random.seed(3)

# Perform Z-standardization using training population scalar mean and std
# standard_2a_data is preprocessed using MATLAB with getData.m from EEG Conformer
# Running this file: (nhie) chenqi@CTG-Workstation:~/Desktop/eeg-conformer-chenqi$ 
# python bcidataset_preprocessing_traintest_psz.py > "/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_psz_subject_tt_dataloader/dataset_summary.txt"
# python bcidataset_preprocessing_traintest_psz.py > "/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_psz_session_tt_dataloader/dataset_summary.txt"

#--------------
preprocessed_dir = "/data/quee4692/BCIdataset/standard_2a_data"
split = 'session'
dataloader_dir = "/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_psz_session_tt_dataloader"
#--------------

# Make directory for saving the files in npy format
if not os.path.exists(dataloader_dir):
    os.makedirs(dataloader_dir)

# Load training file to compute mean and standard deviation
train_data = np.empty([0,22])
for file in tqdm(os.listdir(preprocessed_dir), desc='Gathering all training data for computing mean and std'):
    if split == 'subject':
        if int(file[1:3]) <= 5: #if file[-5:] == 'T.mat':
            loaded = scipy.io.loadmat(os.path.join(preprocessed_dir,file)) 
            eeg_data = loaded['data'] # [1000, 22, 288]
            for i in range(eeg_data.shape[-1]):
                train_data = np.vstack([train_data, eeg_data[:,:,i]])
    elif split == 'session':
        if file[-5:] == 'T.mat':
            loaded = scipy.io.loadmat(os.path.join(preprocessed_dir,file)) 
            eeg_data = loaded['data'] # [1000, 22, 288]
            for i in range(eeg_data.shape[-1]):
                train_data = np.vstack([train_data, eeg_data[:,:,i]])
train_mean = np.mean(train_data)
train_std = np.std(train_data)

print(f'Training Data size: {train_data.shape}')
print(f'Scalar Mean of Training Data: {train_mean}')
print(f'Scalar StD of Training Data: {train_std}')


# Save file in npy format
train_dict = defaultdict(list)
test_dict = defaultdict(list)
for file in tqdm(os.listdir(preprocessed_dir), desc='Saving segments'):
    loaded = scipy.io.loadmat(os.path.join(preprocessed_dir,file)) 
    eeg_data = loaded['data'] # [1000, 22, 288]
    eeg_label = loaded['label'] # [288, 1]
    # plt.plot(data)
    # plt.show()
    # print('Before', eeg_data[0, 0, 0], eeg_data[0, 1, 0], eeg_data[0, 2, 0])
    eeg_data = (eeg_data-train_mean)/train_std
    # print('After', eeg_data[0, 0, 0], eeg_data[0, 1, 0], eeg_data[0, 2, 0])

    # plt.plot(data)
    # plt.show()
    for i in range(eeg_data.shape[-1]):
        filename = f'{file[:-4]}_{str(i).zfill(3)}'
        filepath = os.path.join(dataloader_dir, filename)
        if split == 'subject':
            if int(file[1:3]) <= 5: #if file[3] == 'T':
                train_dict[str(int(eeg_label[i]))].append(filename+'.npy')
            elif int(file[1:3]) > 5:
                test_dict[str(int(eeg_label[i]))].append(filename+'.npy')
        elif split == 'session':
            if file[3] == 'T':
                train_dict[str(int(eeg_label[i]))].append(filename+'.npy')
            elif file[3] == 'E':
                test_dict[str(int(eeg_label[i]))].append(filename+'.npy')
        np.save(filepath, eeg_data[:,:,i].squeeze()) #[1000, 22]
train_dict = dict(train_dict)
test_dict = dict(test_dict)
time.sleep(1)
np.savez(os.path.join(dataloader_dir, 'train_set.npz'), **train_dict)
np.savez(os.path.join(dataloader_dir, 'test_set.npz'), **test_dict)

print(f'Training set: {train_dict}')
print(f'Test set: {test_dict}')
