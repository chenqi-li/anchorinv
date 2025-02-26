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
np.random.seed(3)

# Perform Z-standardization using training population scalar mean and std
# Create folder: mkdir -p /data/quee4692/NHIEdataset/NHIEdataset_bipolar_filtered_64hz_60s_p5overlap_psz_tt_dataloader
# Running this file: (anchorinv) preprocessing$ python nhiedataset_preprocessing_traintest_psz.py > "/data/quee4692/NHIEdataset/NHIEdataset_bipolar_filtered_64hz_60s_p5overlap_psz_tt_dataloader/dataset_summary.txt"

def bandpass_filter(t, signal, sampling_freq, low_cutoff_freq=0.5, high_cutoff_freq=30, visualize=False):
    # Define the parameters
    low_cutoff_freq = 0.5  # Hz
    high_cutoff_freq = 30  # Hz
    order = 5

    # Design the bandpass filter using butterworth filter
    b, a = butter(N=order, Wn=[low_cutoff_freq, high_cutoff_freq], fs=sampling_freq, btype='bandpass')
    
    # Visualize the filter response
    if visualize:
        w, h = freqz(b, a, worN=2000)
        plt.plot((sampling_freq * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        plt.plot([0, 0.5 * sampling_freq], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

    # # Create a sample signal (for demonstration purposes)
    # t = np.arange(0, 10, 1/sampling_freq)
    # signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)

    # Apply the bandpass filter to the signal
    filtered_signal = lfilter(b, a, signal)

    # Plot the original and filtered signals
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(t, signal, label='Original Signal')
        plt.plot(t, filtered_signal, label='Filtered Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
    return filtered_signal





#--------------
dataset_dir = "/data/quee4692/NHIEdataset"
preprocessed_dir = "/data/quee4692/NHIEdataset/NHIEdataset_bipolar_filtered_64hz"
dataloader_dir = "/data/quee4692/NHIEdataset/NHIEdataset_bipolar_filtered_64hz_60s_p5overlap_tt_dataloader"



segment_time = 1*60 # seconds
overlap = 0.5
fold = 5
resample_freq = 64
preprocessing = False
fold_splitting = True
#--------------
metadata = pd.read_csv(os.path.join(dataset_dir,"metadata.csv"))


if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)
if not os.path.exists(dataloader_dir):
    os.makedirs(dataloader_dir)

# Bipolar, bandpass, downsample, save
if preprocessing == True:
    for index, row in tqdm(metadata.iterrows()):
        file = row['file_ID']
        df = pd.read_csv(os.path.join(dataset_dir,"CSV_format",f'{file}.csv.xz'), compression='xz')
        
        # # Bandpass filter before bipolar montage
        # filtered = df.copy(deep=True)
        # for column in df:
        #     if len(column)==2:
        #         filtered[column] = bandpass_filter(df['time'], df[column], sampling_freq=row['sampling_freq'], visualize=False)
        #         break
        # # print(filtered, df)
        
        # Generate bipolar montage
        bipolar = pd.DataFrame()
        bipolar['F4-C4'] = df['F4'] - df['C4']
        bipolar['C4-O2'] = df['C4'] - df['O2']
        bipolar['F3-C3'] = df['F3'] - df['C3']
        bipolar['C3-O1'] = df['C3'] - df['O1']
        bipolar['T4-C4'] = df['T4'] - df['C4']
        bipolar['C4-Cz'] = df['C4'] - df['Cz']
        bipolar['Cz-C3'] = df['Cz'] - df['C3']
        bipolar['C3-T3'] = df['C3'] - df['T3']

        # Bandpass filter after bipolar montage
        filtered = bipolar.copy(deep=True)
        for column in bipolar:
            filtered[column] = bandpass_filter(df['time'], bipolar[column], sampling_freq=row['sampling_freq'], visualize=False)
            # break
        # print(filtered, df)

        # Downsample
        filtered_downsampled = pd.DataFrame()
        secs = filtered.shape[0]/row['sampling_freq'] # Number of seconds in signal X
        samps = secs*resample_freq     # Number of samples to downsample to have 64Hz frequency
        for col in filtered:
            filtered_downsampled[col], t = scipy.signal.resample(filtered[col], int(samps), t=df['time'].to_numpy())
            # t = scipy.signal.resample(df['time'].to_numpy(), int(samps))
            # plt.scatter(df['time'], filtered[col])
            # plt.scatter(t, filtered_downsampled[col])
            # plt.show()
        # print(filtered)
        # print(filtered_downsampled)

        # Save
        np.save(os.path.join(preprocessed_dir, row['file_ID']),np.asarray(filtered_downsampled))
        # break


if fold_splitting == True:
    # Do stratified group k fold
    dataset_count = metadata['grade'].value_counts().to_dict()
    baby_id = metadata['baby_ID'].to_numpy()
    groups = np.array([int(i[2:]) for i in baby_id])
    y = metadata['grade'].to_numpy()
    X = np.zeros((metadata.shape[0], 1))
    sgkf = StratifiedGroupKFold(n_splits=5)
    sgkf.split(X, y, groups)
    sampled_test_set_list = np.empty((0,))
    sampled_train_set_list = np.empty((0,))
    for i, (train_index, val_index) in enumerate(sgkf.split(X, y, groups)): #use first fold for test, second fold for validation, third fourth fifth for training
        train_label, train_cnt = np.unique(y[train_index],return_counts=True)
        val_label, val_cnt = np.unique(y[val_index],return_counts=True)
        if i == 0: #test
            sampled_test_set_list = np.concatenate([sampled_test_set_list, metadata.iloc[val_index]['file_ID'].to_numpy()])
            print(f"Test set")
        elif i==1 or i==2 or i==3 or i==4: #train
            sampled_train_set_list = np.concatenate([sampled_train_set_list, metadata.iloc[val_index]['file_ID'].to_numpy()])
            print(f"Train set")

        print(f"        file_ID={metadata.iloc[val_index]['file_ID'].to_numpy()}")
        print(f"        class_count={val_label}{val_cnt}")
        print(f"        label={y[val_index]}")
        print(f"        group={groups[val_index]}")
        print(f"        Fold Ratio (in terms of the entire dataset): {len(val_index)/(len(train_index)+len(val_index))}")
    np.save(os.path.join(dataloader_dir, "test_set.npy"), sampled_test_set_list)
    np.save(os.path.join(dataloader_dir, "train_set.npy"), sampled_train_set_list)


    # Split and save
    print(f'Splitting and saving')
    # Compute mean and standard deviation for averaging
    train_set = np.load(os.path.join(dataloader_dir, "train_set.npy"),allow_pickle=True)
    test_set = np.load(os.path.join(dataloader_dir, "test_set.npy"),allow_pickle=True)
    print(f'Train set size: {train_set.shape}')
    print(f'Test set size: {test_set.shape}')

    train_data = np.empty([0,8])
    for file in tqdm(os.listdir(preprocessed_dir), desc='Gathering all training data for computing mean and std'):
        if file.split('.')[0] in train_set:
            train_data = np.vstack([train_data, np.load(os.path.join(preprocessed_dir,file))])
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    
    print(f'Training Data size: {train_data.shape}')
    print(f'Scalar Mean of Training Data: {train_mean}')
    print(f'Scalar StD of Training Data: {train_std}')

    for file in tqdm(os.listdir(preprocessed_dir), desc='Saving segments'):
        data = np.load(os.path.join(preprocessed_dir,file))
        # plt.plot(data)
        # plt.show()
        data = (data-train_mean)/train_std
        # plt.plot(data)
        # plt.show()
        segment_len = segment_time*resample_freq
        stride = int((1-overlap)*segment_len)
        for ind, start in enumerate(range(0, data.shape[0]-stride, stride)):       
            filename = os.path.join(dataloader_dir, f'{file[:-4]}_{str(ind).zfill(3)}') 
            # print(data[start:start+segment_len,:].shape)    
            np.save(filename, data[start:start+segment_len,:])
            # print(ind, start, start+segment_len)

        
        # # Visualize downsampling
        # # print(downsampled)
        # plt.scatter(df['time'],df['F4'])
        # plt.show(block=False)
        # plt.figure()
        # plt.scatter(downsampled['time'],downsampled['F4'])
        # plt.show()

    shutil.copyfile(os.path.join(dataset_dir,'metadata.csv'),os.path.join(dataloader_dir,'metadata.csv'))