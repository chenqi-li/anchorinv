
import numpy as np
import pandas as pd
import os
import sys

dataloader_dir = "/data/quee4692/GRABMdataset/GRABMdataset_filtered_256hz_5s_psz_tt_dataloader" #"/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_psz_session_tt_dataloader/" #"/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_psz_tt_dataloader/" #"/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_tt_dataloader/" #"/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_subdep_tt_dataloader/"
support_size = 10
seed_n = 66
n_trial = 100
split = 'train'



with open(os.path.join(dataloader_dir, f'split{split}_ntrial{n_trial}_support{support_size}_seed{seed_n}.txt'), 'w') as f:
    # Redirect stdout to the file
    sys.stdout = f

    # Load all the data for the given split
    np.random.seed(seed_n)
    split_set = np.load(os.path.join(dataloader_dir,f'{split}_set.npz'))
    files = np.array([])
    labels = np.array([])
    for ky in split_set.keys():
        file_segments = split_set[ky]
        label_segments = (np.ones(len(file_segments))*(int(ky))).astype(int)
        files = np.hstack([files,file_segments])
        labels = np.hstack([labels,label_segments])
    
    # Select subset as support for incremental learning
    for label in sorted(np.unique(labels.astype(int))):
        files_subset_support_trials = []
        # Repeat for n_trial number of times
        for trial in range(n_trial):
            label_idx = np.argwhere(labels==label)
            files_subset = files[label_idx]
            labels_subset = labels[label_idx]
            # print(f'For label {label}, we have {files_subset.shape} samples, for example  {files_subset[0]} {files_subset[-1]}')
            np.random.shuffle(files_subset)
            assert len(np.unique(labels_subset)) == 1
            files_subset_support = files_subset[:support_size].reshape(-1)
            files_subset_support_trials.append(files_subset_support)
        print(f'Selected {support_size} samples over {n_trial} number of trials for support: {files_subset_support_trials}')


    
        output_file = os.path.join(dataloader_dir, f'split{split}_ntrial{n_trial}_support{support_size}_seed{seed_n}_class{label}.npy')
        print(f'Saving to {output_file}')
        np.save(output_file, files_subset_support_trials)

        