import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NHIE data loader
class NHIEdataloader(Dataset):
    def __init__(self, dataloader_dir, split, source_class, class_mapping=None, samples_per_class=None): # samples_per_class = Downsamples number of samples for each class to the given size
        self.dataloader_dir = dataloader_dir
        self.split = split
        self.baby_epochs = np.load(os.path.join(self.dataloader_dir,f'{split}_set.npy'), allow_pickle=True)
        self.files = np.array([])
        self.labels = np.array([])
        self.all_files = os.listdir(self.dataloader_dir)
        self.metadata = pd.read_csv(os.path.join(self.dataloader_dir,"metadata.csv"))
        # Keep class_mapping with source_classes only
        self.class_mapping = {}
        for key in class_mapping.keys():
            if key in source_class:
                self.class_mapping[key] = class_mapping[key]
        # Filter all the files and put the desired ones in self.files and self.labels
        for epoch in self.baby_epochs:
            idx = np.flatnonzero(np.core.defchararray.find(self.all_files,epoch)!=-1)
            # self.files.append(self.all_files[idx])
            row = self.metadata.loc[self.metadata['file_ID']==epoch]
            # print(row)
            # print(type(row['grade'].to_numeric()))
            label = np.asarray(row['grade'])-1
            if int(label) in self.class_mapping.keys(): #only add the file if belongs to the demanded classes
                file_segments = np.take(self.all_files,np.array(idx),axis=0)
                label_segments = (np.ones(len(file_segments))*self.class_mapping[int(label)]).astype(int)
                self.files = np.hstack([self.files,file_segments])
                self.labels = np.hstack([self.labels,label_segments])
            # print(type(label))
            # print(np.take(self.all_files,np.array(idx),axis=0))
        # If samples_per_class variable is given, then downsample the dataset to the requested size
        if samples_per_class != None:
            print("Beginning dataset downsample.")
            print(f'Before dataset downsample {np.unique(self.labels,return_counts=True)}{self.labels.dtype}')
            unique_labels = np.unique(self.labels)
            assert len(samples_per_class) == len(unique_labels)
            reduced_files = []
            reduced_labels = []

            for label in unique_labels:
                label = int(label)
                # Find indices of samples belonging to the current class
                indices = np.where(self.labels == label)[0]
                if samples_per_class[label] == -1:
                    # Keep all the samples for this class
                    selected_indices = indices
                else:
                    # Randomly select the required number of samples for this class
                    selected_indices = np.random.choice(indices, size=samples_per_class[label], replace=False)
                # Append selected samples to the reduced files and labels lists
                reduced_files.extend(self.files[selected_indices])
                reduced_labels.extend(self.labels[selected_indices])
            self.files = np.array(reduced_files)
            self.labels = np.array(reduced_labels)
            print(f'After dataset downsample {np.unique(self.labels,return_counts=True)}{self.labels.dtype}')
            print(f'Training dataset has been downsampled. Desired: {samples_per_class}. Result: {np.unique(self.labels,return_counts=True)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        eeg = np.load(os.path.join(self.dataloader_dir,file), allow_pickle=True).T
        eeg = torch.from_numpy(np.expand_dims(eeg, 0))
        label = torch.tensor(self.labels[idx])

        eeg = Variable(eeg.type(torch.FloatTensor).to(device))
        label = Variable(label.type(torch.LongTensor).to(device))
        return eeg, label, file


# BCI data loader
class BCIdataloader(Dataset):
    def __init__(self, dataloader_dir, split, source_class, class_mapping=None, samples_per_class=None, subject_id=None): # samples_per_class = Downsamples number of samples for each class to the given size
        self.dataloader_dir = dataloader_dir
        self.split = split
        self.split_set = np.load(os.path.join(self.dataloader_dir,f'{split}_set.npz'), allow_pickle=True)
        self.files = np.array([])
        self.labels = np.array([])
        # Keep class_mapping with source_classes only
        self.class_mapping = {}
        for key in class_mapping.keys():
            if key in source_class:
                self.class_mapping[key] = class_mapping[key]
        # Filter all the files and put the desired ones in self.files and self.labels
        for ky in self.class_mapping.keys(): # self.class_mapping keys are the classes we want from [0, 1, 2, 3] and values are the model output classes. self.split_set classes are the original dataset classes [1, 2, 3, 4]
            file_segments = self.split_set[str(ky+1)]
            label_segments = (np.ones(len(file_segments))*self.class_mapping[int(ky)]).astype(int)
            self.files = np.hstack([self.files,file_segments])
            self.labels = np.hstack([self.labels,label_segments])
        # If subject_id is variable is given, then only use data from these subjects
        if subject_id != None:
            print('Beginning subject filter.')
            number_parts = np.array([filename.split('_')[0][1:3] for filename in self.files]).astype(int)
            print(f'Before subject filter we have subjects {np.unique(number_parts)} with total of {self.files.shape} files')
            mask = np.isin(number_parts, subject_id)
            self.files = self.files[mask]
            self.labels = self.labels[mask]
            number_parts = np.array([filename.split('_')[0][1:3] for filename in self.files]).astype(int)
            print(f'After subject filter we have subjects {np.unique(number_parts)} with total of {self.files.shape} files')
        # If samples_per_class variable is given, then downsample the dataset to the requested size
        if samples_per_class != None:
            print("Beginning dataset downsample.")
            print(f'Before dataset downsample {np.unique(self.labels,return_counts=True)}{self.labels.dtype}')
            unique_labels = np.unique(self.labels)
            assert len(samples_per_class) == len(unique_labels)
            reduced_files = []
            reduced_labels = []

            for label in unique_labels:
                label = int(label)
                # Find indices of samples belonging to the current class
                indices = np.where(self.labels == label)[0]
                if samples_per_class[label] == -1:
                    # Keep all the samples for this class
                    selected_indices = indices
                else:
                    # Randomly select the required number of samples for this class
                    selected_indices = np.random.choice(indices, size=samples_per_class[label], replace=False)
                # Append selected samples to the reduced files and labels lists
                reduced_files.extend(self.files[selected_indices])
                reduced_labels.extend(self.labels[selected_indices])
            self.files = np.array(reduced_files)
            self.labels = np.array(reduced_labels)
            print(f'After dataset downsample {np.unique(self.labels,return_counts=True)}{self.labels.dtype}')
            print(f'Training dataset has been downsampled. Desired: {samples_per_class}. Result: {np.unique(self.labels,return_counts=True)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        eeg = np.load(os.path.join(self.dataloader_dir,file), allow_pickle=True).T
        eeg = torch.from_numpy(np.expand_dims(eeg, 0))
        label = torch.tensor(self.labels[idx])

        eeg = Variable(eeg.type(torch.FloatTensor).to(device))
        label = Variable(label.type(torch.LongTensor).to(device))
        return eeg, label, file

# GRABM data loader
class GRABMdataloader(Dataset):
    def __init__(self, dataloader_dir, split, source_class, class_mapping=None, samples_per_class=None): # samples_per_class = Downsamples number of samples for each class to the given size
        self.dataloader_dir = dataloader_dir
        self.split = split
        self.split_set = np.load(os.path.join(self.dataloader_dir,f'{split}_set.npz'), allow_pickle=True)
        self.files = np.array([])
        self.labels = np.array([])
        # Keep class_mapping with source_classes only
        self.class_mapping = {}
        for key in class_mapping.keys():
            if key in source_class:
                self.class_mapping[key] = class_mapping[key]
        # Filter all the files and put the desired ones in self.files and self.labels
        for ky in self.class_mapping.keys(): # self.class_mapping keys are the classes we want from [0, 1, ..., 15] and values are the model output classes. self.split_set classes are the original dataset classes [0, 1, ..., 15]
            file_segments = self.split_set[str(ky)]
            label_segments = (np.ones(len(file_segments))*self.class_mapping[int(ky)]).astype(int)
            self.files = np.hstack([self.files,file_segments])
            self.labels = np.hstack([self.labels,label_segments])
        # If samples_per_class variable is given, then downsample the dataset to the requested size
        if samples_per_class != None:
            print("Beginning dataset downsample.")
            print(f'Before dataset downsample {np.unique(self.labels,return_counts=True)}{self.labels.dtype}')
            unique_labels = np.unique(self.labels)
            assert len(samples_per_class) == len(unique_labels)
            reduced_files = []
            reduced_labels = []

            for label in unique_labels:
                label = int(label)
                # Find indices of samples belonging to the current class
                indices = np.where(self.labels == label)[0]
                if samples_per_class[label] == -1:
                    # Keep all the samples for this class
                    selected_indices = indices
                else:
                    # Randomly select the required number of samples for this class
                    selected_indices = np.random.choice(indices, size=samples_per_class[label], replace=False)
                # Append selected samples to the reduced files and labels lists
                reduced_files.extend(self.files[selected_indices])
                reduced_labels.extend(self.labels[selected_indices])
            self.files = np.array(reduced_files)
            self.labels = np.array(reduced_labels)
            print(f'After dataset downsample {np.unique(self.labels,return_counts=True)}{self.labels.dtype}')
            print(f'Training dataset has been downsampled. Desired: {samples_per_class}. Result: {np.unique(self.labels,return_counts=True)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        eeg = np.load(os.path.join(self.dataloader_dir,file), allow_pickle=True).T
        eeg = torch.from_numpy(np.expand_dims(eeg, 0))
        label = torch.tensor(self.labels[idx])

        eeg = Variable(eeg.type(torch.FloatTensor).to(device))
        label = Variable(label.type(torch.LongTensor).to(device))
        return eeg, label, file