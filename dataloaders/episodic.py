import os
import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler
import random

class fixed_sampler(Sampler): #Not in Use

    def __init__(self,data_source,trial,fixed_support):

        self.files = data_source.files
        self.fixed_support_list = np.load(fixed_support)
        self.trial = trial


    def __iter__(self):

        for i in range(self.trial):
            trial_support = self.fixed_support_list[i]
            # print(trial_support)
            indices = [np.argwhere(self.files == b).flatten()[0] for b in trial_support]
            # indices = np.argwhere(np.isin(self.files, trial_support)).flatten().tolist()
            # print(indices)
            # print(self.files[indices])

            yield indices
