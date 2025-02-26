"""
AnchorInv: Few-Shot Class-Incremental Learning of Physiological Signals via Feature Space-Guided Inversion

We use eeg-conformer as backbone for our experiments. https://github.com/eeyhsong/EEG-Conformer
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sn
from torchsummary import summary
from copy import deepcopy

from models.protonet_inc import ProtoNetInc, replace_base_fc, model_inv
from dataloaders.classic import NHIEdataloader, BCIdataloader, GRABMdataloader
from dataloaders.episodic import fixed_sampler
from collections import defaultdict



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FSCIL():
    def __init__(self, output_dir, args):
        super(FSCIL, self).__init__()
        # Initialize hyperparameters
        self.output_dir = output_dir
        self.args = args
        self.chcor_matrix = None

        # Class label mapping
        all_classes = self.args.source_class+self.args.target_class
        self.class_mapping = {clas:i for i,clas in enumerate(all_classes)}

        # Initialize the model
        if 'NHIEdataset' in self.args.dataloader_dir:
            self.model = ProtoNetInc(len(self.args.source_class)+len(self.args.target_class), backbone=self.args.backbone, temperature=self.args.temperature, feat_size=2360)
            self.model = self.model.to(device)
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(device)
            summary(self.model, (1, 8, 3840))
        elif 'BCIdataset' in self.args.dataloader_dir:
            self.model = ProtoNetInc(len(self.args.source_class)+len(self.args.target_class), backbone=self.args.backbone, temperature=self.args.temperature, feat_size=2440)
            self.model = self.model.to(device)
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(device)
            summary(self.model, (1, 22, 1000))
        elif 'GRABMdataset' in self.args.dataloader_dir:
            self.model = ProtoNetInc(len(self.args.source_class)+len(self.args.target_class), backbone=self.args.backbone, temperature=self.args.temperature, feat_size=2320)
            self.model = self.model.to(device)
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(device)
            summary(self.model, (1, 28, 1280))

        # Initialize Loss
        if self.args.weighted_loss:
            self.criterion_cls_base = torch.nn.CrossEntropyLoss(weight=torch.tensor(1/np.array(self.args.weighted_loss),dtype=torch.float32)[self.args.source_class]).to(device)
            self.criterion_cls_new = torch.nn.CrossEntropyLoss(weight=torch.tensor(1/np.array(self.args.weighted_loss),dtype=torch.float32)[self.args.source_class]).to(device)
            self.criterion_cls = torch.nn.CrossEntropyLoss(weight=torch.tensor(1/np.array(self.args.weighted_loss),dtype=torch.float32)[self.args.source_class]).to(device)
        else:
            self.criterion_cls_base = torch.nn.CrossEntropyLoss()
            self.criterion_cls_new = torch.nn.CrossEntropyLoss()
            self.criterion_cls = torch.nn.CrossEntropyLoss()
    
        # Record the hyperparameters
        print("Hyperparameters")
        print(f'args: {self.args}')

        print("Output Directory")
        print(f'output_dir: {self.output_dir}')

        print("Model")
        print(f'model: {self.model}')
        
        print("Class Mapping")
        print(f'class mapping: {self.class_mapping}')


    def train(self, dataloader_dir):
        # Initialize variable 
        self.dataloader_dir = dataloader_dir
        # SummaryWriter
        writer = SummaryWriter(os.path.join(self.output_dir,'tensorboard'))
        # Dataloader for source domain (base session)
        if 'NHIEdataset' in self.args.dataloader_dir:
            self.files = os.listdir(self.dataloader_dir)
            train_dataset = NHIEdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping, samples_per_class=self.args.samples_per_class)
            test_dataset = NHIEdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping)
            self.train_loader = DataLoader(train_dataset,batch_size=self.args.bs,shuffle=True)
            self.test_loader = DataLoader(test_dataset,batch_size=int(sum('ID' in s for s in self.files)/169), shuffle=False)
        elif 'BCIdataset' in self.args.dataloader_dir:
            train_dataset = BCIdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping, samples_per_class=self.args.samples_per_class, subject_id=self.args.subject_id)
            test_dataset = BCIdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping, subject_id=self.args.subject_id)
            self.train_loader = DataLoader(train_dataset,batch_size=self.args.bs,shuffle=True)
            self.test_loader = DataLoader(test_dataset,batch_size=self.args.bs, shuffle=False)
        elif 'GRABMdataset' in self.args.dataloader_dir:
            train_dataset = GRABMdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping, samples_per_class=self.args.samples_per_class)
            test_dataset = GRABMdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping)
            self.train_loader = DataLoader(train_dataset,batch_size=self.args.bs,shuffle=True)
            self.test_loader = DataLoader(test_dataset,batch_size=self.args.bs, shuffle=False)

        # Optimizers
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2), weight_decay=self.args.wd)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.b)
        if self.args.steplr is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.steplr[0], gamma=self.args.steplr[1])
        elif self.args.steplr is None:
            self.scheduler = None
        
        # Initialize training parameters
        best_score = 0
        iteration = 0
        best_epoch = 0

        # Training
        for e in range(self.args.n_epochs):
            #################################
            ##### Training base session #####
            #################################
            self.model.train()
            # Iterate through the dataloader
            for i, (eeg, label, file) in enumerate(self.train_loader):
                # Increment iteration
                iteration += 1

                # Forward Pass
                tok, outputs = self.model(eeg)
                outputs = outputs[:,:len(self.args.source_class)]
                loss = self.criterion_cls(outputs, label)
                # writer.add_scalar("loss/train_iter", loss.item(), iteration) # log training loss for the iteration
            
                # Backward Pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Adjust learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                writer.add_scalar(f"learningrate", self.optimizer.param_groups[0]['lr'], e)

            ######################
            ##### Test Basic #####
            ######################
            # Copy model for loading back later to not interrupt training
            self.model_copy = deepcopy(self.model.state_dict()) # save copy of model to make sure nothing has changed by the end of the epoch
            self.optimizer_copy = deepcopy(self.optimizer.state_dict()) # save copy of optimizer to make sure nothing has changed by the end of the epoch
            # Prepare model for evaluation
            self.model = replace_base_fc(self.train_loader, self.model, self.args)            
            # Test again with saved 
            score = self.evaluate_epoch(data_loader=self.train_loader, split='train_saved', writer=writer, e=e, cm_ax_label=self.args.source_class)
            score = self.evaluate_epoch(data_loader=self.test_loader, split='test_saved', writer=writer, e=e, cm_ax_label=self.args.source_class)
            
            # Save model if good
            if self.args.save_model and (score > best_score):
                best_score = score
                best_epoch = e
                torch.save(self.model.module.state_dict(), os.path.join(self.output_dir,f'best_f1_macroall.pth'))
                print(f"Model saved for epoch {e}, with best score {best_score}")
            if self.args.save_model and (e % 100 == 0):
                torch.save(self.model.module.state_dict(), os.path.join(self.output_dir,f'epoch_{str(e).zfill(4)}.pth'))
                print(f"Model saved for epoch {e}, with score {score}")
            
            # Save model from last epoch
            if e+1 == self.args.n_epochs:
                torch.save(self.model.module.state_dict(), os.path.join(self.output_dir,f'last_epoch.pth'))
                print(f"Finished training, model saved for epoch {e}, with score {score}")

            ################################
            ##### Incremental Sessions #####
            ################################
            # Skip incremental part if there are no incremental classes
            if len(self.args.target_class) == 0 or self.args.n_trial == 0:
                # Load back optimizer, model state_dict to avoid interference with training due to evaluation and/or incremental session
                self.model.load_state_dict(self.model_copy)
                self.optimizer.load_state_dict(self.optimizer_copy)
                continue
            
            # Perform incremental learning every number of epochs
            if e%50 != 0:
                # Load back optimizer, model state_dict to avoid interference with training due to evaluation and/or incremental session
                self.model.load_state_dict(self.model_copy)
                self.optimizer.load_state_dict(self.optimizer_copy)
                continue
            else:
                # Incremental sessions
                self.incremental_sessions(e=e, writer=writer, print_results=False)
                # Load back optimizer, model state_dict to avoid interference with training due to evaluation and/or incremental session
                self.model.load_state_dict(self.model_copy)
                self.optimizer.load_state_dict(self.optimizer_copy)
            
            # Early stopping
            if e - best_epoch > 1000 and self.args.earlystop: # early stopping if no improvements for a number of epochs
                break
            

        
        # Load best epoch and evaluate the results and save to self.args.n_epochs
        if self.args.save_model:
            print(f"Loading model from epoch {best_epoch}")
            state_dict = torch.load(os.path.join(self.output_dir,f'best_f1_macroall.pth'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.'+k # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            self.model.to(device)
            # Base session
            score = self.evaluate_epoch(data_loader=self.test_loader, split='test_checkpoint', writer=writer, e=self.args.n_epochs, cm_ax_label=self.args.source_class, print_results=True)
            # # Incremental sessions
            # if len(self.args.target_class) != 0:
            #     self.incremental_sessions(e=self.args.n_epochs, writer=writer, print_results=True)

        
        writer.flush()
        writer.close()

        return best_score


    def evaluate_epoch(self, data_loader, split, writer, e, cm_ax_label, incremental_support=[(None,None,None)], print_results=False):
        # Initialize variables
        epoch_label = []
        epoch_pred = []
        epoch_loss = 0
        trial_cnt = 0
        self.iter_score_hist = defaultdict(list)
        # (With Support) Enable dynamic growing buffer for storing past session samples
        for trial_idx, (eeg_support, label_support, file_support) in enumerate(incremental_support):
            if eeg_support is not None:
                model_inc_copy = deepcopy(self.model_inc)
        # Go through the support loader, i.e. the trials
        for trial_idx, (eeg_support, label_support, file_support) in enumerate(incremental_support): #Support samples to update the incremental class prototypes if provided, n_trial number of times
            # (With Support) Update model_inc with support set
            if eeg_support is not None:
                # Load self.model_inc with self.model_inc_copy, which is model from last session, to start new trial for current session
                self.model_inc = deepcopy(model_inc_copy) # add this line to allow model state_dict to have growing buffer for storing past session samples
                self.model_inc.load_state_dict(deepcopy(self.statedict_list[trial_idx]), strict=True)
                # Print progress
                print(f"Starting trial {trial_idx}")
                # Adapt self.model_inc with support samples
                self.model_inc.module.mode='proto'
                self.model_inc.eval()
                if self.args.fc_init == 1:
                    self.model_inc.module.update_fc(eeg_support,label_support)
                self.finetune_backbone(eeg_support,label_support, data_loader)
                if len(cm_ax_label) != len(self.args.source_class)+len(self.args.target_class) or self.args.n_trial == 1: # do not need to invert for last session, save unnecessary compute
                        self.model_inc = model_inv([(eeg_support, label_support, file_support)], self.model_inc, self.output_dir, self.args, self.chcor_matrix)
                # Save the adapted model from this trial to the statedict_list
                self.statedict_list[trial_idx] = deepcopy(self.model_inc.state_dict())
                # Set model in eval mode
                self.model_inc.eval()
            # (Without Support) Set model in eval mode
            elif eeg_support is None:
                self.model.eval()
            # Print model mode
            if trial_idx == 0 and eeg_support is not None:
                if eeg_support is None and hasattr(self.model.module, 'mode'):
                    print(f'self.model in {self.model.module.mode} mode.')
                elif eeg_support is not None and hasattr(self.model_inc.module, 'mode'):
                    print(f"self.model_inc in {self.model_inc.module.mode} mode.")
            # Go through data_loader and measure the performance
            for i, (eeg, label, file) in enumerate(data_loader):
                # Forward Pass
                if eeg_support is not None: #(With Support)
                    tok, outputs = self.model_inc(eeg)
                    outputs = outputs[:,:len(cm_ax_label)]
                    loss = self.criterion_cls_inc(outputs, label)
                elif eeg_support is None: #(Without Support)
                    tok, outputs = self.model(eeg)
                    outputs = outputs[:,:len(cm_ax_label)]
                    loss = self.criterion_cls(outputs, label)
                
                # Keep track of all the predictions and labels
                epoch_loss += loss.item()    
                pred = torch.max(outputs, 1)[1]
                epoch_pred.extend(pred.cpu().detach().numpy())
                epoch_label.extend(label.cpu().detach().numpy())
            
            trial_cnt += 1
        # Debugging mode when n_trial <= 10: Help to choose the number of epochs to finetune
        if self.args.n_trial <= 10 and eeg_support is not None:
            best_average = float('-inf')
            best_iter = None
            for key, values in self.iter_score_hist.items():
                average = sum(values) / len(values)
                self.iter_score_hist[key] = average
                if average > best_average:
                    best_average = average
                    best_iter = key
            print(self.iter_score_hist)
            print(f'For this session, finetuning for {best_iter} iterations gives the best performance of {best_average}')
        
        # Convert to numpy array
        epoch_label = np.array(epoch_label).reshape((-1,1))
        epoch_pred = np.array(epoch_pred).reshape((-1,1))

        # Log stats with all trials combined
        cf_matrix = confusion_matrix(epoch_label, epoch_pred)
        cf_matrix_norm = confusion_matrix(epoch_label, epoch_pred, normalize='true')
        perclassacc = cf_matrix_norm.diagonal()
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=cm_ax_label,columns=cm_ax_label)
        df_cm_abs = pd.DataFrame(cf_matrix, index=cm_ax_label,columns=cm_ax_label)
        plt.figure(figsize=(12, 7))
        writer.add_figure(f"cm_trialsum/{split}_percent", sn.heatmap(df_cm, annot=True, fmt='.2f').get_figure(), e)
        plt.figure(figsize=(12, 7))
        writer.add_figure(f"cm_trialsum/{split}_abs", sn.heatmap(df_cm_abs, annot=True, fmt='g').get_figure(), e)
        writer.add_scalar(f"loss_trialsum/{split}", epoch_loss/(i+1), e)
        f1_classwise = metrics.f1_score(epoch_label, epoch_pred, average=None)
        recall_classwise = metrics.recall_score(epoch_label, epoch_pred, average=None)
        precision_classwise = metrics.precision_score(epoch_label, epoch_pred, average=None)
        if len(cm_ax_label) > len(self.args.source_class): # metrics only for new classes
            f1_macrobase = np.mean(f1_classwise[:len(self.args.source_class)])
            f1_macronew = np.mean(f1_classwise[len(self.args.source_class):])
            f1_hmean = (2 * f1_macrobase * f1_macronew) / (f1_macrobase + f1_macronew + 1e-12)
            recall_macrobase = np.mean(recall_classwise[:len(self.args.source_class)])
            recall_macronew = np.mean(recall_classwise[len(self.args.source_class):])
            recall_hmean = (2 * recall_macrobase * recall_macronew) / (recall_macrobase + recall_macronew + 1e-12)
            precision_macrobase = np.mean(precision_classwise[:len(self.args.source_class)])
            precision_macronew = np.mean(precision_classwise[len(self.args.source_class):])
            precision_hmean = (2 * precision_macrobase * precision_macronew) / (precision_macrobase + precision_macronew + 1e-12)
            writer.add_scalar(f"f1_macrobase_trialsum/{split}", f1_macrobase, e)
            writer.add_scalar(f"f1_macronew_trialsum/{split}", f1_macronew, e)
            writer.add_scalar(f"f1_hmean_trialsum/{split}", f1_hmean, e)
            writer.add_scalar(f"recall_macrobase_trialsum/{split}", recall_macrobase, e)
            writer.add_scalar(f"recall_macronew_trialsum/{split}", recall_macronew, e)
            writer.add_scalar(f"recall_hmean_trialsum/{split}", recall_hmean, e)
            writer.add_scalar(f"precision_macrobase_trialsum/{split}", precision_macrobase, e)
            writer.add_scalar(f"precision_macronew_trialsum/{split}", precision_macronew, e)
            writer.add_scalar(f"precision_hmean_trialsum/{split}", precision_hmean, e)
            if print_results:
                print(f"f1_macrobase_trialsum/{split}", f1_macrobase)
                print(f"f1_macronew_trialsum/{split}", f1_macronew)
                print(f"f1_hmean_trialsum/{split}", f1_hmean)
                print(f"recall_macrobase_trialsum/{split}", recall_macrobase)
                print(f"recall_macronew_trialsum/{split}", recall_macronew)
                print(f"recall_hmean_trialsum/{split}", recall_hmean)
                print(f"precision_macrobase_trialsum/{split}", precision_macrobase)
                print(f"precision_macronew_trialsum/{split}", precision_macronew)
                print(f"precision_hmean_trialsum/{split}", precision_hmean)
        f1_macroall = np.mean(f1_classwise)
        recall_macroall = np.mean(recall_classwise)
        precision_macroall = np.mean(precision_classwise)
        writer.add_scalar(f"f1_macroall_trialsum/{split}", f1_macroall, e)
        writer.add_scalar(f"recall_macroall_trialsum/{split}", recall_macroall, e)
        writer.add_scalar(f"precision_macroall_trialsum/{split}", precision_macroall, e)
        if print_results:
            print(f"f1_macroall_trialsum/{split}", f1_macroall)
            print(f"recall_macroall_trialsum/{split}", recall_macroall)
            print(f"precision_macroall_trialsum/{split}", precision_macroall)
        
        # Log stats with trials separated
        assert epoch_label.shape[0] % trial_cnt == 0
        n_per_trial = int(epoch_label.shape[0]/trial_cnt)
        f1_macrobase_list = []
        f1_macronew_list = []
        f1_hmean_list = []
        f1_macroall_list = []
        recall_macrobase_list = []
        recall_macronew_list = []
        recall_hmean_list = []
        recall_macroall_list = []
        precision_macrobase_list = []
        precision_macronew_list = []
        precision_hmean_list = []
        precision_macroall_list = []
        for trial_i in range(trial_cnt):
            epoch_label_trial = epoch_label[(trial_i)*n_per_trial:(trial_i+1)*n_per_trial,:]
            epoch_pred_trial = epoch_pred[(trial_i)*n_per_trial:(trial_i+1)*n_per_trial,:]
            f1_classwise = metrics.f1_score(epoch_label_trial, epoch_pred_trial, average=None)
            recall_classwise = metrics.recall_score(epoch_label_trial, epoch_pred_trial, average=None)
            precision_classwise = metrics.precision_score(epoch_label_trial, epoch_pred_trial, average=None)
            if len(cm_ax_label) > len(self.args.source_class):
                f1_macrobase = np.mean(f1_classwise[:len(self.args.source_class)])
                f1_macronew = np.mean(f1_classwise[len(self.args.source_class):])
                f1_hmean = (2 * f1_macrobase * f1_macronew) / (f1_macrobase + f1_macronew + 1e-12)
                recall_macrobase = np.mean(recall_classwise[:len(self.args.source_class)])
                recall_macronew = np.mean(recall_classwise[len(self.args.source_class):])
                recall_hmean = (2 * recall_macrobase * recall_macronew) / (recall_macrobase + recall_macronew + 1e-12)
                precision_macrobase = np.mean(precision_classwise[:len(self.args.source_class)])
                precision_macronew = np.mean(precision_classwise[len(self.args.source_class):])
                precision_hmean = (2 * precision_macrobase * precision_macronew) / (precision_macrobase + precision_macronew + 1e-12)
                f1_macrobase_list.append(f1_macrobase)
                f1_macronew_list.append(f1_macronew)
                f1_hmean_list.append(f1_hmean)
                recall_macrobase_list.append(recall_macrobase)
                recall_macronew_list.append(recall_macronew)
                recall_hmean_list.append(recall_hmean)
                precision_macrobase_list.append(precision_macrobase)
                precision_macronew_list.append(precision_macronew)
                precision_hmean_list.append(precision_hmean)
            f1_macroall = np.mean(f1_classwise)
            recall_macroall = np.mean(recall_classwise)
            precision_macroall = np.mean(precision_classwise)
            f1_macroall_list.append(f1_macroall)
            recall_macroall_list.append(recall_macroall)
            precision_macroall_list.append(precision_macroall)
            if len(cm_ax_label) > len(self.args.source_class):
                print(f"Trial {trial_i}: F1MacroAll {f1_macroall} F1MacroBase {f1_macrobase} F1MacroNew {f1_macronew} RecallMacroNew {recall_macronew}")
                print(confusion_matrix(epoch_label_trial, epoch_pred_trial))
        if len(cm_ax_label) > len(self.args.source_class):
            writer.add_scalar(f"f1_macrobase_trialavg/{split}", np.mean(f1_macrobase_list), e)
            writer.add_scalar(f"f1_macrobase_trialstd/{split}", np.std(f1_macrobase_list), e)
            writer.add_scalar(f"f1_macronew_trialavg/{split}", np.mean(f1_macronew_list), e)
            writer.add_scalar(f"f1_macronew_trialstd/{split}", np.std(f1_macronew_list), e)
            writer.add_scalar(f"f1_hmean_trialavg/{split}", np.mean(f1_hmean_list), e)
            writer.add_scalar(f"f1_hmean_trialstd/{split}", np.std(f1_hmean_list), e)
            writer.add_scalar(f"recall_macrobase_trialavg/{split}", np.mean(recall_macrobase_list), e)
            writer.add_scalar(f"recall_macrobase_trialstd/{split}", np.std(recall_macrobase_list), e)
            writer.add_scalar(f"recall_macronew_trialavg/{split}", np.mean(recall_macronew_list), e)
            writer.add_scalar(f"recall_macronew_trialstd/{split}", np.std(recall_macronew_list), e)
            writer.add_scalar(f"recall_hmean_trialavg/{split}", np.mean(recall_hmean_list), e)
            writer.add_scalar(f"recall_hmean_trialstd/{split}", np.std(recall_hmean_list), e)
            writer.add_scalar(f"precision_macrobase_trialavg/{split}", np.mean(precision_macrobase_list), e)
            writer.add_scalar(f"precision_macrobase_trialstd/{split}", np.std(precision_macrobase_list), e)
            writer.add_scalar(f"precision_macronew_trialavg/{split}", np.mean(precision_macronew_list), e)
            writer.add_scalar(f"precision_macronew_trialstd/{split}", np.std(precision_macronew_list), e)
            writer.add_scalar(f"precision_hmean_trialavg/{split}", np.mean(precision_hmean_list), e)
            writer.add_scalar(f"precision_hmean_trialstd/{split}", np.std(precision_hmean_list), e)
            if print_results:
                print(f"f1_macrobase_trialavg/{split}", np.mean(f1_macrobase_list))
                print(f"f1_macrobase_trialstd/{split}", np.std(f1_macrobase_list))
                print(f"f1_macronew_trialavg/{split}", np.mean(f1_macronew_list))
                print(f"f1_macronew_trialstd/{split}", np.std(f1_macronew_list))
                print(f"f1_hmean_trialavg/{split}", np.mean(f1_hmean_list))
                print(f"f1_hmean_trialstd/{split}", np.std(f1_hmean_list))
                print(f"recall_macrobase_trialavg/{split}", np.mean(recall_macrobase_list))
                print(f"recall_macrobase_trialstd/{split}", np.std(recall_macrobase_list))
                print(f"recall_macronew_trialavg/{split}", np.mean(recall_macronew_list))
                print(f"recall_macronew_trialstd/{split}", np.std(recall_macronew_list))
                print(f"recall_hmean_trialavg/{split}", np.mean(recall_hmean_list))
                print(f"recall_hmean_trialstd/{split}", np.std(recall_hmean_list))
                print(f"precision_macrobase_trialavg/{split}", np.mean(precision_macrobase_list))
                print(f"precision_macrobase_trialstd/{split}", np.std(precision_macrobase_list))
                print(f"precision_macronew_trialavg/{split}", np.mean(precision_macronew_list))
                print(f"precision_macronew_trialstd/{split}", np.std(precision_macronew_list))
                print(f"precision_hmean_trialavg/{split}", np.mean(precision_hmean_list))
                print(f"precision_hmean_trialstd/{split}", np.std(precision_hmean_list))
        writer.add_scalar(f"f1_macroall_trialavg/{split}", np.mean(f1_macroall_list), e)
        writer.add_scalar(f"f1_macroall_trialstd/{split}", np.std(f1_macroall_list), e)
        writer.add_scalar(f"recall_macroall_trialavg/{split}", np.mean(recall_macroall_list), e)
        writer.add_scalar(f"recall_macroall_trialstd/{split}", np.std(recall_macroall_list), e)
        writer.add_scalar(f"precision_macroall_trialavg/{split}", np.mean(precision_macroall_list), e)
        writer.add_scalar(f"precision_macroall_trialstd/{split}", np.std(precision_macroall_list), e)
        if print_results:
            print(f"f1_macroall_trialavg/{split}", np.mean(f1_macroall_list))
            print(f"f1_macroall_trialstd/{split}", np.std(f1_macroall_list))
            print(f"recall_macroall_trialavg/{split}", np.mean(recall_macroall_list))
            print(f"recall_macroall_trialstd/{split}", np.std(recall_macroall_list))
            print(f"precision_macroall_trialavg/{split}", np.mean(precision_macroall_list))
            print(f"precision_macroall_trialstd/{split}", np.std(precision_macroall_list))
                
        return np.mean(f1_macroall_list)

    def incremental_sessions(self, e, writer, print_results):

        # Save copy of base session model for comparison later
        model_base = deepcopy(self.model.state_dict())
        model_base_copy = deepcopy(self.model)
        # Initialize self.model_inc which will be used in the incremental sessions
        self.model_inc = deepcopy(self.model)
        # Initialize list of state_dict for each trial
        self.statedict_list = [deepcopy(self.model_inc.state_dict()) for _ in range(self.args.n_trial)]
        # (self.args.train==0, i.e. incremental_sessions() called from load_evaluate) Preparation between base and incremental sessions, different across trials
        if self.args.train == 0:
            print('Starting replay prep for base session.')
            for trial_idx in range(len(self.statedict_list)):
                # Invert the base session examples
                if trial_idx == 0 or self.args.inv_base_once == 0:
                    # Load trial_idx-th model
                    self.mode_inc = deepcopy(model_base_copy) # add this line to allow model state_dict to have growing buffer for storing past session samples
                    self.model_inc.load_state_dict(deepcopy(self.statedict_list[trial_idx]), strict=True)
                    # Invert the prototype
                    if 'NHIEdataset' in self.args.dataloader_dir:
                        train_dataset = NHIEdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping)
                        self.train_loader = DataLoader(train_dataset,batch_size=self.args.inv_n_base_samples*len(self.args.source_class)*5,shuffle=True)
                    elif 'BCIdataset' in self.args.dataloader_dir:
                        train_dataset = BCIdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping, subject_id=self.args.subject_id)
                        self.train_loader = DataLoader(train_dataset,batch_size=self.args.inv_n_base_samples*len(self.args.source_class)*5,shuffle=True)
                    elif 'GRABMdataset' in self.args.dataloader_dir:
                        train_dataset = GRABMdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping)
                        self.train_loader = DataLoader(train_dataset,batch_size=self.args.inv_n_base_samples*len(self.args.source_class)*5,shuffle=True)
                    self.model_inc = model_inv(self.train_loader, self.model_inc, self.output_dir, self.args, self.chcor_matrix)
                    # Save the adapted model from this trial to the statedict_list
                    self.statedict_list[trial_idx] = deepcopy(self.model_inc.state_dict())
                # Do not invert the base session examples, just copy the model statedict that contain inverted samples from first trial
                elif trial_idx > 0 and self.args.inv_base_once == 1:
                    # Copy the first trial statedict to the rest
                    self.statedict_list[trial_idx] = deepcopy(self.statedict_list[0])
            print('Replay prep for base session completed.')
        # Loop through the sessions
        for session in range(len(self.args.target_class)):
            print(f"Starting session {session}")
            # Identify classes for current and previous sessions
            self.inc_class = np.asarray([self.args.target_class[session]]).flatten()
            self.ses_class = self.args.source_class+self.args.target_class[:session+1]
            inc_class_set = set(self.inc_class)
            ses_class_set = set(self.ses_class)
            result_set = ses_class_set - inc_class_set
            self.prev_class = np.array(list(result_set))
            # Initialize loss_inc
            if self.args.weighted_loss:
                self.criterion_cls_inc = torch.nn.CrossEntropyLoss(weight=torch.tensor(1/np.array(self.args.weighted_loss),dtype=torch.float32)[self.ses_class]).to(device)
            else:
                self.criterion_cls_inc = torch.nn.CrossEntropyLoss().to(device)
            # Initialize support dataloader for incremental classes and test dataloader for current session classes
            if 'NHIEdataset' in self.args.dataloader_dir:
                train_dataset_target = NHIEdataloader(self.dataloader_dir,'train',self.inc_class,class_mapping = self.class_mapping) #,fixed_support=f'splittrain_support50_seed66_class{int(self.inc_class)}.npy'
                test_dataset_target = NHIEdataloader(self.dataloader_dir,'test',self.ses_class,class_mapping = self.class_mapping)
                self.train_loader_target = DataLoader(
                    train_dataset_target,
                    batch_sampler = fixed_sampler(data_source=train_dataset_target,trial=self.args.n_trial,fixed_support=os.path.join(self.dataloader_dir, f'splittrain_ntrial100_support{self.args.incremental_shots}_seed66_class{int(self.inc_class)}.npy')))
                self.test_loader_target = DataLoader(test_dataset_target,batch_size=int(sum('ID' in s for s in self.files)/169), shuffle=False)
            elif 'BCIdataset' in self.args.dataloader_dir:
                train_dataset_target = BCIdataloader(self.dataloader_dir,'train',self.inc_class,class_mapping = self.class_mapping, subject_id=self.args.subject_id) #,fixed_support=f'splittrain_support50_seed66_class{int(self.inc_class)}.npy'
                test_dataset_target = BCIdataloader(self.dataloader_dir,'test',self.ses_class,class_mapping = self.class_mapping, subject_id=self.args.subject_id)
                self.train_loader_target = DataLoader(
                    train_dataset_target,
                    batch_sampler = fixed_sampler(data_source=train_dataset_target,trial=self.args.n_trial,fixed_support=os.path.join(self.dataloader_dir, f'splittrain_ntrial100_support{self.args.incremental_shots}_seed66_class{int(self.inc_class)}.npy')))
                self.test_loader_target = DataLoader(test_dataset_target,batch_size=self.args.bs, shuffle=False)
            elif 'GRABMdataset' in self.args.dataloader_dir:
                train_dataset_target = GRABMdataloader(self.dataloader_dir,'train',self.inc_class,class_mapping = self.class_mapping) #,fixed_support=f'splittrain_support50_seed66_class{int(self.inc_class)}.npy'
                test_dataset_target = GRABMdataloader(self.dataloader_dir,'test',self.ses_class,class_mapping = self.class_mapping)
                self.train_loader_target = DataLoader(
                    train_dataset_target,
                    batch_sampler = fixed_sampler(data_source=train_dataset_target,trial=self.args.n_trial,fixed_support=os.path.join(self.dataloader_dir, f'splittrain_ntrial100_support{self.args.incremental_shots}_seed66_class{int(self.inc_class)}.npy')))
                self.test_loader_target = DataLoader(test_dataset_target,batch_size=self.args.bs, shuffle=False)
            # Modify model_inc with support samples and evaluate on the test set
            score = self.evaluate_epoch(data_loader=self.test_loader_target, split=f'inc_{session}_test', writer=writer, e=e, cm_ax_label=self.ses_class, incremental_support=self.train_loader_target, print_results=print_results)
            # Compare self.model_inc to model_base to see the changes as a result of using support set
            models_differ = 0
            state_dict_inc = deepcopy(self.model_inc.state_dict())
            state_dict_base = deepcopy(model_base)
            all_keys = set(state_dict_inc.keys()).union(set(state_dict_base.keys())) #create a set of all keys in both models
            for key in sorted(all_keys):
                base_param = state_dict_base.get(key)
                inc_param = state_dict_inc.get(key)
                # Check if the key is missing in model_base
                if base_param is None:
                    print(f'New parameter in model for incremental session: {key}')
                    print(f'Parameter value: {inc_param}')
                    models_differ += 1
                    continue
                # Check if parameters are equal
                if torch.equal(base_param, inc_param):
                    continue
                else:
                    # Increment the difference counter
                    models_differ += 1
                    # Print mismatch information
                    print('Mismatch found at', key)
                    if 'fc' in key:
                        print(f'Prototypes from base session:\n{base_param}')
                        print(f'Prototypes from current session:\n{inc_param}')
            if models_differ == 0:  # Check if models match perfectly
                print('Models match perfectly! :)')
            print(f"End of session {session}")
    
    def finetune_backbone(self, eeg_support, label_support, data_loader):
        # Set mode to eval
        self.model_inc.eval()
        # Freeze part of model
        for name, param in self.model_inc.named_parameters():
            param.requires_grad = False # freeze all layer
            for layer in self.args.ft_bb_layer:
                if layer in name: # unfreeze the layer to be finetuned
                    param.requires_grad = True
            if 'fc' in name: # unfreeze prototype layer
                param.requires_grad = True
        # Intiailize optimizer for finetuning
        if self.args.optimizer == 'Adam':
            self.optimizer_inc = torch.optim.Adam(self.model_inc.parameters(), lr=self.args.ft_lr, betas=(self.args.b1, self.args.b2), weight_decay=self.args.wd)
        elif self.args.optimizer == 'SGD':
            self.optimizer_inc = torch.optim.SGD(self.model_inc.parameters(), lr=self.args.ft_lr, momentum=self.args.b)
        # Initialize variables
        best_score = 0
        best_model_inc = None
        best_iter = -1
        # Identify the number of epochs to finetune
        n_epoch = self.args.ft_epoch[label_support[0]-len(self.args.source_class)]
        # Finetune
        for itr in range(n_epoch):
            # Forward Pass
            tok, outputs = self.model_inc(eeg_support)
            outputs = outputs[:,:len(self.ses_class)]
            loss_new = self.criterion_cls_new(outputs, label_support)
            eeg_replay = self.model_inc.eeg_replay.to(device)
            label_replay = self.model_inc.label_replay.to(device)
            tok_replay, outputs = self.model_inc(eeg_replay)
            outputs = outputs[:,:len(self.ses_class)]
            loss_base = self.criterion_cls_base(outputs, label_replay)
            if self.args.ft_cos_scale > 0:
                all_prototypes = self.model_inc.module.fc.weight
                prev_prototypes = all_prototypes[:label_support[0], :]
                cur_prototype = all_prototypes[label_support[0], :].reshape(1, -1)
                cur_prototypes = cur_prototype.repeat(prev_prototypes.size()[0], 1)
                loss_cos = self.criterion_cos(prev_prototypes, cur_prototypes, -1*torch.ones(prev_prototypes.size()[0]).to(device))
            else:
                loss_cos = 0
            if self.args.ft_push_scale > 0:
                all_prototypes = self.model_inc.module.fc.weight
                cur_prototype = all_prototypes[label_support[0], :].reshape(1, -1)
                cur_prototypes = cur_prototype.repeat(tok_replay.size()[0], 1)
                loss_push = self.criterion_push(tok_replay, cur_prototypes, -1*torch.ones(tok_replay.size()[0]).to(device))
            else:
                loss_push = 0
            total_loss = loss_base + loss_new + self.args.ft_cos_scale*loss_cos + self.args.ft_push_scale*loss_push
        
            # Backward Pass
            self.optimizer_inc.zero_grad()
            total_loss.backward()
            for name, param in self.model_inc.named_parameters(): # zero out gradients for base class prototypes
                if 'fc' in name:
                    param.grad[0:label_support[0],:] = torch.zeros_like(param.grad[0:label_support[0],:])
            self.optimizer_inc.step()
            
            # Debuggin mode when n_trials <= 10: prints progress every 50 epochs and keep track which iteration gave best result
            if self.args.n_trial > 10:
                pass
            elif (itr) % 50 == 0 or itr == 0:
                # Print progress every 50 epochs
                self.model_inc.eval()
                epoch_label = []
                epoch_pred = []
                with torch.no_grad():
                    for i, (eeg, label, file) in enumerate(data_loader):
                        mu_hat, outputs = self.model_inc(eeg)
                        outputs = outputs[:,:label_support[0]+1]
                        pred = torch.max(outputs, 1)[1]
                        epoch_pred.extend(pred.cpu().detach().numpy())
                        epoch_label.extend(label.cpu().detach().numpy())                
                epoch_label = np.array(epoch_label).reshape((-1,1))
                epoch_pred = np.array(epoch_pred).reshape((-1,1))
                f1_classwise = metrics.f1_score(epoch_label, epoch_pred, average=None)
                f1_macroall = np.mean(f1_classwise)
                f1_macrobase = np.mean(f1_classwise[:len(self.args.source_class)])
                f1_macronew = np.mean(f1_classwise[len(self.args.source_class):])
                print(f'F1 at finetune iter {itr} gives f1_macroall of {f1_macroall} f1_macrobase of {f1_macrobase} f1_macronew of {f1_macronew}', end='')
                self.iter_score_hist[itr].append(f1_macroall)
                print('Base Loss:', loss_base.item(), 'New Loss:', loss_new.item(), 'Cos Loss:', loss_cos, 'Push Loss:', loss_push)
                # Keep track which iteration gave best result
                if f1_macroall>best_score:
                    best_score = f1_macroall
                    best_model_inc = deepcopy(self.model_inc.state_dict())
                    best_iter = itr
        # Summarize the finetuning process by printing which iteration gave what best score
        if self.args.n_trial > 10 or n_epoch == 0:
            pass
        else:
            self.model_inc.load_state_dict(best_model_inc)
            print(f"Loaded best model_inc from iter {best_iter} with f1_macroall of {best_score} out of {n_epoch} epochs")

    def load_evaluate(self, dataloader_dir, eval_suffix):
        # Initialize variable 
        self.dataloader_dir = dataloader_dir
        self.files = os.listdir(self.dataloader_dir)
        # Get new writer
        writer = SummaryWriter(os.path.join(self.output_dir+'_loadevaluate'+'_'+str(eval_suffix),'tensorboard'))
        # Get dataloader
        if 'NHIEdataset' in self.args.dataloader_dir:
            test_dataset = NHIEdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping)
            self.test_loader = DataLoader(test_dataset,batch_size=int(sum('ID' in s for s in self.files)/169), shuffle=False)
        elif 'BCIdataset' in self.args.dataloader_dir:
            test_dataset = BCIdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping, subject_id=self.args.subject_id)
            self.test_loader = DataLoader(test_dataset,batch_size=self.args.bs, shuffle=False)        # Load model from checkpoint
        elif 'GRABMdataset' in self.args.dataloader_dir:
            test_dataset = GRABMdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping)
            self.test_loader = DataLoader(test_dataset,batch_size=self.args.bs, shuffle=False)        # Load model from checkpoint
        # Get model checkpoint
        if self.args.checkpoint == -1:
            checkpoint_name = 'best_f1_macroall.pth'
        elif self.args.checkpoint >= 0:
            checkpoint_name = f'epoch_{str(self.args.checkpoint).zfill(4)}.pth'
        print(f"Loading model from {self.output_dir,checkpoint_name}")
        state_dict = torch.load(os.path.join(self.output_dir,checkpoint_name))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.'+k # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict,strict=False)
        self.model.to('cpu')
        # Report the differences between the model loaded against the desired architecture
        matched_keys = []
        mismatched_keys = []
        ckpt_not_model = []
        model_not_ckpt = []
        for name, param in self.model.named_parameters():
            if name in new_state_dict:
                if torch.equal(param, new_state_dict[name]):
                    matched_keys.append(name)
                else:
                    mismatched_keys.append(name)
            else:
                model_not_ckpt.append(name)
        for name in new_state_dict.keys():
            if name not in self.model.state_dict().keys():
                ckpt_not_model.append(name)
        print("Matched keys:")
        print(matched_keys)
        print("Mismatched keys:")
        print(mismatched_keys)
        print("In Checkpoint not in Model:")
        print(ckpt_not_model)
        print("In Model not in Checkpoint:")
        print(model_not_ckpt)
        self.model.to(device)
        # Base session
        print("Starting base session")
        score = self.evaluate_epoch(data_loader=self.test_loader, split='test_saved', writer=writer, e=self.args.n_epochs, cm_ax_label=self.args.source_class, print_results=True)
        torch.cuda.empty_cache()

        # (Optional) Preparation between base and incremental sessions
        # Calculate Channel Correlation
        if 'NHIEdataset' in self.args.dataloader_dir:
            train_dataset = NHIEdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping)
            train_loader = DataLoader(train_dataset,batch_size=self.args.bs,shuffle=True)
            test_dataset = NHIEdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping)
            test_loader = DataLoader(test_dataset,batch_size=int(sum('ID' in s for s in self.files)/169), shuffle=False)
        elif 'BCIdataset' in self.args.dataloader_dir:
            train_dataset = BCIdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping, subject_id=self.args.subject_id)
            train_loader = DataLoader(train_dataset,batch_size=self.args.bs,shuffle=True)
            test_dataset = BCIdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping, subject_id=self.args.subject_id)
            test_loader = DataLoader(test_dataset,batch_size=self.args.bs, shuffle=False)
        elif 'GRABMdataset' in self.args.dataloader_dir:
            train_dataset = GRABMdataloader(self.dataloader_dir,'train',self.args.source_class,class_mapping = self.class_mapping)
            train_loader = DataLoader(train_dataset,batch_size=self.args.bs,shuffle=True)
            test_dataset = GRABMdataloader(self.dataloader_dir,'test',self.args.source_class,class_mapping = self.class_mapping)
            test_loader = DataLoader(test_dataset,batch_size=self.args.bs, shuffle=False)
        # Save EEG and corresponding label
        eeg_list = []
        label_list = []
        with torch.no_grad():
            for i, (eeg, label, file) in enumerate(train_loader):
                eeg_list.append(eeg.cpu())
                label_list.append(label.cpu())
        eeg_list = torch.cat(eeg_list, dim=0).numpy()
        label_list = torch.cat(label_list, dim=0).numpy()
        correlation_matrix_list = []
        for eeg_i, eeg_data in enumerate(eeg_list):
            correlation_matrix = np.corrcoef(eeg_data.squeeze().T, rowvar=False)
            correlation_matrix_list.append(correlation_matrix)
        matrix_shape = correlation_matrix_list[0].shape
        average_matrix = np.zeros(matrix_shape)
        for matrix in correlation_matrix_list:
            average_matrix += matrix
        average_matrix /= len(correlation_matrix_list)
        plt.figure(figsize=(10, 8))
        sn.heatmap(average_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('EEG Channel Correlation Matrix')
        plt.show()
        img_path = os.path.join(self.output_dir+'_loadevaluate'+'_'+str(self.args.eval_suffix),'CorrelationMatrix')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        plt.savefig(os.path.join(img_path,f'average_matrix.png'))
        plt.close()
        self.chcor_matrix = average_matrix
        
        # Incremental sessions
        if len(self.args.target_class) != 0:
            self.incremental_sessions(e=self.args.n_epochs, writer=writer, print_results=True)