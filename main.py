"""
AnchorInv: Few-Shot Class-Incremental Learning of Physiological Signals via Feature Space-Guided Inversion
"""
import argparse
import os
import numpy as np
import random
import datetime
import time
import datetime
import json

import torch
from torch.backends import cudnn

# Setup GPU
cudnn.benchmark = False
cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

def main(experiment_run, dataloader_dir, output_dir, args):
    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Record starttime
    starttime = datetime.datetime.now()

    # Set seeds
    seed_n = args.seed
    print('Seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    # Print experiment number
    print(f'Running {experiment_run}')

    # Train or evaluate
    from trainer.trainer_inc import FSCIL
    exp = FSCIL(output_dir, args)
    if args.train==1:
        exp.train(dataloader_dir)
    elif args.train==0:
        exp.load_evaluate(dataloader_dir, args.eval_suffix)

    endtime = datetime.datetime.now()
    print(f'Experiment run duration: {str(endtime - starttime)}')


if __name__ == "__main__":
    #--------
    parser = argparse.ArgumentParser()
    # Experiment setup parameters
    parser.add_argument("--experiment_run", type=int, required=True)
    parser.add_argument("--dataloader_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True, default="/users/quee4692/anchorinv/experiment_results")
    parser.add_argument("--seed", type=int, required=True)
    # Base stage training parameters
    parser.add_argument("--optimizer", type=str, required=True, default='Adam')
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--earlystop", type=int, required=True, help='0 no early stop, 1 with earlystop')
    parser.add_argument("--weighted_loss", type=json.loads, required=False, help='List of weight for loss to use for each class, should give for all classes')
    parser.add_argument("--wd", type=float, required=False, help='Weight Decay')
    parser.add_argument("--b1", type=float, required=False, help='Adam: momentum')
    parser.add_argument("--b2", type=float, required=False, help='Adam: momentum')
    parser.add_argument("--b", type=float, required=False, help='SGD: momentum')
    parser.add_argument("--steplr", type=json.loads, required=False, help='Step learning rate scheduler [step_size, gamma]')
    parser.add_argument("--samples_per_class", type=json.loads, required=False, help='Define the number of samples to use for each class during training with source_class, length should be the same as source_class. -1 means use all from that class')
    parser.add_argument("--subject_id", type=int, required=False, help='BCI: use a subset of the subjects')
    # Algorithm specific parameters
    parser.add_argument("--backbone", type=str, required=True, choices=['conformer'])
    parser.add_argument("--norm_layer", type=str, required=False, choices=['batch', 'layer'], help='Conformer: choose BatchNorm2D or LayerNorm')
    parser.add_argument("--ft_epoch", type=json.loads, required=False, help='Number of epochs to finetune for each incremental session')
    parser.add_argument("--ft_lr", type=float, required=False, help='Learning rate to finetune for each incremental session')
    parser.add_argument("--temperature", type=int, required=False, help='ProtoNet: Temperature for prototypical network')
    parser.add_argument("--ft_bb_layer", nargs='+', required=False, help='Choose which layers of backbone to finetune, feature_extractor is entire backbone, feature_extractor.1.5 is last layer of backbone, feature_extractor.0 is convolution block, feature_extractor.1 is transformer block')
    parser.add_argument("--ft_cos_scale", type=float, required=False, default=0, help='Cosine embedding loss to push new prototype away from old ones')
    parser.add_argument("--ft_base_scale", type=float, required=False, default=1, help='Scale for base class loss')
    parser.add_argument("--ft_push_scale", type=float, required=False, default=0, help='Cosine embedding loss to push new prototype away from all the old anchors')
    parser.add_argument("--fc_init", type=int, required=False, help='Initialize FC layer for new class as the average of embedding from support samples before finetuning. 0 to not intialize and use Pytorch initialization 1 to initialize')
    parser.add_argument("--inv_n_base_samples", type=int, required=False, help='Number of base samples per class to store for use in incremental sessions.')
    parser.add_argument("--inv_steps", type=int, required=False, help='Inversion number of steps')
    parser.add_argument("--inv_lr", type=float, required=False, help='Inversion learning rate')
    parser.add_argument("--inv_optim", type=str, required=False, choices=['Adam', 'SGD'], help='Inversion optimizer')
    parser.add_argument("--inv_loss", type=str, required=False, choices=['MAE', 'MSE', 'cos'], help='Inversion loss')
    parser.add_argument("--inv_init", type=str, required=False, choices=['zeros', 'randn'], help='Inversion initialization')
    parser.add_argument("--inv_base_select", type=str, required=False, choices=['random_sample', 'sigma', 'kmeans', 'kmedoids', 'condensation', 'kmeans_silhouette', 'closest', 'farthest', 'spaced','closest_cos', 'farthest_cos', 'spaced_cos', 'tsne','kmeans_basemix', 'closest_0.3_random', 'closest_0.5_random', 'closest_0.7_random', 'correct_random'], help='What samples to invert')
    parser.add_argument("--inv_n_cluster", type=int, required=False, help='inv_base_select as clusters: Number of clusters')
    parser.add_argument("--inv_base_once", type=int, required=False, help='Invert base session once or for each trial')
    parser.add_argument("--inv_l2_scale", type=float, required=False, default=0, help='L2 loss scale')
    parser.add_argument("--inv_tv_l2_scale", type=float, required=False, default=0, help='Total variance L2 loss scale')
    parser.add_argument("--inv_tv_l1_scale", type=float, required=False, default=0, help='Total variance L1 loss scale')
    parser.add_argument("--inv_bn_l_scale", type=float, required=False, default=0, help='Batch norm loss scale')
    parser.add_argument("--inv_chcor_scale", type=float, required=False, default=0, help='Channel correlation loss scale')
    parser.add_argument("--inv_laplacian_scale", type=float, required=False, default=0, help='Laplacian loss scale')
    # Incremental settings
    parser.add_argument("--source_class", type=json.loads, required=True, help="Select subset of the classes and remap them starting from 0, e.g. [0, 2, 3] -> [0, 1, 2]")
    parser.add_argument("--target_class", type=json.loads, required=True, help="Class target_class")
    parser.add_argument("--incremental_shots", type=int, required=False, help='Number of shots for each incremental class')
    parser.add_argument("--n_trial", type=int, required=False, help='Number of times few-shot learning is repeated for each incremental session')
    # Save model
    parser.add_argument("--save_model", type=int, required=True, help='Save model with .pth')
    # Train eval toggle
    parser.add_argument("--checkpoint", type=int, required=False, help='Choosing the checkpoint epoch, -1: best_f1_macroall.pth, >0: load the given epoch checkpoint')
    parser.add_argument("--train", type=int, required=True, help='1: train the model with the whole pipeline, 0: just load checkpoint and evaluate')
    parser.add_argument("--eval_suffix", type=int, required=False, help='Suffix for train=0 to keep logs of loading base model and doing incremental sessions')
    




    args = parser.parse_args()
    #--------
    dataloader_dir = args.dataloader_dir
    experiment_run = f'experiment_{str(args.experiment_run).zfill(3)}'
    output_dir = os.path.join(args.result_dir, experiment_run)
    output_dir = os.path.join(output_dir, f"bs_{args.bs}_lr_{args.lr}_wd_{args.wd}_seed_{args.seed}")
    print(time.asctime(time.localtime(time.time())))
    main(experiment_run, dataloader_dir, output_dir, args)
    print(time.asctime(time.localtime(time.time())))