#!/bin/bash

source /users/quee4692/miniconda3/etc/profile.d/conda.sh
conda activate anchorinv
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8



#######################################
#######Deterministic Train NHIE########
#######################################
# python -u /users/quee4692/anchorinv/main.py \
#     --experiment_run 356 \
#     --dataloader_dir "/data/quee4692/NHIEdataset/NHIEdataset_bipolar_filtered_64hz_60s_p5overlap_tt_dataloader" \
#     --result_dir "/users/quee4692/anchorinv/experiment_results" \
#     --seed 5 \
#     --optimizer 'Adam' \
#     --bs 256 \
#     --lr 0.00001 \
#     --n_epochs 1500 \
#     --earlystop 0 \
#     --weighted_loss "[104, 31, 22, 12]" \
#     --wd 0 \
#     --b1 0.5 \
#     --b2 0.999 \
#     --backbone 'conformer' \
#     --norm_layer 'batch' \
#     --temperature 16 \
#     --incremental 1 \
#     --source_class "[0, 1]" \
#     --target_class "[2, 3]" \
#     --n_trial 0 \
#     --save_model 1 \
#     --train 1 \
#     > "/users/quee4692/anchorinv/experiment_results/logs/experiment_356.txt" &

# wait

######################################
#######Deterministic Train BCI########
######################################
# python -u /users/quee4692/anchorinv/main.py \
#     --experiment_run 359 \
#     --dataloader_dir "/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_psz_session_tt_dataloader/" \
#     --result_dir "/users/quee4692/anchorinv/experiment_results" \
#     --seed 5 \
#     --optimizer 'Adam' \
#     --bs 72 \
#     --lr 0.0002 \
#     --n_epochs 2000 \
#     --earlystop 0 \
#     --wd 0 \
#     --b1 0.5 \
#     --b2 0.999 \
#     --backbone 'conformer' \
#     --norm_layer 'batch' \
#     --temperature 16 \
#     --incremental 1 \
#     --source_class "[0, 1]" \
#     --target_class "[2, 3]" \
#     --n_trial 0 \
#     --save_model 1 \
#     --train 1 \
#     > "/users/quee4692/anchorinv/experiment_results/logs/experiment_359.txt" &
# wait

########################################
#######Deterministic Train GRABM########
########################################
# python -u /users/quee4692/anchorinv/main.py \
#     --experiment_run 362 \
#     --dataloader_dir "/data/quee4692/GRABMdataset/GRABMdataset_filtered_256hz_5s_psz_tt_dataloader/" \
#     --result_dir "/users/quee4692/anchorinv/experiment_results" \
#     --seed 5 \
#     --optimizer 'Adam' \
#     --bs 256 \
#     --lr 0.00005 \
#     --n_epochs 2000 \
#     --earlystop 0 \
#     --wd 0 \
#     --b1 0.5 \
#     --b2 0.999 \
#     --backbone 'conformer' \
#     --norm_layer 'batch' \
#     --temperature 16 \
#     --incremental 1 \
#     --source_class "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" \
#     --target_class "[10, 11, 12, 13, 14, 15]" \
#     --n_trial 0 \
#     --save_model 1 \
#     --train 1 \
#     > "/users/quee4692/anchorinv/experiment_results/logs/experiment_362.txt" &


# wait


##############################################
#####Deterministic Eval NHIE 100trials########
##############################################
python -u /users/quee4692/anchorinv/main.py \
    --experiment_run 356 \
    --dataloader_dir "/data/quee4692/NHIEdataset/NHIEdataset_bipolar_filtered_64hz_60s_p5overlap_tt_dataloader" \
    --result_dir "/users/quee4692/anchorinv/experiment_results" \
    --seed 5 \
    --optimizer 'Adam' \
    --bs 256 \
    --lr 0.00001 \
    --n_epochs 1500 \
    --earlystop 0 \
    --wd 0 \
    --b1 0.5 \
    --b2 0.999 \
    --backbone 'conformer' \
    --norm_layer 'batch' \
    --ft_epoch '[1500, 1000]' \
    --ft_lr 0.00001 \
    --temperature 16 \
    --ft_bb_layer 'feature_extractor.1.5' \
    --fc_init 0 \
    --inv_n_base_samples 50 \
    --inv_steps 4000 \
    --inv_lr 0.01 \
    --inv_optim 'Adam' \
    --inv_loss 'MAE' \
    --inv_init 'randn' \
    --inv_base_select 'random_sample' \
    --inv_base_once 1 \
    --inv_laplacian_scale 0.05 \
    --incremental 1 \
    --source_class "[0, 1]" \
    --target_class "[2, 3]" \
    --incremental_shots 10 \
    --n_trial 100 \
    --save_model 1 \
    --checkpoint -1 \
    --train 0 \
    --eval_suffix 1046 \
    > "/users/quee4692/anchorinv/experiment_results/logs/experiment_356_eval1046.txt" 
wait

#############################################
#####Deterministic Eval BCI 100trials########
#############################################
# python -u /users/quee4692/anchorinv/main.py \
#     --experiment_run 359 \
#     --dataloader_dir "/data/quee4692/BCIdataset/BCIdataset_filtered_250hz_4s_psz_session_tt_dataloader/" \
#     --result_dir "/users/quee4692/anchorinv/experiment_results" \
#     --seed 5 \
#     --optimizer 'Adam' \
#     --bs 72 \
#     --lr 0.0002 \
#     --n_epochs 2000 \
#     --earlystop 0 \
#     --wd 0 \
#     --b1 0.5 \
#     --b2 0.999 \
#     --backbone 'conformer' \
#     --norm_layer 'batch' \
#     --ft_epoch '[1250, 700]' \
#     --ft_lr 0.00002 \
#     --temperature 16 \
#     --ft_bb_layer 'feature_extractor.1.5' \
#     --fc_init 0 \
#     --inv_n_base_samples 50 \
#     --inv_steps 4000 \
#     --inv_lr 0.01 \
#     --inv_optim 'Adam' \
#     --inv_loss 'MAE' \
#     --inv_init 'randn' \
#     --inv_base_select 'random_sample' \
#     --inv_base_once 1 \
#     --incremental 1 \
#     --source_class "[0, 1]" \
#     --target_class "[2, 3]" \
#     --incremental_shots 10 \
#     --n_trial 100 \
#     --save_model 1 \
#     --checkpoint 1500 \
#     --train 0 \
#     --eval_suffix 1012 \
#     > "/users/quee4692/anchorinv/experiment_results/logs/experiment_359_eval1012.txt"
# wait

#############################################
#####Deterministic Eval GRABM 100trials######
#############################################
# python -u /users/quee4692/anchorinv/main.py \
#     --experiment_run 362 \
#     --dataloader_dir "/data/quee4692/GRABMdataset/GRABMdataset_filtered_256hz_5s_psz_tt_dataloader/" \
#     --result_dir "/users/quee4692/anchorinv/experiment_results" \
#     --seed 5 \
#     --optimizer 'Adam' \
#     --bs 256 \
#     --lr 0.00005 \
#     --n_epochs 2000 \
#     --earlystop 0 \
#     --wd 0 \
#     --b1 0.5 \
#     --b2 0.999 \
#     --backbone 'conformer' \
#     --norm_layer 'batch' \
#     --ft_epoch '[1100, 1300, 0, 0, 0, 0]' \
#     --ft_lr 0.000005 \
#     --temperature 16 \
#     --ft_bb_layer 'feature_extractor.1.5' \
#     --fc_init 1 \
#     --inv_n_base_samples 50 \
#     --inv_steps 2000 \
#     --inv_lr 0.01 \
#     --inv_optim 'Adam' \
#     --inv_loss 'MAE' \
#     --inv_init 'randn' \
#     --inv_base_select 'random_sample' \
#     --inv_base_once 1 \
#     --incremental 1 \
#     --source_class "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" \
#     --target_class "[10, 11, 12, 13, 14, 15]" \
#     --incremental_shots 10 \
#     --n_trial 100 \
#     --save_model 1 \
#     --checkpoint 1900 \
#     --train 0 \
#     --eval_suffix 1014 \
#     > "/users/quee4692/anchorinv/experiment_results/logs/experiment_362_eval1014.txt" &

# wait