#!/bin/bash

DATADIR="/scratch/vision_robustness/data"
LISTDIR="/scratch/TeSLA/dataloaders/segmentation/visda" 

SEED=42
# --pretrained_source_path ../Source_Segmentation/VisDA/seed_${SEED}/deeplab_epoch_5_lr_0.0002 \
# torchrun --nproc_per_node=2 

CUDA_VISIBLE_DEVICES=0 torchrun run_tta_seg.py \
    --pretrain \
    --target_data_path ${DATADIR}/gta5/ \
    --target_info_path ${LISTDIR}/cityscapes_list/info.json \
    --target_list_path ${LISTDIR}/gta5_list/all.txt \
    --target_eval_list_path ${LISTDIR}/gta5_list/all.txt \
    --experiment_dir ./exp_segmentation/pretrain/visdas/seed_${SEED} \
    --dataset_name visdas \
    --dataset_type rgb \
    --target_sites cityscape \
    --n_classes 19 \
    --seed ${SEED} \
    --sub_policy_dim 3 \
    --ema_momentum 0.999 \
    --n_epochs 3 \
    --bn_epochs 2 \
    --weak_mult 3