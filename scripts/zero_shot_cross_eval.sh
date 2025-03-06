#!/bin/bash
# custom config
DATA="/l/users/umaima.rahman/datasets/"
MODEL_DIR_PATH="/l/users/umaima.rahman/research/sem6/baseline_lafter_checkpoints"
TRAINER=LaFTer
CFG=vit_b32
dset="$1"
txt_cls=zero_shot
CUDA_VISIBLE_DEVICES=1 python /home/umaima.rahman/research/sem6/LaFTer/LaFTer_original_cross_eval.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--lr 1e-2 \
--zero_shot \
--txt_cls ${txt_cls} \
--model_path ${MODEL_DIR_PATH}/"model_lafter_original_${dset}".pth \
--pickle_file_path zero_shot_pickle/"zero_shot_original_${dset}".pkl