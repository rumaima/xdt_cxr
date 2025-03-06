#!/bin/bash
# custom config
DATA="/l/users/umaima.rahman/datasets/"
MODEL_DIR_PATH="/l/users/umaima.rahman/research/sem6/mlhc_10_checkpoints"
TRAINER=LaFTer
CFG=vit_b32
LOGSPEC="_sup_SGD_10_percent"
dset="$1"
dset_target="$2"
txt_cls=lafter
CUDA_VISIBLE_DEVICES=3 python /home/umaima.rahman/research/sem6/LaFTer/mlhc_anchor_cross_eval.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset_target}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset_target}" \
--lr 1e-2 \
--epochs 1 \
--lambda1 1e-3 \
--txt_cls ${txt_cls} \
--logspec ${LOGSPEC} \
--model_path ${MODEL_DIR_PATH}/"${dset}${LOGSPEC}".pth \
--pickle_file_path pickle/"${dset}_${dset_target}${LOGSPEC}".pkl