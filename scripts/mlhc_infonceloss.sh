#!/bin/bash
# custom config
DATA="/l/users/umaima.rahman/datasets/"
MODEL_DIR_PATH="/l/users/umaima.rahman/research/sem6/mlhc_loss_checkpoints"
LOSS="INFONCE_only_"
TRAINER=LaFTer
LOGSPEC="_sup_SGD_LOSS_"
CFG=vit_b32
dset="$1"
txt_cls=lafter
CUDA_VISIBLE_DEVICES=2 python /home/umaima.rahman/research/sem6/LaFTer/mlhc_anchor_infonceloss.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--lr 1e-2 \
--epochs 55 \
--lambda1 1e-3 \
--txt_cls ${txt_cls} \
--logspec ${LOGSPEC} \
--model_path ${MODEL_DIR_PATH}/"${LOSS}${dset}${LOGSPEC}".pth 
