#!/bin/bash
# custom config
DATA="/l/users/umaima.rahman/datasets/"
TRAINER=LaFTer
CFG=vit_b32
dset="$1"
txt_cls=lafter
CUDA_VISIBLE_DEVICES=1 python /home/umaima.rahman/research/sem6/LaFTer/mlhc_anchor_text.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--lr 1e-2 \
--epochs 100 \
--lambda1 1e-3 \
--txt_cls ${txt_cls} \
--logspec "split_60_SGD_"