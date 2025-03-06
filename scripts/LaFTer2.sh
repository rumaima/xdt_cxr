#!/bin/bash
# custom config
DATA="/l/users/umaima.rahman/datasets/"
TRAINER=LaFTer
CFG=vit_l14
dset="$1"
txt_cls=lafter
CUDA_VISIBLE_DEVICES=1 python /home/umaima.rahman/research/sem6/LaFTer/LaFTer_loss2.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--lr 0.0005 \
--txt_cls ${txt_cls} \
--lossfn ce \
