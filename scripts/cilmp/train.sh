#!/bin/bash

DATA=${DATA:-/path/to/datasets}
TRAINER=CILMP

DATASET=$1
CFG=${2:-vit_b16}
SHOTS=${3:--1}

if [ -z "${DATASET}" ]; then
    echo "Usage: DATA=/path/to/datasets $0 <dataset> [cfg] [shots]"
    exit 1
fi

if [ "${SHOTS}" = "-1" ]; then
    RUN_TAG=fullshot
else
    RUN_TAG=${SHOTS}shots
fi

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${RUN_TAG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "The results exist at ${DIR}"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TEST.PER_CLASS_RESULT True
    fi
done
