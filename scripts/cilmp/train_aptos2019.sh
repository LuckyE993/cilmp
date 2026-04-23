#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CILMP_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

DATA_ROOT=${DATA_ROOT:-${HOME}/qixuan/datasets}
DATASET=${DATASET:-aptos2019}
TRAINER=${TRAINER:-CILMP}
CFG=${CFG:-vit_b16}
SHOTS=${SHOTS:--1}
SEEDS=${SEEDS:-"1 2 3"}
OUTPUT_BASE=${OUTPUT_BASE:-output/${DATASET}}

if [ "${SHOTS}" = "-1" ]; then
    RUN_TAG=fullshot
else
    RUN_TAG=${SHOTS}shots
fi

cd "${CILMP_DIR}"

for SEED in ${SEEDS}
do
    DIR=${OUTPUT_BASE}/${TRAINER}/${CFG}_${RUN_TAG}/seed${SEED}
    if [ -d "${DIR}" ]; then
        echo "The results exist at ${DIR}"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
            --root "${DATA_ROOT}" \
            --seed "${SEED}" \
            --trainer "${TRAINER}" \
            --dataset-config-file "configs/datasets/${DATASET}.yaml" \
            --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
            --output-dir "${DIR}" \
            DATASET.NUM_SHOTS "${SHOTS}" \
            TEST.PER_CLASS_RESULT True \
            "$@"
    fi
done
