#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS=$5
GPUS_PER_NODE=$6
CPUS_PER_TASK=$7
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:8}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}




# Original args
# GPUS=${GPUS:-4}
# GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# CPUS_PER_TASK=${CPUS_PER_TASK:-5}
# PY_ARGS=${@:5}
# SRUN_ARGS=${SRUN_ARGS:-""}
