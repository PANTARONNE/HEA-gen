#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=outputs/hea_finetune_run

GPUS="${GPUS:-2,6}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

NUM_GPUS="$(awk -F',' '{print NF}' <<< "${GPUS}")"

NUM_NODES="${NUM_NODES:-1}"

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "=========================================================="
echo " HEA atom-only multi-GPU training"
echo "   CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "   trainer.devices      = ${NUM_GPUS}"
echo "   trainer.num_nodes    = ${NUM_NODES}"
echo "=========================================================="

mattergen-train \
    --config-name=hea_finetune \
    trainer.devices="${NUM_GPUS}" \
    trainer.num_nodes="${NUM_NODES}" \
    "$@"
