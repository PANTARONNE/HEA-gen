#!/usr/bin/env bash
set -euo pipefail

# All paths below are relative to the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")"

# RUN_ID picks the run directory under outputs/hea_finetune_run/.
#   unset -> fresh timestamped dir, weights injected  = new fine-tune from mattergen_base
#   set    -> reuse that dir, injection skipped if it already holds checkpoints
#             = auto_resume picks up where the crashed run left off
#
# The timestamp is resolved ONCE here and exported, because both the injection
# step and mattergen-train must land in the SAME directory. If we let hydra
# expand ${now:...} instead, each process would get its own timestamp, training
# would start in an empty run dir, _find_latest_checkpoint would return None and
# the model would silently train from random init with no warm start at all.
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d/%H-%M-%S)}"
export OUTPUT_DIR="outputs/hea_finetune_run/${RUN_ID}"

PRETRAINED="${PRETRAINED:-checkpoints/mattergen_base/checkpoints/last.ckpt}"

GPUS="${GPUS:-5,7}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

NUM_GPUS="$(awk -F',' '{print NF}' <<< "${GPUS}")"

NUM_NODES="${NUM_NODES:-1}"

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

shopt -s nullglob
EXISTING_CKPTS=("${OUTPUT_DIR}"/checkpoints/*.ckpt)
shopt -u nullglob

echo "=========================================================="
echo " HEA atom-only multi-GPU fine-tuning"
echo "   OUTPUT_DIR           = ${OUTPUT_DIR}"
echo "   CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "   trainer.devices      = ${NUM_GPUS}"
echo "   trainer.num_nodes    = ${NUM_NODES}"
echo "=========================================================="

if [ ${#EXISTING_CKPTS[@]} -gt 0 ]; then
    echo "Found ${#EXISTING_CKPTS[@]} checkpoint(s) in ${OUTPUT_DIR}/checkpoints"
    echo "-> skipping weight injection, auto_resume will continue this run."
else
    if [ ! -f "${PRETRAINED}" ]; then
        echo "ERROR: pretrained checkpoint not found: ${PRETRAINED}" >&2
        exit 1
    fi
    echo "Injecting mattergen_base backbone from ${PRETRAINED} ..."
    python inject_pretrained_weights.py \
        --pretrained "${PRETRAINED}" \
        --output-dir "${OUTPUT_DIR}"
fi

mattergen-train \
    --config-name=hea_finetune \
    trainer.devices="${NUM_GPUS}" \
    trainer.num_nodes="${NUM_NODES}" \
    "$@"
