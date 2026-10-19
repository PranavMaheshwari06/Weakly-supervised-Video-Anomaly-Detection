#!/usr/bin/env bash
# Reproducible Stage-2 dispatch experiment workflow.
# Assumes the existing frozen LoRA feature store and teacher scores are present.
# This prepares fixed evaluation bags, trains one fresh run, and evaluates validation only.
# Test evaluation and demonstrations are never run automatically.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_dispatch_pipeline.sh RUN_TAG [TRAINING_SEED]"
    exit 2
fi

RUN_TAG="$1"
TRAINING_SEED="${2:-}"
CFG="stage2_class/configs/hierA_dispatch_v1.yaml"

echo "=============================================="
echo " DISPATCH MIL REPRODUCIBILITY WORKFLOW"
echo " Run tag: ${RUN_TAG}"
echo " Config:  ${CFG}"
echo "=============================================="

echo ""
echo ">>> Preparing immutable validation and test bags..."
python -m stage2_class.src.preprocess.m30_prepare_eval_bags --cfg "${CFG}"

TRAIN_ARGS=(--cfg "${CFG}" --run_tag "${RUN_TAG}")
if [[ -n "${TRAINING_SEED}" ]]; then
    TRAIN_ARGS+=(--seed "${TRAINING_SEED}")
fi

echo ""
echo ">>> Training dispatch MIL in a fresh run directory..."
python -m stage2_class.src.train.m27_train_dispatch_mil "${TRAIN_ARGS[@]}"

echo ""
echo ">>> Evaluating the fixed validation bags at threshold 0.5..."
python -m stage2_class.src.eval.m29_eval_dispatch \
    --split val \
    --run_tag "${RUN_TAG}"

if [[ "${RUN_DISPATCH_DEMO:-0}" == "1" ]]; then
    echo ""
    echo ">>> Running explicitly requested indexed-video demonstrations..."
    DEMO_VIDEOS=(
        "Shooting/Shooting001_x264"
        "Arson/Arson016_x264"
        "Robbery/Robbery050_x264"
    )

    for VIDEO_ID in "${DEMO_VIDEOS[@]}"; do
        python inference/inference_any_video.py \
            --video_id "${VIDEO_ID}" \
            --run_tag "${RUN_TAG}" \
            --mode dispatch
    done
fi

echo ""
echo "Workflow complete. Test evaluation was not run."
echo "Validation result: artifacts/stage2/eval/${RUN_TAG}/dispatch_eval_val.json"
