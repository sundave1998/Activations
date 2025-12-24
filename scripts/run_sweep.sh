#!/usr/bin/env bash
# Run a parameter sweep over activations and learning rates
# Creates descriptive experiment names and stores logs under logs/<exp_name>/run.log

set -euo pipefail

# --- Configuration (edit as needed) ---
# ACTS=(relu selu exp reu elu softmax)
ACTS=(relu selu)

# LRS=(1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1)
LRS=(5e-2 1e-1)
EPOCHS=20
HIDDEN=512
BATCH=128
P_DROP=0.05
OUTDIR=logs
NUM_WORKERS=4
# Set to "true" to enable wandb (default); set to "false" to disable
RUN_WANDB=true

# Optional: run only first N combinations for quick smoke testing
#SMOKE_ONLY=false
#MAX_RUNS=3

mkdir -p "${OUTDIR}"

echo "Sweep configuration: activations=${ACTS[*]}, lrs=${LRS[*]}, epochs=${EPOCHS}, hidden=${HIDDEN}, batch=${BATCH}, outdir=${OUTDIR}"

run_count=0
for act in "${ACTS[@]}"; do
  for lr in "${LRS[@]}"; do
    exp_name="sweep_${act}_lr${lr}_hd${HIDDEN}_bs${BATCH}_p${P_DROP}"
    run_dir="${OUTDIR}/${exp_name}"
    mkdir -p "${run_dir}"

    echo "\n=== Running ${exp_name} ==="
    echo "Activation: ${act}, LR: ${lr}, Hidden: ${HIDDEN}, Epochs: ${EPOCHS}"

    # Build command
    cmd=(python scripts/run_experiment.py --exp-name "${exp_name}" --activation "${act}" --hidden-dim "${HIDDEN}" --epochs "${EPOCHS}" --lr "${lr}" \
         --batch-size "${BATCH}" --p-drop "${P_DROP}" --outdir "${OUTDIR}" --num-workers "${NUM_WORKERS}")

    if [ "${RUN_WANDB}" != "true" ]; then
        cmd+=(--no-wandb)
    fi

    # Run and save stdout/stderr to run.log
    "${cmd[@]}" 2>&1 | tee "${run_dir}/run.log"

    echo "Finished ${exp_name} — logs: ${run_dir}/run.log"

    run_count=$((run_count+1))
    # Quick throttle
    sleep 1

    # Optional smoke-only mode
    #if [ "${SMOKE_ONLY}" = "true" ] && [ "${run_count}" -ge "${MAX_RUNS}" ]; then
    #  echo "Smoke-only: stopping after ${MAX_RUNS} runs"
    #  exit 0
    #fi
  done
done

echo "\nAll done — total runs: ${run_count}."