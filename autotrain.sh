#!/bin/bash

MODEL="albert-base-v2"
MODEL_TYPE="albert"
N_TRAIN=(32)
TASKS=("copa" "record" "cb" "boolq")
PATTERN_IDS=("0 1" "0" "0 1 2 3 4" "0 1 2 3 4")
DATA_PATH=/home/alderson/Desktop/MVA/NLP/fewglue/FewGLUE/"${TASK}"

config_len=${#N_TRAIN[@]}
n_tasks=${#TASKS[@]}

for (( task_idx=0; task_idx < ${n_tasks}; task_idx++ ))
do
  base_out_path=/home/alderson/Desktop/MVA/NLP/data/outputs/"${MODEL}"/"${TASKS[$task_idx]}"
  base_idx=0
  while [ -d "${base_out_path}"/"${TASK}"/"${base_idx}" ]
  do
    base_idx=$((base_idx+1))
  done
  for (( config_idx=0; config_idx < ${config_len}; config_idx++ ))
  do
    dir_idx=$(($config_idx+$base_idx))
    mkdir -p "${base_out_path}"/"${dir_idx}"

    python cli.py \
     --method pet \
     --pattern_ids "${PATTERN_IDS[$task_idx]}" \
     --data_dir "${DATA_PATH}" \
     --model_type "${MODEL_TYPE}" \
     --model_name_or_path "${MODEL}" \
     --task_name "${TASKS[$task_idx]}" \
     --pet_per_gpu_eval_batch_size 1 \
     --output_dir ../"${base_out_path}"/"${TASK}"/"${dir_idx}" \
     --do_train \
     --do_eval

  done
done
