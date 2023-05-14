#!/bin/bash

MODEL="albert-base-v2"
MODEL_TYPE="albert"
N_TRAIN=(32)
TASKS=("record" "cb" "boolq")
PATTERN_IDS=("0" "0 1 2 3 4" "0 1 2 3 4")
DATA_PATH=/home/alderson/Desktop/MVA/NLP/fewglue/FewGLUE

config_len=${#N_TRAIN[@]}
n_tasks=${#TASKS[@]}

for (( task_idx=0; task_idx < ${n_tasks}; task_idx++ ))
do
  task_data_path="${DATA_PATH}"/"${TASKS[$task_idx]}"
  base_out_path=/home/alderson/Desktop/MVA/NLP/data/outputs/"${MODEL}"/"${TASKS[$task_idx]}"
  base_idx=0
  while [ -d "${base_out_path}"/"${TASK}"/"${base_idx}" ]
  do
    base_idx=$((base_idx+1))
  done

  if [ "${TASKS[$task_idx]}" == "cb" ]; then
    n_test_examples=-1
  else
    n_test_examples=500
  fi

  for (( config_idx=0; config_idx < ${config_len}; config_idx++ ))
  do
    dir_idx=$(($config_idx+$base_idx))
    mkdir -p "${base_out_path}"/"${dir_idx}"

    python cli.py \
     --method pet \
     --pattern_ids ${PATTERN_IDS[$task_idx]} \
     --data_dir "${task_data_path}" \
     --model_type "${MODEL_TYPE}" \
     --model_name_or_path "${MODEL}" \
     --task_name "${TASKS[$task_idx]}" \
     --pet_per_gpu_train_batch_size 8 \
     --pet_per_gpu_eval_batch_size 1 \
     --output_dir "${base_out_path}"/"${dir_idx}" \
     --test_examples $n_test_examples \
     --unlabeled_examples 1000 \
     --do_train \
     --do_eval

  done
done
