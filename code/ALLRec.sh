#!/bin/bash

# 设置输出目录
# movie
name="21_ALLRec64"
data="movie"
cutoff_len=256

output_dir="output/output${name}"
result_json_data="compare/compare${name}.json"
log_file="log/log${name}.txt"

train_data_path="../data/${data}/train.json"
val_data_path="../data/${data}/valid.json"
test_data_path="../data/${data}/test.json"


# LoRA 层数设置
models=("ALLRec")

# 采样数量
sample=64
CYCLES=6
UPDATE=4
# 运行循环次数
nums=(1 2 3 4 5)
batch_size=4
alpha=0.25

# 计算并格式化时间差
format_duration() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60))
    local seconds=$((duration % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# 微调模型的函数
fine_tune_model() {
    model=$1
    num=$2

    dir=${model}_${data}_${UPDATE}_${batch_size}_${num}
    user_bin_file="${output_dir}/${dir}/adapter_config.json"

    start_time=$(date "+%Y-%m-%d %H:%M:%S")
    start_time_epoch=$(date +%s)
    echo "[$start_time] start finetune model --> ${dir} " | tee -a ${log_file}
    if [ ! -d "${output_dir}/${dir}" ] || [ ! -f "${user_bin_file}" ]; then

      python ALLRec_finetune.py --train_data_path ${train_data_path} \
                          --val_data_path ${val_data_path} \
                          --output_dir ${output_dir}/${dir} \
                          --sample ${sample} \
                          --CYCLES ${CYCLES} \
                          --UPDATE ${UPDATE} \
                          --batch_size ${batch_size} \
                          --lora_layer ${model} \
                          --cutoff_len ${cutoff_len} \
                          --alpha ${alpha}
    fi

    end_time=$(date "+%Y-%m-%d %H:%M:%S")
    end_time_epoch=$(date +%s)
    duration=$((end_time_epoch - start_time_epoch))
    formatted_duration=$(format_duration $duration)
    echo "[$end_time] complete finetune model --> ${dir} " | tee -a ${log_file}
    echo "Finetune duration: ${formatted_duration}" | tee -a ${log_file}

    # 记录微调结束时间和耗时
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finetune completed for ${dir} in ${formatted_duration}" >> ${log_file}

    # 运行评估文件
    start_eval_time=$(date "+%Y-%m-%d %H:%M:%S")
    start_eval_time_epoch=$(date +%s)
    echo "[$start_eval_time] start evaluate model --> ${dir} " | tee -a ${log_file}

    if [  -f "${user_bin_file}" ]; then
      python ALLRec_evaluate.py --lora_weights ${output_dir}/${dir} \
                              --test_data_path ${test_data_path} \
                              --result_json_data ${result_json_data} \
                              --cutoff_len ${cutoff_len}
    fi
    end_eval_time=$(date "+%Y-%m-%d %H:%M:%S")
    end_eval_time_epoch=$(date +%s)
    eval_duration=$((end_eval_time_epoch - start_eval_time_epoch))
    formatted_eval_duration=$(format_duration $eval_duration)
    echo "[$end_eval_time] complete evaluate model --> ${dir} " | tee -a ${log_file}
    echo "Evaluation duration: ${formatted_eval_duration}" | tee -a ${log_file}

    # 记录评估结束时间和耗时
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Evaluation completed for ${dir} in ${formatted_eval_duration}" >> ${log_file}

}

evaluate(){
    model=$1
    num=$2

    dir=${model}_${data}_${UPDATE}_${batch_size}_${num}

    echo "${dir}"

    if [ -d "${output_dir}/${dir}" ]; then
      python ALLRec_evaluate.py --lora_weights ${output_dir}/${dir} \
                              --test_data_path ${test_data_path} \
                              --result_json_data ${result_json_data} \
                              --cutoff_len ${cutoff_len}
    else
      echo "Directory ${output_dir}/${dir} does not exist."
    fi
}

# 遍历不同的配置组合
run_loop(){
  for model in "${models[@]}"; do
    for i in "${nums[@]}"; do
      fine_tune_model $model $i
    done
  done
}


