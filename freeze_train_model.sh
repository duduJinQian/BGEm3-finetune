export DEBUG_MODE=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_data="\
   /root/autodl-tmp/2025-project/BGE/ana_data_2/all_samples.jsonl"

# set large epochs and small batch size for testing
num_train_epochs=10
per_device_train_batch_size=2

# set num_gpus to 2 for testing
num_gpus=1

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path /root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181 \
    --cache_dir /root/autodl-tmp/model/BGEm3_Law_free \
    --trust_remote_code True \
"

data_args="\
    --train_data $train_data \
    --cache_path  /root/autodl-tmp/2025-project/BGE/FlagEmbedding/examples/finetune/embedder\
    --train_group_size 6 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
"

training_args="\
    --output_dir /root/autodl-tmp/model/BGEm3_Law_new \
    --overwrite_output_dir \
    --learning_rate 5e-5 \
    --fp16 True\
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --negatives_cross_device False\
    --gradient_accumulation_steps 16 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --logging_steps 50\
    --save_strategy steps\
    --save_total_limit 2\
    --save_steps 1000 \
    --temperature 0.05 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --unified_finetuning False \
    --use_self_distill False \
    --fix_encoder False \
    --fix_position_embedding True \
"

# 更简单的方法：确保变量不包含换行符
model_args=$(echo "$model_args" | tr -d '\n')
data_args=$(echo "$data_args" | tr -d '\n')
training_args=$(echo "$training_args" | tr -d '\n')

# 将所有参数合并成一个单行字符串
full_cmd="torchrun --nproc_per_node $num_gpus -m FlagEmbedding.finetune.embedder.encoder_only.m3"
full_cmd="$full_cmd $(echo "$model_args" | tr -d '\n' | tr -s ' ')"
full_cmd="$full_cmd $(echo "$data_args" | tr -d '\n' | tr -s ' ')"
full_cmd="$full_cmd $(echo "$training_args" | tr -d '\n' | tr -s ' ')"

# 打印命令
echo "$full_cmd"

# 使用 nohup 执行命令
nohup bash -c "$full_cmd" > output.log 2>&1 &
