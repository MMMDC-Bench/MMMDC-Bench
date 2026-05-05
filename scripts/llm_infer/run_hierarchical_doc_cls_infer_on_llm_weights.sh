

MODEL_NAME="/mnt/workspace/workgroup/yuqing/models/Qwen/Qwen3.5-9B"
# ADAPTER_PATH="/mnt/workspace/workgroup/yuqing/checkpoints/doc_cls/qwen3.5_9B_sft_lora_train_with_doc_3_0_20251126_augmented"
BATCH_SIZE=1

DATA_PATH="/mnt/workspace/workgroup/yuqing/datas/docCls/processed/doc_paper/synth_ei_doc_v0/vlm_retrieval_candidates_precomputed_synth_ei_doc_v0/test_wTop10_wGT.json"
PROMPT_TEMPLATE="qwen3_5_nothink"
OUTPUT_PATH="outputs/predict_results/qwen3.5_9B_infer/predicted.json"


# --adapter_name_or_path=$ADAPTER_PATH \
# deepspeed --num_gpus=1 src/tuning_factory/evaluate.py \
# python -m debugpy --listen 16777 --wait-for-client src/tuning_factory/evaluate.py \
python src/tuning_factory/generate.py \
    --model_name_or_path=$MODEL_NAME \
    --inputs=$DATA_PATH \
    --batch_size=$BATCH_SIZE \
    --prompt_column=instruction \
    --image=images \
    --template=$PROMPT_TEMPLATE \
    --image_processor=True \
    --image_resolution=262144 \
    --outputs=$OUTPUT_PATH \
    --cutoff_len=16384 \
    --max_new_tokens=256