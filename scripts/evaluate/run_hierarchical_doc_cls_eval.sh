#!/bin/bash
# ============================================================
# 层级文档分类评测脚本
# 
# 支持所有层级分类策略的全面评测：
# - DP: 直接预测（多粒度）
# - DL: 直接叶节点预测
# - DH: 直接层级预测（路径输出）
# - TMH: 自顶向下多步预测
# - DH-CoT: 带推理的层级预测
# - Few-Shot: 带示例的层级分类
#
# 说明:
#   推理结果支持两种格式：
#   1. 单文件: test_predicted_base.json
#   2. 分片文件(并行推理): test_predicted_base.json_0, test_predicted_base.json_1, ...
#   脚本会自动检测并合并分片文件
#
# 用法:
#   bash scripts/evaluate/run_hierarchical_doc_cls_eval.sh
# ============================================================

set -e

# ============================================================
# 配置参数（根据实际情况修改）
# ============================================================

# 策略类型
STRATEGY="dp"  # dp, dl, dh, tmh, dh_cot, few_shot, retrieval_candidates_precomputed

# TMH 聚合评估（按 sample_id 聚合，链式准确率）
# 设为 true 时，会按原始样本聚合多步预测结果，中间出错则整体错误
TMH_AGGREGATE=false

# 评测任务类型
TASK="hierarchical_classify"

# 推理结果路径配置

PREDICT_ROOT="./outputs/predict_results"
CONCISE_DATASET_NAME="SynthEIDocV0"
DATASET_NAME="vlm_dp_synth_ei_doc_v0_noTaxonomy"

# 推理结果目录（包含分片文件）
# PREDICT_DIR="${PREDICT_ROOT}/${DATASET_NAME}/${DATASET_TYPE}"
PREDICT_DIR="${PREDICT_ROOT}/${DATASET_NAME}"
# 文件名前缀（用于匹配分片文件）
MODEL_NAME="qwenw3_VL_8B_text_image_lora_sft"
# PREDICT_FILE_PREFIX="test_with_10DocRetrieveCandidates_${MODEL_NAME}_predicted_base.json"
PREDICT_FILE_PREFIX="test_qwenw3_VL_8B_lora_sft_text_image_predicted.json"

# 评测结果保存目录
EVAL_ROOT="outputs/eval_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_NAME="${CONCISE_DATASET_NAME}_${STRATEGY}_${MODEL_NAME}_eval"
SAVE_DIR="${EVAL_ROOT}/${EVAL_NAME}"

# 标签配置文件（用于层级评测）
LABEL_CONFIG_PATH="./cache/taxonomy652/doc_map.json"
# LABEL_CONFIG_PATH="./cache/rvl_cdip_taxonomy/doc_map.json"

# 字段名配置
OUTPUT_FIELD="output"      # 真实标签字段
PREDICT_FIELD="predict"    # 预测结果字段

# 标签配置中的字段名
LABEL_CODE_KEY="label_code"
LABEL_NAME_KEY="label_name"  
LABEL_PARENT_KEY="label_parent"

# 路径分隔符（用于DH/DH-CoT/Few-Shot策略的路径解析）
PATH_SEPARATOR=" > "

# 是否区分大小写
CASE_SENSITIVE=""  # 设为 "--case_sensitive" 启用大小写敏感

# ============================================================
# 检查输入路径
# ============================================================

if [ ! -d "$PREDICT_DIR" ]; then
    echo "错误: 推理结果目录不存在: $PREDICT_DIR"
    exit 1
fi

if [ ! -f "$LABEL_CONFIG_PATH" ]; then
    echo "错误: 标签配置文件不存在: $LABEL_CONFIG_PATH"
    exit 1
fi

# 检查是否有匹配的文件
PREDICT_FILES=$(find "$PREDICT_DIR" -maxdepth 1 -name "${PREDICT_FILE_PREFIX}*" -type f 2>/dev/null | sort)
if [ -z "$PREDICT_FILES" ]; then
    echo "错误: 未找到匹配的推理结果文件: ${PREDICT_DIR}/${PREDICT_FILE_PREFIX}*"
    exit 1
fi

# 统计文件数量
FILE_COUNT=$(echo "$PREDICT_FILES" | wc -l | tr -d ' ')

# ============================================================
# 执行评测
# ============================================================

echo "============================================================"
echo "  层级文档分类评测 - ${STRATEGY} 策略"
echo "============================================================"
echo "  策略类型: ${STRATEGY}"
echo "  推理结果目录: ${PREDICT_DIR}"
echo "  文件名前缀: ${PREDICT_FILE_PREFIX}"
echo "  匹配文件数: ${FILE_COUNT}"
echo "  保存目录: ${SAVE_DIR}"
echo "  标签配置: ${LABEL_CONFIG_PATH}"
echo "============================================================"

# 列出匹配的文件
echo ""
echo "匹配的推理结果文件:"
echo "$PREDICT_FILES" | while read f; do echo "  - $(basename "$f")"; done
echo ""

# 构建 TMH 聚合参数
TMH_AGGREGATE_FLAG=""
if [ "${TMH_AGGREGATE}" = "true" ] && [ "${STRATEGY}" = "tmh" ]; then
    TMH_AGGREGATE_FLAG="--tmh_aggregate"
    echo "启用 TMH 聚合评估模式（按原始样本聚合，链式准确率）"
    echo ""
fi

# python -m debugpy --listen 16777 --wait-for-client src/evaluate/offline_evaluate.py \
python src/evaluate/offline_evaluate.py \
    --task="hierarchical_classify" \
    --predict_dir="${PREDICT_DIR}" \
    --predict_file_prefix="${PREDICT_FILE_PREFIX}" \
    --save_dir="${SAVE_DIR}" \
    --response="${OUTPUT_FIELD}" \
    --predict_field="${PREDICT_FIELD}" \
    --label_config_path="${LABEL_CONFIG_PATH}" \
    --label_code_key="${LABEL_CODE_KEY}" \
    --label_name_key="${LABEL_NAME_KEY}" \
    --label_parent_key="${LABEL_PARENT_KEY}" \
    --strategy="${STRATEGY}" \
    --path_separator="${PATH_SEPARATOR}" \
    ${CASE_SENSITIVE} \
    ${TMH_AGGREGATE_FLAG}

echo ""
echo "评测完成！结果保存在: ${SAVE_DIR}"