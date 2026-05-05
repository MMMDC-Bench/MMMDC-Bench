#!/bin/bash
set +x

# ============================================================
# 层级文档分类推理脚本 - 基于大模型API（网关调用）
# ============================================================
# 通过大模型API（如 Qwen-VL-Max）进行层级文档分类推理
# 
# 适用于所有层级分类Prompt策略生成的数据集:
# - DP: 直接预测（多粒度）
# - DL: 直接叶节点预测  
# - DH: 直接层级预测（路径输出）
# - TMH: 自顶向下多步预测
# - DH-CoT: 带推理的层级预测
# - Few-Shot: 带示例的层级分类
# - Retrieval: 检索增强的层级分类
#
# 与 mdl_hierarchical_doc_cls_infer_on_vlm.sh 的区别:
# - MDL版本: 加载本地模型进行推理（需要GPU集群）
# - 本版本: 通过网关调用大模型API（无需本地GPU）
#
# 用法:
#   bash scripts/llm/infer/run_hierarchical_doc_cls_infer_on_llm_api.sh
# ============================================================

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Working directory: $(pwd)"

# ============================================================
# 日志配置
# ============================================================
# 日志级别: DEBUG, INFO, WARNING, ERROR
export LOGURU_LEVEL="INFO"

# ============================================================
# 基础配置
# ============================================================

# 模型配置
# 模型名由目标API服务端识别
MODEL_NAME="qwen3-vl-235b-a22b-instruct"
API_PROTOCOL="openai"   # 可选: openai, anthropic
BASE_URL="https://your-api-endpoint/v1"
API_KEY="${API_KEY:-}"

# 并发配置
MAX_WORKERS=4          # 最大并发请求数
BATCH_SIZE=5          # 批处理大小（每批保存一次）
MAX_RETRIES=3          # 单次请求最大重试次数
RETRY_DELAY=5.0        # 重试间隔（秒）

# ============================================================
# 数据配置（修改此处以适配不同策略的数据）
# ============================================================

# 策略类型：dp, dl, dh, tmh, dh_cot, few_shot, retrieval_candidates_precomputed
STRATEGY="retrieval_candidates_precomputed"

# 数据根目录
DATA_ROOT="/mnt/workspace/workgroup/yuqing/datas/docCls/processed/doc_paper/synth_ei_doc_v0"
DATASET_NAME="vlm_retrieval_candidates_precomputed_synth_ei_doc_v0"
DATASET_TYPE="test_wTop10_wGT"
# DATASET_NAME=rvl_cdip
# DATASET_TYPE=sample_test

# 输入数据文件（根据策略自动设置，也可手动指定）
# 自动路径模式
INPUT="${DATA_ROOT}/${DATASET_NAME}/${DATASET_TYPE}.json"
# 手动指定模式（取消注释以覆盖自动路径）
# INPUT="/data/oss_bucket_0/yuqing/data/dataset/docCls/doc_3_0/sft_hierarchical_dp/test.json"

# 输出目录
OUTPUT_ROOT="./outputs/predict_results"
OUTPUT="${OUTPUT_ROOT}/${DATASET_NAME}/${DATASET_TYPE}_${MODEL_NAME}_predicted_api.json"

# ============================================================
# 字段配置
# ============================================================

# SFT数据格式字段
PROMPT_COLUMN="instruction"    # Prompt字段名
IMAGE_COLUMN="images"          # 图片字段名
OUTPUT_COLUMN="output"         # 真实标签字段名
PREDICT_COLUMN="predict"       # 预测结果字段名

# ============================================================
# 运行模式配置
# ============================================================

# 断点续推（启用后会跳过已处理的样本）
# RESUME_FLAG=""
RESUME_FLAG="--resume"

# 调试模式（只处理少量样本）
DEBUG_FLAG=""
DEBUG_SAMPLES=10
# DEBUG_FLAG="--debug --debug_samples=${DEBUG_SAMPLES}"

# 详细输出模式
VERBOSE_FLAG=""
# VERBOSE_FLAG="--verbose"

# ============================================================
# 分片配置（用于并行推理，可选）
# ============================================================

# 如需分片推理，设置以下变量
# 例如：4个进程并行，分别设置 SHARD_ID=0,1,2,3, NUM_SHARDS=4
SHARD_ID=""
NUM_SHARDS=""

# 构建分片参数
SHARD_ARGS=""
if [ -n "${SHARD_ID}" ] && [ -n "${NUM_SHARDS}" ]; then
    SHARD_ARGS="--shard_id=${SHARD_ID} --num_shards=${NUM_SHARDS}"
fi

# ============================================================
# 检查配置
# ============================================================

# 检查输入文件
if [ ! -f "${INPUT}" ]; then
    echo "错误: 输入文件不存在: ${INPUT}"
    echo "请检查数据路径配置"
    exit 1
fi

# 检查API配置
if [ -z "${BASE_URL}" ]; then
    echo "错误: BASE_URL 不能为空"
    exit 1
fi

if [ -z "${API_KEY}" ]; then
    echo "错误: API_KEY 不能为空（可通过环境变量 API_KEY 传入）"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR=$(dirname "${OUTPUT}")
mkdir -p "${OUTPUT_DIR}"

# ============================================================
# 执行推理
# ============================================================

echo "============================================================"
echo "层级文档分类推理 - ${STRATEGY} 策略（API模式）"
echo "============================================================"
echo "模型: ${MODEL_NAME}"
echo "输入: ${INPUT}"
echo "输出: ${OUTPUT}"
echo "并发数: ${MAX_WORKERS}"
echo "批大小: ${BATCH_SIZE}"
echo "============================================================"

# 构建完整命令
CMD="python src/llm_infer/llm_api_hierarchical_infer.py \
    --input=\"${INPUT}\" \
    --output=\"${OUTPUT}\" \
    --model=\"${MODEL_NAME}\" \
    --base_url=\"${BASE_URL}\" \
    --api_key=\"${API_KEY}\" \
    --api_protocol=\"${API_PROTOCOL}\" \
    --max_workers=${MAX_WORKERS} \
    --batch_size=${BATCH_SIZE} \
    --max_retries=${MAX_RETRIES} \
    --retry_delay=${RETRY_DELAY} \
    --prompt_column=\"${PROMPT_COLUMN}\" \
    --image_column=\"${IMAGE_COLUMN}\" \
    --output_column=\"${OUTPUT_COLUMN}\" \
    --predict_column=\"${PREDICT_COLUMN}\" \
    ${RESUME_FLAG} \
    ${DEBUG_FLAG} \
    ${VERBOSE_FLAG} \
    ${SHARD_ARGS}"

echo ""
echo "执行命令:"
echo "${CMD}"
echo ""

# 执行推理
eval ${CMD}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "推理完成！"
    echo "结果保存在: ${OUTPUT}"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "推理失败，退出码: ${EXIT_CODE}"
    echo "============================================================"
    exit ${EXIT_CODE}
fi
