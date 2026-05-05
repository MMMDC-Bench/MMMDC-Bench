
#!/bin/bash

# ============================================================
# Taxonomy Adapt 启动脚本（MMMDC-Bench 迁移版）
# ============================================================
# 迁移自 DocTaxoAdapt/scripts/run_cluster_based_example.sh
# 主入口:
#   src/taxonomy_adpt/enterprise_main.py
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# -----------------------------
# 输入输出配置
# -----------------------------
INPUT_TABLE="./datasets/test_demo.csv"
TITLE_COL="file_name"
CONTENT_COL="ocr_text"
IMAGE_COL="image_path"              # 本地图片路径列
CLUSTER_LABEL_COL="cluster_labels"  # 若无该列，会自动回退到非分组扩展

OUTPUT_DIR="./outputs/taxonomy_adapt"
IMPORT_TAXONOMY=""  # 例如: ./outputs/taxonomy_adapt/taxonomy_structure.json
EXPORT_TAXONOMY="${OUTPUT_DIR}/taxonomy_structure.json"

# -----------------------------
# 任务与模型配置
# -----------------------------
TOPIC="企业文档"
DIMENSIONS="doc_type"
LLM="gpt"
MODEL_NAME="qwen3-vl-235b-a22b-instruct"
API_PROTOCOL="openai"  # openai / anthropic / dashscope
BASE_URL="${BASE_URL:-https://your-api-endpoint/v1}"
API_KEY="${API_KEY:-}"

# -----------------------------
# 扩展控制参数
# -----------------------------
MAX_DEPTH=10
MAX_DENSITY=3
USE_CLUSTER_BASED_EXPANSION=true
USE_INTERPRETABLE_EXPANSION=false
ENABLE_SCHEMA_ENRICHMENT=true
RESUME=false

mkdir -p "${OUTPUT_DIR}"

if [ -z "${BASE_URL}" ]; then
  echo "错误: BASE_URL 不能为空"
  exit 1
fi

if [ -z "${API_KEY}" ]; then
  echo "错误: API_KEY 不能为空（请通过环境变量 API_KEY 或脚本内变量设置）"
  exit 1
fi

CMD="python src/taxonomy_adpt/enterprise_main.py \
  --input_table \"${INPUT_TABLE}\" \
  --title_col \"${TITLE_COL}\" \
  --content_col \"${CONTENT_COL}\" \
  --image_col \"${IMAGE_COL}\" \
  --cluster_label_col \"${CLUSTER_LABEL_COL}\" \
  --output_dir \"${OUTPUT_DIR}\" \
  --topic \"${TOPIC}\" \
  --dimensions ${DIMENSIONS} \
  --llm \"${LLM}\" \
  --model_name \"${MODEL_NAME}\" \
  --api_protocol \"${API_PROTOCOL}\" \
  --base_url \"${BASE_URL}\" \
  --api_key \"${API_KEY}\" \
  --max_depth ${MAX_DEPTH} \
  --max_density ${MAX_DENSITY} \
  --export_taxonomy \"${EXPORT_TAXONOMY}\""

if [ "${USE_CLUSTER_BASED_EXPANSION}" = "true" ]; then
  CMD="${CMD} --use_cluster_based_expansion"
else
  CMD="${CMD} --no_cluster_based_expansion"
fi

if [ "${USE_INTERPRETABLE_EXPANSION}" = "true" ]; then
  CMD="${CMD} --use_interpretable_expansion"
else
  CMD="${CMD} --no_interpretable_expansion"
fi

if [ "${ENABLE_SCHEMA_ENRICHMENT}" = "true" ]; then
  CMD="${CMD} --enable_schema_enrichment"
fi

if [ "${RESUME}" = "true" ]; then
  CMD="${CMD} --resume"
fi

if [ -n "${IMPORT_TAXONOMY}" ]; then
  CMD="${CMD} --import_taxonomy \"${IMPORT_TAXONOMY}\""
fi

echo "============================================================"
echo "Taxonomy Adapt (MMMDC-Bench)"
echo "输入: ${INPUT_TABLE}"
echo "输出: ${OUTPUT_DIR}"
echo "执行命令:"
echo "${CMD}"
echo "============================================================"

eval "${CMD}"
