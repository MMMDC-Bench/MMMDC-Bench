#!/bin/bash

# ============================================================
# 层级分类SFT数据构建脚本 - RETRIEVAL_CANDIDATES_PRECOMPUTED 策略 - 可配置版本
# ============================================================
# 预计算检索候选策略：从已计算好的候选标签列中读取候选，构建SFT数据
# 
# 数据来源：
# - 使用 retrieval_candidates_augment.sh 脚本预先生成的候选标签数据
# - 候选列中只包含 label_code，taxonomy 信息（name, desc）从 LabelTree 补充
#
# 新增功能：
# - 支持超参数控制文档解析文本、文档页图片、文件名的使用
# - 根据配置动态调整prompt模板
# - 智能输出目录命名
#
# 适用场景：
# - 需要缩小候选范围的文档分类任务
# - 利用检索结果辅助分类决策
# - 训练数据已预先进行检索增强
# 
# 优点：
# - 不需要实时调用检索服务，构建速度快
# - 候选已预计算，可以离线批量处理
# - 支持多种候选格式（label_code列表、带分数字典等）
# - 可灵活配置输入组件
# 
# 缺点：
# - 需要预先运行检索增强脚本准备数据
# - 候选结果是静态的，无法动态更新
# ============================================================

export TASK_TYPE="hierarchicalDataConstruct"

# 基础配置
EXP_NAME="run_${TASK_TYPE}_retrieval_candidates_precomputed_configurable"
EXP_NOTE="构建层级分类SFT数据-预计算检索候选策略-可配置版本"
SEED=42
DEVICE="cuda"

# ============================================================
# 策略配置
# ============================================================
STRATEGY="retrieval_candidates_precomputed"

# ============================================================
# 输入组件配置（新增）
# ============================================================
# 控制prompt中包含哪些输入组件
USE_OCR_TEXT=true       # 是否使用文档解析文本
USE_IMAGE_PAGES=false    # 是否使用文档页图片
USE_FILE_NAME=false     # 是否使用文件名（默认关闭）

# ============================================================
# 数据配置
# ============================================================

# 输入数据（带候选标签列的数据，由 retrieval_candidates_augment.sh 生成）
DATA_PATH="./datasets/test_demo.csv"

# 输出目录（包含配置信息）
OUTPUT_SUFFIX=""
if [ "$USE_OCR_TEXT" = "false" ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_noText"
fi
if [ "$USE_IMAGE_PAGES" = "false" ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_noImg"
fi
if [ "$USE_FILE_NAME" = "true" ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_fname"
fi
OUTPUT_DIR="./datasets/vlm_${STRATEGY}${OUTPUT_SUFFIX}"

# ============================================================
# 检索候选策略特定参数
# ============================================================

# 候选标签列名（retrieval_candidates_augment.sh 输出的列名）
CANDIDATE_COL="candidate_labels"

# 是否确保GT在候选列表中（训练时推荐设为true）
INCLUDE_GT_IN_CANDIDATES=true

# 是否在prompt中显示相关度分数
SHOW_SCORES=true

# ============================================================
# 标签树配置（必需，用于补充taxonomy信息）
# ============================================================
LABEL_CONFIG_PATH="./datasets/document_taxonomy.json"
LABEL_ATTR_NAME_MAP='{"label_code":"label_code","label_name":"label_name","label_desc":"label_desc","label_parent":"label_parent"}'

# ============================================================
# 列名映射
# ============================================================
COL_MAP='{"ocr_text":"ocr_text","image_path":"image_path","file_name":"file_name","file_type":"file_type"}'

# ============================================================
# 原始数据保留配置
# ============================================================
# 需要保留的原始数据列，会添加到输出的 "raw" 字段中
# - "__all__": 保留所有原始列
# - 逗号分隔的列名: 只保留指定列，如 "file_name,scene_code,file_type"
KEEP_RAW_COLS="__all__"

# ============================================================
# 执行
# ============================================================

echo "============================================================"
echo "层级分类SFT数据构建 - 预计算检索候选策略 - 可配置版本"
echo "============================================================"
echo "策略: ${STRATEGY}"
echo "输入: ${DATA_PATH}"
echo "输出: ${OUTPUT_DIR}"
echo "候选列: ${CANDIDATE_COL}"
echo "包含GT: ${INCLUDE_GT_IN_CANDIDATES}"
echo "显示分数: ${SHOW_SCORES}"
echo "标签配置: ${LABEL_CONFIG_PATH}"
echo "============================================================"
echo "输入组件配置:"
echo "- 使用OCR文本: ${USE_OCR_TEXT}"
echo "- 使用图片页: ${USE_IMAGE_PAGES}"
echo "- 使用文件名: ${USE_FILE_NAME}"
echo "============================================================"

# 检查输入文件是否存在
if [ ! -f "${DATA_PATH}" ]; then
    echo "错误: 输入文件不存在: ${DATA_PATH}"
    echo "请先运行 retrieval_candidates_augment.sh 生成候选数据"
    exit 1
fi

# 检查标签配置文件是否存在
if [ ! -f "${LABEL_CONFIG_PATH}" ]; then
    echo "错误: 标签配置文件不存在: ${LABEL_CONFIG_PATH}"
    exit 1
fi

# 构建命令参数
ARGS="--task_type=${TASK_TYPE}"
ARGS="${ARGS} --exp_name=${EXP_NAME}"
ARGS="${ARGS} --exp_note='${EXP_NOTE}'"
ARGS="${ARGS} --seed=${SEED}"
ARGS="${ARGS} --device=${DEVICE}"
ARGS="${ARGS} --strategy=${STRATEGY}"
ARGS="${ARGS} --col_map='${COL_MAP}'"
ARGS="${ARGS} --output_dir=${OUTPUT_DIR}"

# 数据路径
ARGS="${ARGS} --data_path=${DATA_PATH}"

# 标签树配置
ARGS="${ARGS} --label_config_path=${LABEL_CONFIG_PATH}"
ARGS="${ARGS} --label_attr_name_map='${LABEL_ATTR_NAME_MAP}'"

# 检索候选策略参数
ARGS="${ARGS} --candidate_col=${CANDIDATE_COL}"
ARGS="${ARGS} --include_gt_in_candidates=${INCLUDE_GT_IN_CANDIDATES}"
ARGS="${ARGS} --show_scores=${SHOW_SCORES}"

# 输入组件配置参数（新增）
ARGS="${ARGS} --use_ocr_text=${USE_OCR_TEXT}"
ARGS="${ARGS} --use_image_pages=${USE_IMAGE_PAGES}"
ARGS="${ARGS} --use_file_name=${USE_FILE_NAME}"

# 原始数据保留
if [ -n "${KEEP_RAW_COLS}" ]; then
    ARGS="${ARGS} --keep_raw_cols=${KEEP_RAW_COLS}"
fi

echo ""
echo "执行命令:"
echo "python src/data_process/hierarchical_data_construct.py ${ARGS}"
echo ""

# 调试模式（取消注释以启用）
# eval "python -m debugpy --listen 16777 --wait-for-client src/data_process/hierarchical_data_construct.py ${ARGS}"

eval "python src/data_process/hierarchical_data_construct.py ${ARGS}"

echo ""
echo "============================================================"
echo "处理完成！"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================================"
echo ""
echo "下一步："
echo "1. 检查生成的SFT数据文件"
echo "2. 使用该数据微调VLM模型"
echo ""
echo "配置说明："
echo "- 通过修改 USE_OCR_TEXT、USE_IMAGE_PAGES、USE_FILE_NAME 控制输入组件"
echo "- 输出目录会根据配置自动添加后缀便于区分"
echo "============================================================"