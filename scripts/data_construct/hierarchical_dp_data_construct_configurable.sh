#!/bin/bash

# ============================================================
# 层级分类推理数据构建脚本 - DP (Direct Predict) 策略 - 可配置版本
# ============================================================
# 直接预测策略：候选列表包含所有节点（叶子+非叶子）
# 适用场景：多粒度分类，GT可能是非叶子节点
# 
# 新增功能：
# - 支持超参数控制文档解析文本、文档页图片、文件名的使用
# - 根据配置动态调整prompt模板
# 
# 优点：
# - 支持多粒度GT
# - 简单直接，输出单个code
# - token消耗适中
# - 可灵活配置输入组件
# 
# 缺点：
# - 候选列表较长
# - 不输出完整路径
# ============================================================

export TASK_TYPE="hierarchicalDataConstruct"

# 基础配置
EXP_NAME="run_${TASK_TYPE}_dp_configurable"
EXP_NOTE="构建层级分类数据-DP直接预测策略-可配置版本"
SEED=42
DEVICE="cuda"

# ============================================================
# 策略配置
# ============================================================
STRATEGY="dp"

# DP策略特定参数
SHOW_DETAILS=true      # 是否显示类别详细信息（名称、描述）
SHOW_HIERARCHY=true    # 是否显示层级结构（缩进表示）

# ============================================================
# 输入组件配置（新增）
# ============================================================
# 控制prompt中包含哪些输入组件
USE_OCR_TEXT=true       # 是否使用文档解析文本
USE_IMAGE_PAGES=true    # 是否使用文档页图片
USE_FILE_NAME=false     # 是否使用文件名（默认关闭）

# ============================================================
# Taxonomy配置（新增）
# ============================================================
# 控制是否在prompt中包含分类体系信息
INCLUDE_TAXONOMY=true   # 是否包含分类体系（true=包含完整体系，false=让模型从训练中学习）

# ============================================================
# 数据配置
# ============================================================

# 输入数据（支持单文件或目录）
# 单文件模式
DATA_PATH="./datasets/test_demo.csv"
# 目录模式（注释掉DATA_PATH，使用DATA_DIR）
# DATA_DIR="/mnt/workspace/workgroup/yuqing/datas/docCls/processed/doc_3_0/file_classify_ds_20251126_augmented"

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
if [ "$USE_OCR_TEXT" = "true" ] && [ "$MASKED_TEXT_VALUE" = "true" ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_maskedTextValue"
fi
if [ "$USE_IMAGE_PAGES" = "true" ] && [ "$MASKED_IMAGE_VALUE" = "true" ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_maskedImageValue"
fi
if [ "$INCLUDE_TAXONOMY" = "false" ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_noTaxonomy"
fi
OUTPUT_DIR="./datasets/vlm_dp_infer_dataset"

# ============================================================
# 标签树配置（必需）
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
# - 逗号分隔的列名: 只保留指定列
KEEP_RAW_COLS="__all__"

# ============================================================
# 执行
# ============================================================

echo "============================================================"
echo "层级分类数据构建 - DP策略（直接预测）- 可配置版本"
echo "============================================================"
echo "策略: ${STRATEGY}"
echo "输入: ${DATA_PATH:-$DATA_DIR}"
echo "输出: ${OUTPUT_DIR}"
echo "显示详情: ${SHOW_DETAILS}"
echo "显示层级: ${SHOW_HIERARCHY}"
echo "============================================================"
echo "输入组件配置:"
echo "- 使用OCR文本: ${USE_OCR_TEXT}"
echo "- 使用图片页: ${USE_IMAGE_PAGES}"
echo "- 使用文件名: ${USE_FILE_NAME}"
echo "- 包含分类体系: ${INCLUDE_TAXONOMY}"
echo "============================================================"

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
if [ -n "${DATA_PATH}" ]; then
    ARGS="${ARGS} --data_path=${DATA_PATH}"
else
    ARGS="${ARGS} --data_dir=${DATA_DIR}"
fi

# 标签树配置
ARGS="${ARGS} --label_config_path=${LABEL_CONFIG_PATH}"
ARGS="${ARGS} --label_attr_name_map='${LABEL_ATTR_NAME_MAP}'"

# DP策略参数
ARGS="${ARGS} --show_details=${SHOW_DETAILS}"
ARGS="${ARGS} --show_hierarchy=${SHOW_HIERARCHY}"

# 输入组件配置参数（新增）
ARGS="${ARGS} --use_ocr_text=${USE_OCR_TEXT}"
ARGS="${ARGS} --use_image_pages=${USE_IMAGE_PAGES}"
ARGS="${ARGS} --use_file_name=${USE_FILE_NAME}"

# Taxonomy配置参数（新增）
ARGS="${ARGS} --include_taxonomy=${INCLUDE_TAXONOMY}"

# 原始数据保留
if [ -n "${KEEP_RAW_COLS}" ]; then
    ARGS="${ARGS} --keep_raw_cols=${KEEP_RAW_COLS}"
fi

echo "执行命令:"
echo "python src/data_process/hierarchical_data_construct.py ${ARGS}"
echo ""

# 调试模式（取消注释以启用）
# eval "python -m debugpy --listen 16777 --wait-for-client src/data_process/hierarchical_data_construct.py ${ARGS}"

eval "python src/data_process/hierarchical_data_construct.py ${ARGS}"