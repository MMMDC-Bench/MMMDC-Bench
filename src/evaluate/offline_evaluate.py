#!/usr/bin/env python
"""
离线评测入口脚本

基于已有的推理结果进行评测，无需加载模型。

用法：
    python src/tuning_factory/offline_evaluate.py \
        --task offline_classify \
        --predict_dir outputs/predict_results/demo_exp_predict_results \
        --save_dir outputs/eval_results/demo_exp_offline_eval \
        --response output \
        --predict_field predict
"""
from offline_eval_core import run_offline_eval


def main():
    run_offline_eval()


if __name__ == "__main__":
    main()
