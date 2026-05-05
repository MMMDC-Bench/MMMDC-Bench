#!/usr/bin/env python
"""
离线评测核心逻辑（单文件版）。

目标：
- 不依赖 llmtuner/tf_ext 等历史目录结构
- 提供统一参数解析入口
- 支持 offline_classify / hierarchical_classify
- 支持策略：dp, dl, dh, tmh, dh_cot, few_shot
- 支持 TMH 链式聚合评测（--tmh_aggregate）
"""

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class OfflineEvalDataArgs:
    response: str = field(default="output")
    prompt: Optional[str] = field(default=None)
    query: Optional[str] = field(default=None)


@dataclass
class OfflineEvalArgs:
    task: str = field(default="offline_classify")
    predict_dir: str = field(default="")
    predict_file_prefix: Optional[str] = field(default=None)
    save_dir: Optional[str] = field(default=None)
    predict_field: str = field(default="predict")
    case_sensitive: bool = field(default=False)
    label_config_path: Optional[str] = field(default=None)
    label_code_key: str = field(default="label_code")
    label_name_key: str = field(default="label_name")
    label_parent_key: str = field(default="parent_label_code")
    strategy: str = field(default="dp")
    path_separator: str = field(default=" > ")
    tmh_aggregate: bool = field(default=False)


def parse_offline_eval_args() -> Tuple[OfflineEvalDataArgs, OfflineEvalArgs]:
    parser = argparse.ArgumentParser(
        description="Offline evaluation based on existing prediction results."
    )
    parser.add_argument("--task", type=str, default="offline_classify")
    parser.add_argument("--predict_dir", type=str, required=True)
    parser.add_argument("--predict_file_prefix", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--predict_field", type=str, default="predict")
    parser.add_argument("--case_sensitive", action="store_true", default=False)

    parser.add_argument("--response", type=str, default="output")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)

    parser.add_argument("--label_config_path", type=str, default=None)
    parser.add_argument("--label_code_key", type=str, default="label_code")
    parser.add_argument("--label_name_key", type=str, default="label_name")
    parser.add_argument("--label_parent_key", type=str, default="parent_label_code")

    parser.add_argument("--strategy", type=str, default="dp")
    parser.add_argument("--path_separator", type=str, default=" > ")
    parser.add_argument("--tmh_aggregate", action="store_true", default=False)

    args = parser.parse_args()

    data_args = OfflineEvalDataArgs(
        response=args.response,
        prompt=args.prompt,
        query=args.query,
    )
    eval_args = OfflineEvalArgs(
        task=args.task,
        predict_dir=args.predict_dir,
        predict_file_prefix=args.predict_file_prefix,
        save_dir=args.save_dir,
        predict_field=args.predict_field,
        case_sensitive=args.case_sensitive,
        label_config_path=args.label_config_path,
        label_code_key=args.label_code_key,
        label_name_key=args.label_name_key,
        label_parent_key=args.label_parent_key,
        strategy=args.strategy,
        path_separator=args.path_separator,
        tmh_aggregate=args.tmh_aggregate,
    )
    return data_args, eval_args


class LabelNode:
    def __init__(self, code: str, name: str, parent: str, description: str = ""):
        self.code = code
        self.name = name
        self.parent = parent
        self.description = description
        self.children: List[str] = []
        self.level: int = -1


class LabelTree:
    def __init__(
        self,
        label_config_path: str,
        label_code_key: str = "label_code",
        label_name_key: str = "label_name",
        label_parent_key: str = "parent_label_code",
    ):
        self.max_level = -1
        self.min_level = 1
        self.label_code_key = label_code_key
        self.label_name_key = label_name_key
        self.label_parent_key = label_parent_key
        self.label_roots, self.label_code_to_node = self._build_tree(label_config_path)
        self.name_to_code = {}
        for code, node in self.label_code_to_node.items():
            if node.name:
                self.name_to_code[node.name.lower()] = code
        self._detect_min_level()

    def _build_tree(self, label_config_path: str):
        with open(label_config_path, "r", encoding="utf-8") as f:
            label_config = json.load(f)

        nodes: Dict[str, LabelNode] = {}
        roots: List[LabelNode] = []

        for label_key, label_attrs in label_config.items():
            label_code = str(label_attrs.get(self.label_code_key, label_key)).strip()
            label_name = str(label_attrs.get(self.label_name_key, "")).strip()
            label_parent = str(label_attrs.get(self.label_parent_key, "")).strip()
            label_desc = str(label_attrs.get("label_desc", "")).strip()
            node = LabelNode(label_code, label_name, label_parent, label_desc)
            nodes[label_code] = node
            if (not label_parent) or (label_parent == label_code):
                roots.append(node)

        for label_code, node in nodes.items():
            if node.parent and node.parent != label_code and node.parent in nodes:
                nodes[node.parent].children.append(label_code)

        self._recover_level(roots, nodes)
        return roots, nodes

    def _recover_level(self, roots: List[LabelNode], nodes: Dict[str, LabelNode]):
        container = roots
        level = 1
        while container:
            next_level_container: List[LabelNode] = []
            for node in container:
                node.level = level
                next_level_container.extend([nodes[child] for child in node.children])
            container = next_level_container
            level += 1
        self.max_level = level - 1

    def _detect_min_level(self):
        if len(self.label_roots) == 1 and self.label_roots[0].children:
            root = self.label_roots[0]
            root_name = root.name.lower() if root.name else ""
            root_code = root.code.lower() if root.code else ""
            virtual_root_indicators = {"root", "all", "总", "全部", ""}
            if root_name in virtual_root_indicators or root_code in virtual_root_indicators:
                self.min_level = 2
                return
            if not root_name or root_name == root_code:
                self.min_level = 2
                return
        self.min_level = 1

    def get_effective_level(self, level: int) -> int:
        return level - self.min_level + 1

    def get_effective_max_level(self) -> int:
        return self.max_level - self.min_level + 1

    def find_label_path(self, label_code: str, exclude_virtual_root: bool = True) -> Dict[int, str]:
        label_node = self.label_code_to_node.get(label_code)
        if not label_node:
            return {}

        raw_path = {label_node.level: label_code}
        while label_node.parent and label_node.parent != label_node.code:
            parent_code = label_node.parent
            parent_node = self.label_code_to_node.get(parent_code)
            if not parent_node:
                break
            raw_path[parent_node.level] = parent_code
            label_node = parent_node

        if not exclude_virtual_root:
            return dict(sorted(raw_path.items(), key=lambda x: x[0]))

        effective_path = {}
        for level, code in raw_path.items():
            if level >= self.min_level:
                effective_path[self.get_effective_level(level)] = code
        return dict(sorted(effective_path.items(), key=lambda x: x[0]))

    def find_label_path_list(self, label_code: str, exclude_virtual_root: bool = True) -> List[str]:
        path_dict = self.find_label_path(label_code, exclude_virtual_root=exclude_virtual_root)
        return [path_dict[level] for level in sorted(path_dict.keys())]

    def get_label_level(self, label_code: str, effective: bool = True) -> int:
        node = self.label_code_to_node.get(label_code)
        if not node:
            return -1
        return self.get_effective_level(node.level) if effective else node.level

    def get_lca(self, code1: str, code2: str) -> Optional[str]:
        path1 = set(self.find_label_path_list(code1))
        path2 = self.find_label_path_list(code2)
        for code in reversed(path2):
            if code in path1:
                return code
        return None

    def resolve_label(self, text: str) -> Optional[str]:
        text = text.strip()
        if text in self.label_code_to_node:
            return text
        lowered = text.lower()
        if lowered in self.name_to_code:
            return self.name_to_code[lowered]

        match = re.match(r"^([^\s(]+)", text)
        if match:
            possible_code = match.group(1).strip()
            if possible_code in self.label_code_to_node:
                return possible_code
        return None


class HierarchicalClassifyEvaluator:
    STRATEGY_DP = "dp"
    STRATEGY_DL = "dl"
    STRATEGY_DH = "dh"
    STRATEGY_TMH = "tmh"
    STRATEGY_DH_COT = "dh_cot"
    STRATEGY_FEW_SHOT = "few_shot"
    PATH_OUTPUT_STRATEGIES = {STRATEGY_DH, STRATEGY_DH_COT, STRATEGY_FEW_SHOT}

    def __init__(self, label_tree: LabelTree, strategy: str = "dp", path_separator: str = " > ", case_sensitive: bool = False):
        self.label_tree = label_tree
        self.strategy = strategy.lower()
        self.path_separator = path_separator
        self.case_sensitive = case_sensitive

    def _compare(self, a: Optional[str], b: Optional[str]) -> bool:
        if a is None or b is None:
            return False
        if self.case_sensitive:
            return a == b
        return a.lower() == b.lower()

    def _parse_direct_output(self, pred_text: str) -> Tuple[Optional[str], List[str]]:
        code = self.label_tree.resolve_label(pred_text)
        if code:
            return code, self.label_tree.find_label_path_list(code)
        return (pred_text if pred_text else None, [pred_text] if pred_text else [])

    def _complete_path_prefix(self, path: List[str]) -> List[str]:
        if not path:
            return path
        full_path_of_first = self.label_tree.find_label_path_list(path[0], exclude_virtual_root=True)
        if not full_path_of_first:
            return path
        prefix_len = len(full_path_of_first) - 1
        if prefix_len > 0:
            return full_path_of_first[:-1] + path
        return path

    def _parse_path_output(self, pred_text: str) -> Tuple[Optional[str], List[str]]:
        parts = [p.strip() for p in pred_text.split(self.path_separator)]
        path: List[str] = []
        for part in parts:
            if not part:
                continue
            code = self.label_tree.resolve_label(part)
            if code:
                path.append(code)
        if not path:
            return (pred_text if pred_text else None, [pred_text] if pred_text else [])
        path = self._complete_path_prefix(path)
        return path[-1], path

    def _parse_cot_output(self, pred_text: str) -> Tuple[Optional[str], List[str]]:
        match = re.search(r"最终分类路径[：:]\s*(.+?)(?:\n|$)", pred_text)
        if match:
            return self._parse_path_output(match.group(1).strip())
        match = re.search(r"Final[：:]?\s*(.+?)(?:\n|$)", pred_text, re.IGNORECASE)
        if match:
            return self._parse_path_output(match.group(1).strip())
        lines = pred_text.strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            if self.path_separator in last_line:
                return self._parse_path_output(last_line)
        return None, []

    def parse_prediction(self, pred_text: str) -> Tuple[Optional[str], List[str]]:
        pred_text = pred_text.strip()
        if self.strategy == self.STRATEGY_DH_COT:
            return self._parse_cot_output(pred_text)
        if self.strategy in self.PATH_OUTPUT_STRATEGIES:
            return self._parse_path_output(pred_text)
        return self._parse_direct_output(pred_text)

    def _parse_label(self, label_text: str) -> Tuple[Optional[str], List[str]]:
        label_text = label_text.strip()
        if self.path_separator in label_text:
            parts = [p.strip() for p in label_text.split(self.path_separator)]
            path: List[str] = []
            for part in parts:
                code = self.label_tree.resolve_label(part)
                if code:
                    path.append(code)
            if path:
                path = self._complete_path_prefix(path)
                return path[-1], path
            return label_text, [label_text] if label_text else []

        code = self.label_tree.resolve_label(label_text)
        if code:
            return code, self.label_tree.find_label_path_list(code)
        return label_text, [label_text] if label_text else []

    def _compute_level_f1(self, class_stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        f1s: List[float] = []
        for _, stats in class_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if (tp + fn) == 0:
                continue
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1s.append(f1)

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0
        return {"micro_f1": micro_f1, "macro_f1": macro_f1}

    def _compute_classification_metrics(self, per_class_stats: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []

        for _, stats in per_class_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if stats.get("total", 0) > 0:
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

        macro_precision = sum(precisions) / len(precisions) if precisions else 0
        macro_recall = sum(recalls) / len(recalls) if recalls else 0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0
        return {
            "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
            "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        }

    def _analyze_confusion(self, confusion_pairs: List[Tuple[str, str]], top_k: int = 20) -> Dict[str, Any]:
        counter = Counter(confusion_pairs)
        top_confusions = []
        for (pred, label), count in counter.most_common(top_k):
            pred_node = self.label_tree.label_code_to_node.get(pred)
            label_node = self.label_tree.label_code_to_node.get(label)
            top_confusions.append(
                {
                    "predicted": pred,
                    "predicted_name": pred_node.name if pred_node else "",
                    "actual": label,
                    "actual_name": label_node.name if label_node else "",
                    "count": count,
                    "same_parent": pred_node.parent == label_node.parent if (pred_node and label_node) else False,
                }
            )
        return {"top_confusions": top_confusions}

    def evaluate(self, predictions: List[str], labels: List[str], raw_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        total = len(labels)
        if total == 0:
            return {"metrics": {"strategy": self.strategy, "total_samples": 0}, "results": []}

        results = []
        exact_match = 0
        level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        path_stats = {"correct": 0, "partial_correct": 0, "total": 0}
        lca_depth_sum = 0
        confusion_pairs: List[Tuple[str, str]] = []
        per_class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
        level_class_stats = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))

        for i, (pred_text, label_text) in enumerate(zip(predictions, labels)):
            pred_code, pred_path = self.parse_prediction(pred_text)
            label_code, label_path = self._parse_label(label_text)

            result = {
                "index": i,
                "label": label_text,
                "label_code": label_code,
                "label_path": label_path,
                "prediction": pred_text,
                "pred_code": pred_code,
                "pred_path": pred_path,
            }

            exact = self._compare(pred_code, label_code)
            result["exact_match"] = exact
            if exact:
                exact_match += 1

            label_level = self.label_tree.get_label_level(label_code or "")
            pred_path_dict = {self.label_tree.get_label_level(code): code for code in pred_path} if pred_path else {}
            label_path_dict = self.label_tree.find_label_path(label_code or "")
            level_matches = {}
            effective_max_level = self.label_tree.get_effective_max_level()
            for level in range(1, effective_max_level + 1):
                pred_at_level = pred_path_dict.get(level)
                label_at_level = label_path_dict.get(level)
                if label_at_level:
                    key = f"level-{level}"
                    level_stats[key]["total"] += 1
                    match = self._compare(pred_at_level, label_at_level)
                    level_matches[key] = match
                    if match:
                        level_stats[key]["correct"] += 1
                        level_class_stats[key][label_at_level]["tp"] += 1
                    else:
                        level_class_stats[key][label_at_level]["fn"] += 1
                        if pred_at_level:
                            level_class_stats[key][pred_at_level]["fp"] += 1
            result["level_matches"] = level_matches

            path_stats["total"] += 1
            path_full_match = False
            if pred_path and label_path:
                prefix_len = 0
                for pred_node, label_node in zip(pred_path, label_path):
                    if self._compare(pred_node, label_node):
                        prefix_len += 1
                    else:
                        break
                result["prefix_match_len"] = prefix_len
                result["label_path_len"] = len(label_path)
                if prefix_len == len(label_path) and len(pred_path) == len(label_path):
                    path_stats["correct"] += 1
                    path_full_match = True
                elif prefix_len > 0:
                    path_stats["partial_correct"] += 1

            level_stats["path"]["total"] += 1
            if path_full_match:
                level_stats["path"]["correct"] += 1
            if pred_code and label_code:
                lca = self.label_tree.get_lca(pred_code, label_code)
                if lca:
                    lca_depth_sum += max(0, self.label_tree.get_label_level(lca))

            if label_code:
                per_class_stats[label_code]["total"] += 1
                if exact:
                    per_class_stats[label_code]["tp"] += 1
                else:
                    per_class_stats[label_code]["fn"] += 1
                    if pred_code:
                        per_class_stats[pred_code]["fp"] += 1
                        confusion_pairs.append((pred_code, label_code))

            if label_level > 0:
                level_stats["leaf"]["total"] += 1
                pred_at_label_level = pred_path_dict.get(label_level)
                if self._compare(pred_at_label_level, label_code):
                    level_stats["leaf"]["correct"] += 1

            level_stats["final"]["total"] += 1
            if exact:
                level_stats["final"]["correct"] += 1

            results.append(result)

        metrics: Dict[str, Any] = {
            "strategy": self.strategy,
            "total_samples": total,
            "exact_match": {"accuracy": exact_match / total, "correct": exact_match, "total": total},
            "hierarchical": {},
            "path_match": {
                "full_match_rate": path_stats["correct"] / path_stats["total"] if path_stats["total"] > 0 else 0,
                "partial_match_rate": path_stats["partial_correct"] / path_stats["total"] if path_stats["total"] > 0 else 0,
                "full_match": path_stats["correct"],
                "partial_match": path_stats["partial_correct"],
                "total": path_stats["total"],
            },
            "avg_lca_depth": lca_depth_sum / total if total > 0 else 0,
            "confusion_analysis": self._analyze_confusion(confusion_pairs),
            "classification": self._compute_classification_metrics(per_class_stats),
        }

        for level_name, stats in level_stats.items():
            if stats["total"] <= 0:
                continue
            entry = {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
            class_stats = level_class_stats.get(level_name)
            if class_stats:
                entry.update(self._compute_level_f1(class_stats))
            else:
                entry["micro_f1"] = 0.0
                entry["macro_f1"] = 0.0
            metrics["hierarchical"][level_name] = entry

        return {"metrics": metrics, "results": results}

    def format_report(self, metrics: Dict[str, Any]) -> str:
        lines = [
            "",
            "=" * 80,
            "  层级文档分类评测报告",
            "=" * 80,
            f"  策略: {metrics.get('strategy', 'unknown').upper()}",
            f"  样本数: {metrics.get('total_samples', 0)}",
            "",
        ]
        em = metrics.get("exact_match", {})
        lines.append(f"Exact Match Accuracy: {em.get('accuracy', 0) * 100:.2f}% ({em.get('correct', 0)}/{em.get('total', 0)})")
        lines.append("")
        hierarchical = metrics.get("hierarchical", {})
        if hierarchical:
            lines.append("Hierarchical Accuracy:")
            for key in sorted(hierarchical.keys()):
                v = hierarchical[key]
                lines.append(
                    f"  {key}: {v.get('accuracy', 0) * 100:.2f}% "
                    f"(Micro-F1 {v.get('micro_f1', 0) * 100:.2f}%, Macro-F1 {v.get('macro_f1', 0) * 100:.2f}%)"
                )
            lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)


class TMHChainEvaluator:
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def _compare(self, a: Optional[str], b: Optional[str]) -> bool:
        if a is None or b is None:
            return False
        if self.case_sensitive:
            return a == b
        return a.lower() == b.lower()

    def evaluate(self, predictions: List[str], labels: List[str], meta_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(predictions) != len(labels) or len(predictions) != len(meta_list):
            raise ValueError("predictions, labels, meta_list 长度必须一致")

        sample_groups = defaultdict(list)
        for i, (pred, label, meta) in enumerate(zip(predictions, labels, meta_list)):
            sample_id = meta.get("sample_id")
            if sample_id is None:
                continue
            sample_groups[sample_id].append(
                {
                    "index": i,
                    "prediction": pred.strip(),
                    "label": label.strip(),
                    "step_index": meta.get("step_index", 0),
                    "target_level": meta.get("target_level", 1),
                    "is_stop": meta.get("is_stop", False),
                }
            )

        total_samples = len(sample_groups)
        sample_correct = 0
        step_stats = defaultdict(lambda: {"total": 0, "step_correct": 0, "chain_correct": 0})
        sample_results = []
        step_results = []

        for sample_id, steps in sample_groups.items():
            steps = sorted(steps, key=lambda x: x["step_index"])
            chain_failed = False
            all_steps_correct = True
            one_sample_steps = []
            for step in steps:
                step_correct = self._compare(step["prediction"], step["label"])
                chain_correct = False if chain_failed else step_correct
                if not step_correct:
                    chain_failed = True
                    all_steps_correct = False

                step_key = f"step_{step['step_index']}"
                level_key = f"level_{step['target_level']}"
                for key in (step_key, level_key):
                    step_stats[key]["total"] += 1
                    if step_correct:
                        step_stats[key]["step_correct"] += 1
                    if chain_correct:
                        step_stats[key]["chain_correct"] += 1

                row = {
                    "sample_id": sample_id,
                    "step_index": step["step_index"],
                    "target_level": step["target_level"],
                    "prediction": step["prediction"],
                    "label": step["label"],
                    "step_correct": step_correct,
                    "chain_correct": chain_correct,
                    "is_stop": step["is_stop"],
                }
                one_sample_steps.append(row)
                step_results.append(row)

            if all_steps_correct:
                sample_correct += 1

            sample_results.append(
                {
                    "sample_id": sample_id,
                    "total_steps": len(steps),
                    "all_correct": all_steps_correct,
                    "steps": one_sample_steps,
                }
            )

        metrics: Dict[str, Any] = {
            "total_samples": total_samples,
            "total_steps": len(step_results),
            "sample_accuracy": {
                "accuracy": sample_correct / total_samples if total_samples > 0 else 0,
                "correct": sample_correct,
                "total": total_samples,
            },
            "step_metrics": {},
        }
        for key, stats in sorted(step_stats.items()):
            total = stats["total"]
            if total <= 0:
                continue
            metrics["step_metrics"][key] = {
                "total": total,
                "step_accuracy": stats["step_correct"] / total,
                "step_correct": stats["step_correct"],
                "chain_accuracy": stats["chain_correct"] / total,
                "chain_correct": stats["chain_correct"],
            }
        return {"metrics": metrics, "sample_results": sample_results, "step_results": step_results}

    def format_report(self, metrics: Dict[str, Any]) -> str:
        acc = metrics.get("sample_accuracy", {})
        return (
            "\n"
            + "=" * 80
            + "\nTMH 链式聚合评估报告\n"
            + "=" * 80
            + f"\n样本级准确率: {acc.get('accuracy', 0) * 100:.2f}% ({acc.get('correct', 0)}/{acc.get('total', 0)})\n"
            + "=" * 80
        )


class OfflineClassifyEvaluator:
    def __init__(self, data_args: OfflineEvalDataArgs, eval_args: OfflineEvalArgs):
        self.data_args = data_args
        self.eval_args = eval_args
        self.output_field = data_args.response or "output"
        self.predict_field = eval_args.predict_field
        self.case_sensitive = eval_args.case_sensitive
        self.save_dir = eval_args.save_dir
        self.predict_dir = eval_args.predict_dir
        self.predict_file_prefix = eval_args.predict_file_prefix

    def _clean_prediction(self, pred: str) -> str:
        pred = re.sub(r"<think>.*?</think>\s*", "", pred, flags=re.DOTALL | re.IGNORECASE)
        pred = re.sub(r"<thinking>.*?</thinking>\s*", "", pred, flags=re.DOTALL | re.IGNORECASE)
        pred = re.sub(r"<thought>.*?</thought>\s*", "", pred, flags=re.DOTALL | re.IGNORECASE)
        return pred.strip()

    def _load_file(self, filepath: str) -> List[Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        try:
            return [json.loads(line.strip()) for line in content.split("\n") if line.strip()]
        except json.JSONDecodeError:
            pass
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            logger.warning("Failed to parse file: %s", filepath)
            return []

    def load_predict_results(self) -> List[Dict[str, Any]]:
        predict_path = self.predict_dir
        if not predict_path:
            raise ValueError("`predict_dir` must be specified.")
        if not os.path.exists(predict_path):
            raise FileNotFoundError(f"Predict path not found: {predict_path}")

        if os.path.isfile(predict_path):
            return self._load_file(predict_path)

        all_files = sorted([f for f in os.listdir(predict_path) if not f.startswith(".")])
        if self.predict_file_prefix:
            matched = []
            for name in all_files:
                if name == self.predict_file_prefix:
                    matched.append(name)
                    continue
                if name.startswith(self.predict_file_prefix + "_"):
                    suffix = name[len(self.predict_file_prefix) + 1 :]
                    if suffix.isdigit():
                        matched.append(name)
            if not matched:
                raise FileNotFoundError(
                    f"No files matching prefix '{self.predict_file_prefix}' found in {predict_path}"
                )
            all_files = sorted(matched, key=lambda x: -1 if x == self.predict_file_prefix else int(x.split("_")[-1]))

        results: List[Dict[str, Any]] = []
        for name in all_files:
            filepath = os.path.join(predict_path, name)
            if os.path.isfile(filepath):
                results.extend(self._load_file(filepath))
        return results

    def compute_metrics(self, predictions: List[str], labels: List[str]) -> Dict[str, Any]:
        total = len(labels)
        if total == 0:
            return {"accuracy": 0.0, "correct": 0, "total": 0}
        correct = 0
        for pred, label in zip(predictions, labels):
            match = pred == label if self.case_sensitive else pred.lower() == label.lower()
            if match:
                correct += 1
        return {"accuracy": correct / total, "correct": correct, "total": total}

    def _save_results(self, metrics: Dict[str, Any], results: List[Dict[str, Any]]):
        report = f"Accuracy: {metrics.get('accuracy', 0) * 100:.2f}% ({metrics.get('correct', 0)}/{metrics.get('total', 0)})"
        print(report)
        if not self.save_dir:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.save_dir, "eval_results.jsonl"), "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(self.save_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    def eval(self) -> Dict[str, Any]:
        data = self.load_predict_results()
        predictions = []
        labels = []
        results = []
        for sample in data:
            label = sample.get(self.output_field)
            pred = sample.get(self.predict_field)
            if label is None or pred is None:
                continue
            label_s = str(label).strip()
            pred_s = self._clean_prediction(str(pred).strip())
            predictions.append(pred_s)
            labels.append(label_s)
            match = pred_s == label_s if self.case_sensitive else pred_s.lower() == label_s.lower()
            results.append({"label": label_s, "prediction": pred_s, "correct": match})
        metrics = self.compute_metrics(predictions, labels)
        self._save_results(metrics, results)
        return {"metrics": metrics, "results": results}


class ExtendedOfflineClassifyEvaluator(OfflineClassifyEvaluator):
    def __init__(self, data_args: OfflineEvalDataArgs, eval_args: OfflineEvalArgs):
        super().__init__(data_args, eval_args)
        self.strategy = eval_args.strategy
        self.path_separator = eval_args.path_separator
        self.tmh_aggregate = eval_args.tmh_aggregate
        self.label_tree: Optional[LabelTree] = None
        if eval_args.label_config_path:
            self.label_tree = LabelTree(
                label_config_path=eval_args.label_config_path,
                label_code_key=eval_args.label_code_key,
                label_name_key=eval_args.label_name_key,
                label_parent_key=eval_args.label_parent_key,
            )
        self.hierarchical_evaluator = (
            HierarchicalClassifyEvaluator(
                label_tree=self.label_tree,
                strategy=self.strategy,
                path_separator=self.path_separator,
                case_sensitive=self.case_sensitive,
            )
            if self.label_tree
            else None
        )
        self.tmh_chain_evaluator = (
            TMHChainEvaluator(case_sensitive=self.case_sensitive)
            if self.tmh_aggregate and self.strategy.lower() == "tmh"
            else None
        )

    def _save_hierarchical_result(self, eval_result: Dict[str, Any], raw_data: List[Dict[str, Any]]):
        metrics = eval_result.get("metrics", {})
        results = eval_result.get("results", [])
        if raw_data and len(raw_data) == len(results):
            merged = []
            for raw, result in zip(raw_data, results):
                row = dict(raw)
                row.update(result)
                merged.append(row)
            results = merged

        report = self.hierarchical_evaluator.format_report(metrics) if self.hierarchical_evaluator else "No report"
        print(report)
        if not self.save_dir:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.save_dir, "eval_results.jsonl"), "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(self.save_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        errors = [r for r in results if not r.get("exact_match", True)]
        with open(os.path.join(self.save_dir, "errors.jsonl"), "w", encoding="utf-8") as f:
            for row in errors:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_tmh_result(self, eval_result: Dict[str, Any]):
        metrics = eval_result.get("metrics", {})
        sample_results = eval_result.get("sample_results", [])
        step_results = eval_result.get("step_results", [])
        report = self.tmh_chain_evaluator.format_report(metrics) if self.tmh_chain_evaluator else "No report"
        print(report)
        if not self.save_dir:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.save_dir, "sample_results.jsonl"), "w", encoding="utf-8") as f:
            for row in sample_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(self.save_dir, "step_results.jsonl"), "w", encoding="utf-8") as f:
            for row in step_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(self.save_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    def eval(self) -> Dict[str, Any]:
        data = self.load_predict_results()
        predictions = []
        labels = []
        raw_data = []
        meta_list: List[Dict[str, Any]] = []
        for sample in data:
            label = sample.get(self.output_field)
            pred = sample.get(self.predict_field)
            if label is None or pred is None:
                continue
            predictions.append(self._clean_prediction(str(pred).strip()))
            labels.append(str(label).strip())
            raw_data.append(sample)
            meta_list.append(sample.get("meta", {}))

        if self.tmh_chain_evaluator and any(meta.get("sample_id") is not None for meta in meta_list):
            eval_result = self.tmh_chain_evaluator.evaluate(predictions, labels, meta_list)
            self._save_tmh_result(eval_result)
            return eval_result

        if self.hierarchical_evaluator:
            eval_result = self.hierarchical_evaluator.evaluate(predictions, labels, raw_data=raw_data)
            self._save_hierarchical_result(eval_result, raw_data)
            return eval_result

        # 没有层级配置则退化为 flat 评测
        return super().eval()


def run_offline_eval():
    data_args, eval_args = parse_offline_eval_args()
    if eval_args.task == "offline_classify":
        evaluator = OfflineClassifyEvaluator(data_args, eval_args)
    elif eval_args.task == "hierarchical_classify":
        evaluator = ExtendedOfflineClassifyEvaluator(data_args, eval_args)
    else:
        raise ValueError(
            f"Unknown task: {eval_args.task}. Available: offline_classify, hierarchical_classify"
        )
    evaluator.eval()


if __name__ == "__main__":
    run_offline_eval()
