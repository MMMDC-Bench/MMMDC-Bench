import argparse
import ast
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

STOP_TOKEN = "[STOP]"


def str_to_bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")


def str_to_dict(value: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    return json.loads(str(value).strip())


def load_dict(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return []


def read_dataframe(file_path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in {".xls", ".xlsx"}:
        df = pd.read_excel(file_path, engine="openpyxl")
    elif ext in {".pickle", ".pkl"}:
        df = pd.read_pickle(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return df.replace([np.nan, None], "")


@dataclass
class LabelNode:
    code: str
    name: str = ""
    description: str = ""
    parent: str = ""
    children: List[str] = field(default_factory=list)
    level: int = -1

    def add_child(self, child_code: str) -> None:
        if child_code not in self.children:
            self.children.append(child_code)


class LabelTree:
    DEFAULT_ATTR_MAP = {
        "label_code": "label_code",
        "label_name": "label_name",
        "label_desc": "label_desc",
        "label_parent": "label_parent",
    }

    def __init__(
        self,
        label_config_path: str,
        label_attr_name_map: Optional[Union[Dict[str, str], str]] = None,
    ):
        self.label_attr_name_map = self._parse_attr_map(label_attr_name_map)
        self.label_roots: List[LabelNode] = []
        self.label_code_to_node: Dict[str, LabelNode] = {}
        self.max_level = -1
        self.min_level = 1
        self._build_from_file(label_config_path)

    def _parse_attr_map(
        self, attr_map: Optional[Union[Dict[str, str], str]]
    ) -> Dict[str, str]:
        if attr_map is None:
            return self.DEFAULT_ATTR_MAP.copy()
        if isinstance(attr_map, str):
            try:
                parsed = json.loads(attr_map)
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(attr_map, dict):
            parsed = attr_map
        else:
            parsed = {}
        merged = self.DEFAULT_ATTR_MAP.copy()
        merged.update(parsed)
        return merged

    def _build_from_file(self, label_config_path: str) -> None:
        with open(label_config_path, "r", encoding="utf-8") as f:
            label_config = json.load(f)
        self._build_from_dict(label_config)

    def _build_from_dict(self, label_config: Dict[str, Any]) -> None:
        attr_map = self.label_attr_name_map
        for label_key, label_attrs in label_config.items():
            code = str(label_attrs.get(attr_map["label_code"], label_key)).strip()
            name = str(label_attrs.get(attr_map["label_name"], "")).strip()
            desc = str(label_attrs.get(attr_map["label_desc"], "")).strip()
            parent = str(label_attrs.get(attr_map["label_parent"], "")).strip()
            node = LabelNode(code=code, name=name, description=desc, parent=parent)
            self.label_code_to_node[code] = node
            if not parent or parent == code:
                self.label_roots.append(node)

        for code, node in self.label_code_to_node.items():
            if node.parent and node.parent != code and node.parent in self.label_code_to_node:
                self.label_code_to_node[node.parent].add_child(code)

        self._recover_level()
        self._detect_virtual_root()

    def _recover_level(self) -> None:
        container = list(self.label_roots)
        level = 1
        while container:
            next_level = []
            for node in container:
                node.level = level
                for child_code in node.children:
                    child = self.label_code_to_node.get(child_code)
                    if child is not None:
                        next_level.append(child)
            container = next_level
            level += 1
        self.max_level = level - 1

    def _detect_virtual_root(self) -> None:
        if len(self.label_roots) != 1:
            self.min_level = 1
            return
        root = self.label_roots[0]
        root_name = root.name.lower() if root.name else ""
        root_code = root.code.lower() if root.code else ""
        indicators = {"root", "all", "总", "全部", "", "0", "-1"}
        self.min_level = 2 if root.children and (root_name in indicators or root_code in indicators) else 1

    def find_label_path_list(self, label_code: str, exclude_virtual_root: bool = True) -> List[str]:
        node = self.label_code_to_node.get(label_code)
        if node is None:
            return [label_code]
        path = [node.code]
        while node.parent and node.parent != node.code:
            parent = self.label_code_to_node.get(node.parent)
            if parent is None:
                break
            path.append(parent.code)
            node = parent
        path = list(reversed(path))
        if exclude_virtual_root and self.min_level > 1 and len(path) > 1:
            return path[1:]
        return path

    def get_children_codes(self, parent_code: str) -> List[str]:
        node = self.label_code_to_node.get(parent_code)
        return list(node.children) if node else []

    def get_label_level(self, label_code: str, effective: bool = True) -> int:
        node = self.label_code_to_node.get(label_code)
        if node is None:
            return -1
        if effective and self.min_level > 1:
            return node.level - self.min_level + 1
        return node.level

    def format_tree_string(self, max_depth: Optional[int] = None, show_description: bool = False) -> str:
        def build(node: LabelNode, depth: int) -> List[str]:
            if max_depth is not None and depth > max_depth:
                return []
            indent = "  " * depth
            if show_description and node.description:
                line = f"{indent}- {node.code} ({node.name}): {node.description}"
            else:
                line = f"{indent}- {node.code} ({node.name})"
            lines = [line]
            for child_code in node.children:
                child = self.label_code_to_node.get(child_code)
                if child is not None:
                    lines.extend(build(child, depth + 1))
            return lines

        roots = list(self.label_roots)
        if self.min_level > 1 and len(roots) == 1:
            roots = [
                self.label_code_to_node[c]
                for c in roots[0].children
                if c in self.label_code_to_node
            ]

        result = []
        for root in roots:
            result.extend(build(root, 0))
        return "\n".join(result)


class HierarchicalPromptBuilder:
    def __init__(
        self,
        label_tree: Optional[LabelTree],
        use_ocr_text: bool = True,
        use_image_pages: bool = True,
        use_file_name: bool = False,
        masked_text_value: bool = False,
        masked_image_value: bool = False,
        include_taxonomy: bool = True,
    ):
        self.label_tree = label_tree
        self.use_ocr_text = use_ocr_text
        self.use_image_pages = use_image_pages
        self.use_file_name = use_file_name
        self.masked_text_value = masked_text_value
        self.masked_image_value = masked_image_value
        self.include_taxonomy = include_taxonomy

    def _normalize_image_paths(self, raw_value: Any) -> List[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, float) and np.isnan(raw_value):
            return []

        if isinstance(raw_value, list):
            result = [str(item).strip() for item in raw_value if str(item).strip()]
            return result
        if isinstance(raw_value, tuple):
            result = [str(item).strip() for item in raw_value if str(item).strip()]
            return result

        if isinstance(raw_value, dict):
            for key in ("image_path", "path", "url"):
                value = raw_value.get(key)
                if value and str(value).strip():
                    return [str(value).strip()]
            return []

        text = str(raw_value).strip()
        if not text:
            return []

        parsed = load_dict(text)
        if isinstance(parsed, list):
            result = [str(item).strip() for item in parsed if str(item).strip()]
            if result:
                return result
        elif isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]

        # 兼容单图字段（如 image_path）直接传字符串路径的场景。
        return [text]

    def _extract_image_paths_from_row(self, row: pd.Series, col_map: Dict[str, str]) -> List[str]:
        candidate_cols: List[str] = []
        if self.masked_image_value:
            candidate_cols.extend(
                [
                    col_map.get("masked_value_image_path_list", "masked_value_image_path_list"),
                    col_map.get("masked_value_image_path", "masked_value_image_path"),
                ]
            )
        candidate_cols.extend(
            [
                col_map.get("image_path_list", "image_path_list"),
                col_map.get("image_path", "image_path"),
                "image_path_list",
                "image_path",
            ]
        )

        seen = set()
        dedup_candidate_cols = []
        for col in candidate_cols:
            if col and col not in seen:
                dedup_candidate_cols.append(col)
                seen.add(col)

        for col in dedup_candidate_cols:
            if col not in row.index:
                continue
            image_paths = self._normalize_image_paths(row.get(col))
            if image_paths:
                return image_paths
        return []

    def construct_dataset(
        self,
        data_df: pd.DataFrame,
        col_map: Dict[str, str],
        strategy: str,
        oss_to_url: bool,
        keep_raw_cols: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        strategy = strategy.lower()
        if strategy == "dp":
            return self._construct_dp_data(data_df, col_map, oss_to_url, keep_raw_cols, **kwargs)
        if strategy == "retrieval_candidates_precomputed":
            return self._construct_retrieval_candidates_precomputed_data(
                data_df, col_map, oss_to_url, keep_raw_cols, **kwargs
            )
        raise ValueError(
            f"Unsupported strategy in MMMDC-Bench lightweight builder: {strategy}. "
            "Supported: dp, retrieval_candidates_precomputed"
        )

    def _extract_row_data(self, row: pd.Series, col_map: Dict[str, str]) -> Dict[str, Any]:
        file_type_col = col_map.get("file_type", "file_type")
        data = {"file_type": row.get(file_type_col, "")}

        if self.use_ocr_text:
            ocr_col = col_map.get("no_value_ocr_text", "no_value_ocr_text") if self.masked_text_value else col_map.get("ocr_text", "ocr_text")
            ocr_text = row.get(ocr_col, "")
            if self.masked_text_value and not str(ocr_text).strip():
                ocr_text = row.get(col_map.get("ocr_text", "ocr_text"), "")
            data["ocr_text"] = "" if pd.isna(ocr_text) else str(ocr_text)
        else:
            data["ocr_text"] = ""

        if self.use_image_pages:
            data["image_path_list"] = self._extract_image_paths_from_row(row, col_map)
        else:
            data["image_path_list"] = []

        if self.use_file_name:
            data["file_name"] = str(row.get(col_map.get("file_name", "file_name"), ""))
        else:
            data["file_name"] = ""
        return data

    def _process_images(self, image_paths: List[str], oss_to_url: bool) -> List[str]:
        # MMMDC-Bench 当前未集成 OSSLinker，保留原路径行为。
        _ = oss_to_url
        return image_paths

    def _get_image_placeholders(self, num_images: int) -> str:
        return "\n".join(["<image>" for _ in range(num_images)])

    def _format_candidate_list_simple(self, codes: List[str]) -> str:
        if not codes:
            return "（无候选类别）"
        return "\n".join([f"- {code}" for code in codes])

    def _format_candidate_list_detailed(self, codes: List[str]) -> str:
        if self.label_tree is None:
            return self._format_candidate_list_simple(codes)
        lines = []
        for code in codes:
            node = self.label_tree.label_code_to_node.get(code)
            if node is None:
                lines.append(f"- {code}")
            elif node.description:
                lines.append(f"- {code} ({node.name}): {node.description}")
            else:
                lines.append(f"- {code} ({node.name})")
        return "\n".join(lines) if lines else "（无候选类别）"

    def _format_categories_with_hierarchy(self, codes: List[str], show_details: bool) -> str:
        if self.label_tree is None:
            return self._format_candidate_list_simple(codes)
        lines = []
        sortable = []
        for code in codes:
            node = self.label_tree.label_code_to_node.get(code)
            if node is not None:
                sortable.append((max(node.level - self.label_tree.min_level + 1, 1), code, node))
        for level, code, node in sorted(sortable, key=lambda x: (x[0], x[1])):
            indent = "  " * (level - 1)
            if show_details and node.description:
                lines.append(f"{indent}- {code} ({node.name}): {node.description}")
            else:
                lines.append(f"{indent}- {code} ({node.name})")
        return "\n".join(lines) if lines else "（无候选类别）"

    def _get_all_categories(self) -> List[str]:
        if self.label_tree is None:
            return []
        if self.label_tree.min_level > 1 and len(self.label_tree.label_roots) == 1:
            root_code = self.label_tree.label_roots[0].code
            return [code for code in self.label_tree.label_code_to_node.keys() if code != root_code]
        return list(self.label_tree.label_code_to_node.keys())

    def _extract_raw_info(
        self,
        row: pd.Series,
        index: int,
        keep_raw_cols: Optional[Union[str, List[str]]],
    ) -> Optional[Dict[str, Any]]:
        if not keep_raw_cols:
            return None
        raw = {"_index": index}
        cols = list(row.index) if keep_raw_cols == "__all__" else keep_raw_cols
        for col in cols:
            if col not in row.index:
                continue
            value = row[col]
            if hasattr(value, "tolist"):
                value = value.tolist()
            elif hasattr(value, "item"):
                value = value.item()
            elif pd.isna(value):
                value = None
            raw[col] = value
        return raw

    def _build_input_section(self, data: Dict[str, Any], placeholders: str) -> str:
        sections = []
        if self.use_ocr_text:
            sections.append(f"文档解析文本: {data['ocr_text']}")
        if self.use_image_pages:
            sections.append(f"文档页图片:\n{placeholders}")
        if self.use_file_name:
            sections.append(f"文件名: {data['file_name']}")
        return "\n".join(sections) if sections else "无输入内容"

    def _construct_dp_data(
        self,
        data_df: pd.DataFrame,
        col_map: Dict[str, str],
        oss_to_url: bool,
        keep_raw_cols: Optional[Union[str, List[str]]],
        show_details: bool = True,
        show_hierarchy: bool = True,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        all_categories = self._get_all_categories()
        if self.include_taxonomy:
            if show_hierarchy:
                candidate_text = self._format_categories_with_hierarchy(all_categories, show_details)
            elif show_details:
                candidate_text = self._format_candidate_list_detailed(all_categories)
            else:
                candidate_text = self._format_candidate_list_simple(all_categories)
        else:
            candidate_text = ""

        items: List[Dict[str, Any]] = []
        for index, row in data_df.iterrows():
            data = self._extract_row_data(row, col_map)
            images = self._process_images(data["image_path_list"], oss_to_url)
            input_section = self._build_input_section(data, self._get_image_placeholders(len(images)))
            if self.include_taxonomy:
                instruction = (
                    "# 角色与任务\n"
                    "你是一名文档分类专家。\n"
                    "请分析文档内容，从候选类别中选择最匹配的文档类型Code。\n"
                    "仅输出Code，严禁输出其他字符。\n\n"
                    "# 候选类别\n"
                    f"{candidate_text}\n\n"
                    "# 输入\n"
                    f"{input_section}\n"
                )
            else:
                instruction = (
                    "# 角色与任务\n"
                    "你是一名文档分类专家。\n"
                    "请分析文档内容，预测最匹配的文档类型Code。\n"
                    "仅输出Code，严禁输出其他字符。\n\n"
                    "# 输入\n"
                    f"{input_section}\n"
                )
            item: Dict[str, Any] = {
                "instruction": instruction,
                "output": data["file_type"],
                "images": images,
            }
            raw = self._extract_raw_info(row, int(index), keep_raw_cols)
            if raw is not None:
                item["raw"] = raw
            items.append(item)
        return items

    def _parse_precomputed_candidates(self, raw_candidates: Any) -> List[Dict[str, Any]]:
        raw = load_dict(raw_candidates)
        if not isinstance(raw, list):
            return []
        seen = set()
        parsed = []
        for item in raw:
            if isinstance(item, str):
                code = item
                score = 0.0
            elif isinstance(item, dict):
                code = item.get("label_code") or item.get("file_type")
                score = item.get("score", 0.0)
            else:
                continue
            if not code or code in seen:
                continue
            seen.add(code)
            node = self.label_tree.label_code_to_node.get(code) if self.label_tree else None
            parsed.append(
                {
                    "label_code": code,
                    "label_name": node.name if node else code,
                    "label_desc": node.description if node else "",
                    "score": float(score),
                }
            )
        return parsed

    def _format_candidate_labels(self, candidates: List[Dict[str, Any]], show_scores: bool) -> str:
        if not candidates:
            return "（无候选类别）"
        lines = []
        for idx, c in enumerate(candidates, 1):
            code = c.get("label_code", "")
            name = c.get("label_name", "")
            desc = c.get("label_desc", "")
            score = float(c.get("score", 0.0))
            if show_scores:
                if desc:
                    lines.append(f"{idx}. {code} ({name}): {desc} [相关度: {score:.2f}]")
                else:
                    lines.append(f"{idx}. {code} ({name}) [相关度: {score:.2f}]")
            else:
                if desc:
                    lines.append(f"{idx}. {code} ({name}): {desc}")
                else:
                    lines.append(f"{idx}. {code} ({name})")
        return "\n".join(lines)

    def _construct_retrieval_candidates_precomputed_data(
        self,
        data_df: pd.DataFrame,
        col_map: Dict[str, str],
        oss_to_url: bool,
        keep_raw_cols: Optional[Union[str, List[str]]],
        candidate_col: str = "candidate_labels",
        include_gt_in_candidates: bool = True,
        show_scores: bool = True,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for index, row in data_df.iterrows():
            data = self._extract_row_data(row, col_map)
            images = self._process_images(data["image_path_list"], oss_to_url)
            candidates = self._parse_precomputed_candidates(row.get(candidate_col, []))

            gt_code = data["file_type"]
            gt_in_list = any(c.get("label_code") == gt_code for c in candidates)
            if include_gt_in_candidates and not gt_in_list:
                node = self.label_tree.label_code_to_node.get(gt_code) if self.label_tree else None
                candidates.append(
                    {
                        "label_code": gt_code,
                        "label_name": node.name if node else gt_code,
                        "label_desc": node.description if node else "",
                        "score": 0.0,
                    }
                )
            input_section = self._build_input_section(data, self._get_image_placeholders(len(images)))
            candidate_text = self._format_candidate_labels(candidates, show_scores=show_scores)
            if self.include_taxonomy:
                instruction = (
                    "# 角色与任务\n"
                    "你是一名文档分类专家。\n"
                    "系统已通过检索为你预筛选了最可能的候选类别。\n"
                    "请从这些候选类别中选择最匹配的一个。\n\n"
                    "# 候选类别（按相关度排序）\n"
                    f"{candidate_text}\n\n"
                    "# 输入\n"
                    f"{input_section}\n\n"
                    "# 输出要求\n"
                    "请从上述候选类别中选择最匹配的类别Code。\n"
                    "如果所有候选都不匹配，请输出: [OTHER]\n"
                    "仅输出一个类别Code或[OTHER]，严禁输出其他字符。\n"
                )
            else:
                instruction = (
                    "# 角色与任务\n"
                    "你是一名文档分类专家。\n"
                    "请分析文档内容，预测最匹配的文档类型Code。\n\n"
                    "# 输入\n"
                    f"{input_section}\n\n"
                    "# 输出要求\n"
                    "请输出最匹配的类别Code。\n"
                    "仅输出一个类别Code，严禁输出其他字符。\n"
                )

            item: Dict[str, Any] = {
                "instruction": instruction,
                "output": gt_code,
                "images": images,
                "meta": {
                    "retrieval_type": "precomputed",
                    "candidates": [c.get("label_code") for c in candidates],
                    "gt_in_candidates": gt_in_list,
                    "gt_code": gt_code,
                },
            }
            raw = self._extract_raw_info(row, int(index), keep_raw_cols)
            if raw is not None:
                item["raw"] = raw
            items.append(item)
        return items


def parse_keep_raw_cols(keep_raw_cols: str) -> Optional[Union[str, List[str]]]:
    if not keep_raw_cols:
        return None
    keep_raw_cols = keep_raw_cols.strip()
    if not keep_raw_cols:
        return None
    if keep_raw_cols.lower() == "__all__":
        return "__all__"
    cols = [col.strip() for col in keep_raw_cols.split(",") if col.strip()]
    return cols or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical data construct for MMMDC-Bench")

    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--exp_note", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--col_map", type=str_to_dict, default={})
    parser.add_argument("--oss_to_url", type=str_to_bool, default=False)

    parser.add_argument("--strategy", type=str, default="dp")
    parser.add_argument("--label_config_path", type=str, default="")
    parser.add_argument("--label_attr_name_map", type=str_to_dict, default={})

    parser.add_argument("--show_details", type=str_to_bool, default=True)
    parser.add_argument("--show_hierarchy", type=str_to_bool, default=True)
    parser.add_argument("--candidate_col", type=str, default="candidate_labels")
    parser.add_argument("--include_gt_in_candidates", type=str_to_bool, default=True)
    parser.add_argument("--show_scores", type=str_to_bool, default=True)

    parser.add_argument("--use_ocr_text", type=str_to_bool, default=True)
    parser.add_argument("--use_image_pages", type=str_to_bool, default=True)
    parser.add_argument("--use_file_name", type=str_to_bool, default=False)
    parser.add_argument("--masked_text_value", type=str_to_bool, default=False)
    parser.add_argument("--masked_image_value", type=str_to_bool, default=False)
    parser.add_argument("--include_taxonomy", type=str_to_bool, default=True)
    parser.add_argument("--keep_raw_cols", type=str, default="")

    return parser.parse_args()


def hierarchical_data_construct(configs: argparse.Namespace) -> None:
    strategy = configs.strategy.lower()
    logger.info("Using hierarchical classification strategy: %s", strategy)

    label_tree = None
    if configs.label_config_path and os.path.exists(configs.label_config_path):
        label_tree = LabelTree(
            label_config_path=configs.label_config_path,
            label_attr_name_map=configs.label_attr_name_map,
        )
        logger.info("Loaded LabelTree with %s labels", len(label_tree.label_code_to_node))
    else:
        logger.warning("label_config_path not provided or not exists: %s", configs.label_config_path)

    use_ocr_text = configs.use_ocr_text
    use_image_pages = configs.use_image_pages
    masked_text_value = configs.masked_text_value if use_ocr_text else False
    masked_image_value = configs.masked_image_value if use_image_pages else False

    builder = HierarchicalPromptBuilder(
        label_tree=label_tree,
        use_ocr_text=use_ocr_text,
        use_image_pages=use_image_pages,
        use_file_name=configs.use_file_name,
        masked_text_value=masked_text_value,
        masked_image_value=masked_image_value,
        include_taxonomy=configs.include_taxonomy,
    )
    base_builder = None
    use_masked_augment = (
        (masked_text_value and use_ocr_text) or (masked_image_value and use_image_pages)
    )
    if use_masked_augment:
        base_builder = HierarchicalPromptBuilder(
            label_tree=label_tree,
            use_ocr_text=use_ocr_text,
            use_image_pages=use_image_pages,
            use_file_name=configs.use_file_name,
            masked_text_value=False,
            masked_image_value=False,
            include_taxonomy=configs.include_taxonomy,
        )

    os.makedirs(configs.output_dir, exist_ok=True)
    keep_raw_cols = parse_keep_raw_cols(configs.keep_raw_cols)

    strategy_kwargs: Dict[str, Any] = {}
    if strategy == "dp":
        strategy_kwargs["show_details"] = configs.show_details
        strategy_kwargs["show_hierarchy"] = configs.show_hierarchy
    elif strategy == "retrieval_candidates_precomputed":
        strategy_kwargs["candidate_col"] = configs.candidate_col
        strategy_kwargs["include_gt_in_candidates"] = configs.include_gt_in_candidates
        strategy_kwargs["show_scores"] = configs.show_scores

    def process_file(file_path: str, output_file_path: str) -> None:
        data_df = read_dataframe(file_path)
        logger.info("Load Dataset Size: %s from %s", len(data_df), file_path)

        if use_masked_augment and base_builder is not None:
            normal_list = base_builder.construct_dataset(
                data_df=data_df,
                col_map=configs.col_map,
                strategy=strategy,
                oss_to_url=configs.oss_to_url,
                keep_raw_cols=keep_raw_cols,
                **strategy_kwargs,
            )
            masked_list = builder.construct_dataset(
                data_df=data_df,
                col_map=configs.col_map,
                strategy=strategy,
                oss_to_url=configs.oss_to_url,
                keep_raw_cols=keep_raw_cols,
                **strategy_kwargs,
            )
            data_json_list = normal_list + masked_list
            logger.info(
                "Masked augment enabled: unmasked=%s, masked=%s, merged=%s",
                len(normal_list),
                len(masked_list),
                len(data_json_list),
            )
        else:
            data_json_list = builder.construct_dataset(
                data_df=data_df,
                col_map=configs.col_map,
                strategy=strategy,
                oss_to_url=configs.oss_to_url,
                keep_raw_cols=keep_raw_cols,
                **strategy_kwargs,
            )

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(data_json_list, f, ensure_ascii=False, indent=2)
        logger.info("Generated %s samples, saved to %s", len(data_json_list), output_file_path)

    if configs.data_path:
        file_path = configs.data_path
        pure_file_name = ".".join(os.path.basename(file_path).split(".")[:-1]) or os.path.basename(file_path)
        output_file_path = os.path.join(configs.output_dir, f"{pure_file_name}.json")
        process_file(file_path, output_file_path)
        return

    if not configs.data_dir:
        raise ValueError("Either data_path or data_dir must be provided")

    for file_name in os.listdir(configs.data_dir):
        file_path = os.path.join(configs.data_dir, file_name)
        if os.path.isdir(file_path):
            continue
        try:
            pure_file_name = ".".join(file_name.split(".")[:-1]) or file_name
            output_file_path = os.path.join(configs.output_dir, f"{pure_file_name}.json")
            process_file(file_path, output_file_path)
        except ValueError as e:
            if "Unsupported file format" in str(e):
                logger.warning("Skipping unsupported file: %s", file_path)
            else:
                raise


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    configs = parse_args()
    random.seed(configs.seed)
    np.random.seed(configs.seed)

    valid_task_types = {"hierarchicalDataConstruct", "hierarchicalSftDataConstruct"}
    if configs.task_type not in valid_task_types:
        raise ValueError(
            f"Only task_type in {sorted(valid_task_types)} is supported, got: {configs.task_type}"
        )
    hierarchical_data_construct(configs)


if __name__ == "__main__":
    main()
