#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将嵌套格式的 taxonomy_structure.json 转换为扁平格式。
扁平格式中每个节点通过 label_parent 字段引用其父节点的 code，
而非通过 children 嵌套。
"""

import argparse
import json
import sys
from pathlib import Path


def nested_to_flat(taxonomy_structure: dict) -> dict:
    """
    将嵌套格式的分类体系 dict 转换为扁平格式。

    Args:
        taxonomy_structure: 嵌套格式 {dim: {id, label, code, children: {code: {…}}, …}}

    Returns:
        扁平格式 {code: {label_code, label_name, label_parent, …}}
    """
    flat_result = {}

    def flatten_node(node_dict: dict, parent_code: str = ""):
        code = node_dict.get('code', '')
        children = node_dict.get('children', {})

        flat_entry = {
            'label_code': code,
            'label_name': node_dict.get('label', ''),
            'label_parent': parent_code,
            'label_desc': node_dict.get('description', ''),
            'label_keys': '',
            'label_code_children': list(children.keys()),
            'label_name_children': [c.get('label', '') for c in children.values()],
            'level': node_dict.get('level', 0),
            'is_leaf': len(children) == 0,
        }

        if 'region_schema' in node_dict and node_dict['region_schema']:
            flat_entry['region_schema'] = node_dict['region_schema']
            flat_entry['region_schema_reasoning'] = node_dict.get('region_schema_reasoning', '')

        if 'node_kv_schema' in node_dict and node_dict['node_kv_schema']:
            flat_entry['node_kv_schema'] = node_dict['node_kv_schema']
            flat_entry['node_kv_schema_reasoning'] = node_dict.get('node_kv_schema_reasoning', '')

        flat_result[code] = flat_entry

        for child_dict in children.values():
            flatten_node(child_dict, parent_code=code)

    for _dim, root_dict in taxonomy_structure.items():
        flatten_node(root_dict, parent_code="")

    return flat_result


def flatten_taxonomy_json(input_path: str, output_path: str):
    """
    读取嵌套格式的 taxonomy JSON 文件，输出扁平格式的 JSON 文件。

    Args:
        input_path: 嵌套格式的 taxonomy_structure.json 路径
        output_path: 输出的扁平格式 JSON 路径
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        nested = json.load(f)

    flat = nested_to_flat(nested)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    print(f"扁平格式分类体系已导出到: {output_path}")
    print(f"  - 总节点数: {len(flat)}")
    leaf_count = sum(1 for v in flat.values() if v.get('is_leaf'))
    print(f"  - 叶子节点数: {leaf_count}")


def main():
    parser = argparse.ArgumentParser(
        description='将嵌套格式 taxonomy_structure.json 转换为扁平格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python flatten_taxonomy.py --input taxonomy_structure.json --output flat_taxonomy.json
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='嵌套格式的 taxonomy_structure.json 路径'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='输出的扁平格式 JSON 路径（默认在输入文件同目录下生成 *_flat.json）'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    if args.output is None:
        output_path = input_path.with_name(input_path.stem + '_flat.json')
    else:
        output_path = Path(args.output)

    flatten_taxonomy_json(str(input_path), str(output_path))


if __name__ == '__main__':
    main()
