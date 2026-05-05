"""
分类体系的导入导出功能
支持保存/加载树结构，用于复用已有的分类体系
"""

import json
import pickle
import os
from typing import Dict, Any, Optional
from src.taxonomy_adpt.taxonomy_construct.taxonomy import Node


# ============================================================
# Legacy Schema → JSON Schema 迁移工具
# ============================================================

def is_json_schema(schema) -> bool:
    """判断 schema 是否已经是 JSON Schema 格式"""
    return (
        isinstance(schema, dict)
        and schema.get("type") in ("object", "array", "string")
        and ("properties" in schema or "items" in schema or schema.get("type") == "string")
    )


def _migrate_kv_value(value):
    """将旧格式的 kv value 占位符转为 JSON Schema 类型定义"""
    if isinstance(value, str):
        return {"type": "string"}
    elif isinstance(value, list):
        if len(value) == 0:
            return {"type": "array", "items": {"type": "string"}}
        first = value[0]
        if isinstance(first, str):
            return {"type": "array", "items": {"type": "string"}}
        elif isinstance(first, dict):
            return {
                "type": "array",
                "items": _migrate_legacy_kv_schema(first),
            }
        return {"type": "array", "items": {"type": "string"}}
    elif isinstance(value, dict):
        return _migrate_legacy_kv_schema(value)
    return {"type": "string"}


def _migrate_legacy_kv_schema(kv: dict) -> dict:
    """
    旧格式 node_kv_schema (flat dict with placeholder values) → JSON Schema object。
    如 {"姓名": "", "明细": [{"品名": ""}]} → {"type":"object","properties":{...}}
    """
    if is_json_schema(kv):
        return kv
    props = {}
    for key, val in kv.items():
        props[key] = _migrate_kv_value(val)
    return {"type": "object", "properties": props}


def _migrate_region_list(regions: list) -> dict:
    """
    旧格式 region_schema (list of region dicts) → JSON Schema object。
    每个 region 有 name/description + children|kv_schema。
    """
    props = {}
    for region in regions:
        if not isinstance(region, dict):
            continue
        name = region.get("name") or region.get("region_name", "")
        if not name:
            continue
        desc = region.get("description", "")
        children = region.get("children")
        kv = region.get("kv_schema")

        if children and isinstance(children, list):
            sub_schema = _migrate_region_list(children)
            if desc:
                sub_schema["description"] = desc
            props[name] = sub_schema
        elif kv and isinstance(kv, dict):
            leaf_schema = _migrate_legacy_kv_schema(kv)
            if desc:
                leaf_schema["description"] = desc
            props[name] = leaf_schema
        else:
            entry: dict = {"type": "object"}
            if desc:
                entry["description"] = desc
            props[name] = entry
    return {"type": "object", "properties": props}


def migrate_legacy_region_schema(schema) -> Optional[dict]:
    """
    将任意格式的 region_schema 迁移为 JSON Schema dict。
    支持：list、{"regions": [...]}, 以及已经是 JSON Schema 的 dict。
    返回 None 如果输入为空/无效。
    """
    if schema is None:
        return None
    if is_json_schema(schema):
        return schema
    if isinstance(schema, dict) and "regions" in schema:
        schema = schema["regions"]
    if isinstance(schema, list):
        return _migrate_region_list(schema)
    return None


def migrate_legacy_kv_schema(schema) -> Optional[dict]:
    """
    将任意格式的 node_kv_schema 迁移为 JSON Schema dict。
    支持：旧 flat dict 和已有 JSON Schema dict。
    返回 None 如果输入为空/无效。
    """
    if schema is None:
        return None
    if is_json_schema(schema):
        return schema
    if isinstance(schema, dict):
        return _migrate_legacy_kv_schema(schema)
    return None


def _flat_to_nested(flat_data):
    """将扁平格式转换为嵌套格式"""
    # 如果已经是嵌套格式，直接返回
    if 'all_nodes' not in flat_data and 'dimensions' not in flat_data:
        return flat_data
    
    # 构建 id 到节点数据的映射
    id2data = {}
    code2data = {}
    for node_data in flat_data.get('all_nodes', []):
        id2data[node_data['id']] = node_data
        if node_data.get('code'):
            code2data[node_data['code']] = node_data
    
    def build_nested_node(node_data):
        """递归构建嵌套节点"""
        nested_node = {
            'id': node_data['id'],
            'label': node_data['label'],
            'code': node_data.get('code'),
            'dimension': node_data['dimension'],
            'description': node_data.get('description'),
            'level': node_data.get('level', 0),
            'source': node_data.get('source'),
            'children': {}
        }
        
        # 复制 Region Schema（自动迁移旧格式 → JSON Schema）
        if 'region_schema' in node_data:
            rs = migrate_legacy_region_schema(node_data['region_schema'])
            if rs:
                nested_node['region_schema'] = rs
                nested_node['region_schema_reasoning'] = node_data.get('region_schema_reasoning', '')
        
        # 复制 node_kv_schema（自动迁移旧格式 → JSON Schema）
        if 'node_kv_schema' in node_data:
            kv = migrate_legacy_kv_schema(node_data['node_kv_schema'])
            if kv:
                nested_node['node_kv_schema'] = kv
                nested_node['node_kv_schema_reasoning'] = node_data.get('node_kv_schema_reasoning', '')
        
        # [兼容] 旧版 element_schema
        if 'element_schema' in node_data:
            nested_node['element_schema'] = node_data['element_schema']
            nested_node['schema_reasoning'] = node_data.get('schema_reasoning', '')
            nested_node['schema_complexity'] = node_data.get('schema_complexity', 'medium')
            nested_node['should_distinct'] = node_data.get('should_distinct', True)
            nested_node['refinement_rounds'] = node_data.get('refinement_rounds', 0)
            nested_node['total_changes'] = node_data.get('total_changes', 0)
        
        # 递归处理子节点（children 是 code 列表）
        for child_code in node_data.get('children', []):
            if child_code in code2data:
                child_data = code2data[child_code]
                nested_node['children'][child_code] = build_nested_node(child_data)
        
        return nested_node
    
    # 构建嵌套结构
    nested_structure = {}
    for dim_data in flat_data.get('dimensions', []):
        dim_name = dim_data['name']
        root_id = dim_data['root_id']
        if root_id in id2data:
            nested_structure[dim_name] = build_nested_node(id2data[root_id])
    
    return nested_structure


def export_taxonomy_structure(roots: Dict[str, Node], output_path: str, format: str = 'json'):
    """
    导出分类体系结构（不包含文档数据）
    
    Args:
        roots: 各维度的根节点字典
        output_path: 输出文件路径
        format: 输出格式 ('json' 或 'pickle')
    """
    def node_to_dict(node: Node) -> Dict[str, Any]:
        """将节点转换为字典（递归）"""
        node_dict = {
            'id': node.id,
            'label': node.label,
            'code': node.code,
            'dimension': node.dimension,
            'description': node.description,
            'level': node.level,
            'source': node.source,
            'children': {}
        }
        
        # 语义 Region Schema（仅叶子节点）
        if hasattr(node, 'region_schema') and node.region_schema:
            node_dict['region_schema'] = node.region_schema
            node_dict['region_schema_reasoning'] = getattr(node, 'region_schema_reasoning', '')
        
        # 节点级 kv_schema（所有节点都可能有）
        if hasattr(node, 'node_kv_schema') and node.node_kv_schema:
            node_dict['node_kv_schema'] = node.node_kv_schema
            node_dict['node_kv_schema_reasoning'] = getattr(node, 'node_kv_schema_reasoning', '')
        
        # 递归处理子节点
        for child_key, child_node in node.children.items():
            node_dict['children'][child_key] = node_to_dict(child_node)
        
        return node_dict
    
    # 转换所有根节点
    taxonomy_structure = {}
    for dim, root in roots.items():
        taxonomy_structure[dim] = node_to_dict(root)
    
    # 保存
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(taxonomy_structure, f, ensure_ascii=False, indent=2)
    elif format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(taxonomy_structure, f)
    else:
        raise ValueError(f"不支持的格式: {format}，仅支持 'json' 或 'pickle'")
    
    print(f"分类体系结构已导出到: {output_path}")
    
    # 统计信息
    total_nodes = sum(_count_nodes(root) for root in roots.values())
    print(f"  - 维度数: {len(roots)}")
    print(f"  - 总节点数: {total_nodes}")
    for dim, root in roots.items():
        print(f"  - {dim}: {_count_nodes(root)} 个节点, 最大深度: {_max_depth(root)}")


def import_taxonomy_structure(input_path: str, format: str = None) -> Dict[str, Node]:
    """
    导入分类体系结构
    
    Args:
        input_path: 输入文件路径
        format: 输入格式 ('json' 或 'pickle')，如果为None则自动判断
        
    Returns:
        roots: 各维度的根节点字典
        id2node: 节点ID到节点的映射
        label2node: 标签到节点的映射
    """
    # 自动判断格式
    if format is None:
        if input_path.endswith('.json'):
            format = 'json'
        elif input_path.endswith('.pkl') or input_path.endswith('.pickle'):
            format = 'pickle'
        else:
            raise ValueError(f"无法自动判断文件格式，请指定 format 参数")
    
    # 加载数据
    if format == 'json':
        with open(input_path, 'r', encoding='utf-8') as f:
            taxonomy_structure = json.load(f)
    elif format == 'pickle':
        with open(input_path, 'rb') as f:
            taxonomy_structure = pickle.load(f)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    # 检测格式并转换（如果需要）
    if 'all_nodes' in taxonomy_structure or 'dimensions' in taxonomy_structure:
        # 扁平格式，需要转换为嵌套格式
        print("检测到扁平格式，正在转换为嵌套格式...")
        taxonomy_structure = _flat_to_nested(taxonomy_structure)
    
    def dict_to_node(node_dict: Dict[str, Any], parent_nodes=None, id2node=None) -> Node:
        """将字典转换为节点（递归）"""
        if parent_nodes is None:
            parent_nodes = []
        if id2node is None:
            id2node = {}
        
        # 如果是引用，直接返回已存在的节点
        if "__ref__" in node_dict:
            ref_id = node_dict["__ref__"]
            if ref_id in id2node:
                return id2node[ref_id]
            else:
                raise ValueError(f"引用的节点ID {ref_id} 不存在，可能文件结构有误")
        
        # 检查节点是否已存在（避免重复创建）
        node_id = node_dict['id']
        if node_id in id2node:
            node = id2node[node_id]
            # 更新父节点（如果有新的父节点）
            if parent_nodes:
                for parent in parent_nodes:
                    if parent not in node.parents:
                        node.parents.append(parent)
            return node
        
        # 创建新节点
        node = Node(
            id=node_id,
            label=node_dict['label'],
            dimension=node_dict['dimension'],
            code=node_dict.get('code'),
            description=node_dict.get('description'),
            parents=parent_nodes,
            source=node_dict.get('source')
        )
        node.level = node_dict.get('level', 0)
        
        # 加载 Region Schema（自动迁移旧格式 → JSON Schema）
        if 'region_schema' in node_dict:
            rs = migrate_legacy_region_schema(node_dict['region_schema'])
            if rs:
                node.region_schema = rs
                node.region_schema_reasoning = node_dict.get('region_schema_reasoning', '')
        
        # 加载 node_kv_schema（自动迁移旧格式 → JSON Schema）
        if 'node_kv_schema' in node_dict:
            kv = migrate_legacy_kv_schema(node_dict['node_kv_schema'])
            if kv:
                node.node_kv_schema = kv
                node.node_kv_schema_reasoning = node_dict.get('node_kv_schema_reasoning', '')
        
        # [兼容] 旧版 element_schema
        if 'element_schema' in node_dict:
            node.element_schema = node_dict['element_schema']
            node.schema_reasoning = node_dict.get('schema_reasoning', '')
            node.schema_complexity = node_dict.get('schema_complexity', 'medium')
            node.should_distinct = node_dict.get('should_distinct', True)
            node.refinement_rounds = node_dict.get('refinement_rounds', 0)
            node.total_changes = node_dict.get('total_changes', 0)
        
        # 先将节点加入映射，以支持循环引用
        id2node[node_id] = node
        
        # 递归处理子节点
        for child_key, child_dict in node_dict.get('children', {}).items():
            child_node = dict_to_node(child_dict, parent_nodes=[node], id2node=id2node)
            node.children[child_key] = child_node
        
        return node
    
    # 转换所有根节点
    roots = {}
    id2node = {}
    label2node = {}
    
    for dim, root_dict in taxonomy_structure.items():
        root = dict_to_node(root_dict, parent_nodes=None, id2node=id2node)
        roots[dim] = root
        
        # 构建映射
        _build_mappings(root, id2node, label2node)
    
    # 检查并修复id重复问题
    id_duplicates = _check_and_fix_duplicate_ids(roots, id2node)
    if id_duplicates:
        print(f"  ⚠️ 警告: 检测到 {len(id_duplicates)} 个重复的id，已自动修复")
        print(f"     重复的id: {sorted(id_duplicates)}")
    
    print(f"分类体系结构已导入: {input_path}")
    print(f"  - 维度数: {len(roots)}")
    print(f"  - 总节点数: {len(id2node)}")
    for dim, root in roots.items():
        print(f"  - {dim}: {_count_nodes(root)} 个节点, 最大深度: {_max_depth(root)}")
    
    return roots, id2node, label2node


def _check_and_fix_duplicate_ids(roots: Dict[str, Node], id2node: Dict[int, Node]) -> set:
    """
    检查并修复重复的节点ID
    
    Args:
        roots: 各维度的根节点字典
        id2node: 节点ID到节点的映射
    
    Returns:
        重复的id集合
    """
    # 收集所有节点及其id
    all_nodes = []
    id_to_nodes = {}  # id -> list of nodes
    
    def collect_nodes(node):
        all_nodes.append(node)
        if node.id not in id_to_nodes:
            id_to_nodes[node.id] = []
        id_to_nodes[node.id].append(node)
        
        for child in node.children.values():
            collect_nodes(child)
    
    for root in roots.values():
        collect_nodes(root)
    
    # 找出重复的id
    duplicate_ids = {nid for nid, nodes in id_to_nodes.items() if len(nodes) > 1}
    
    if duplicate_ids:
        # 找到当前最大的id
        max_id = max(id2node.keys()) if id2node else -1
        
        # 为重复的节点重新分配id
        for dup_id in sorted(duplicate_ids):
            nodes_with_dup_id = id_to_nodes[dup_id]
            # 保留第一个节点的id，为其他节点重新分配
            for i, node in enumerate(nodes_with_dup_id[1:], 1):
                max_id += 1
                old_id = node.id
                node.id = max_id
                # 更新id2node映射
                if old_id in id2node and id2node[old_id] == node:
                    del id2node[old_id]
                id2node[max_id] = node
                print(f"     节点 '{node.label}' (code={node.code}) 的id从 {old_id} 更改为 {max_id}")
    
    return duplicate_ids


def _build_mappings(node: Node, id2node: Dict[int, Node], label2node: Dict[str, Node]):
    """递归构建ID和标签映射"""
    # 检查id是否已存在
    if node.id in id2node and id2node[node.id] != node:
        print(f"  ⚠️ 警告: 检测到重复的id {node.id}，节点: {node.label} 和 {id2node[node.id].label}")
    
    id2node[node.id] = node
    full_label = f"{node.code}_{node.dimension}"
    label2node[full_label] = node
    
    for child in node.children.values():
        _build_mappings(child, id2node, label2node)


def _count_nodes(node: Node) -> int:
    """递归统计节点数量"""
    count = 1
    for child in node.children.values():
        count += _count_nodes(child)
    return count


def _max_depth(node: Node) -> int:
    """递归计算最大深度"""
    if not node.children:
        return node.level
    return max(_max_depth(child) for child in node.children.values())


def export_taxonomy_flat(roots: Dict[str, Node], output_path: str):
    """
    将 Node 树导出为扁平格式的 JSON 文件。
    每个节点通过 label_parent 引用父节点 code，而非嵌套 children。

    Args:
        roots: 各维度的根节点字典
        output_path: 输出文件路径
    """
    from src.taxonomy_adpt.flatten_taxonomy import nested_to_flat

    def node_to_dict(node: Node) -> Dict[str, Any]:
        node_dict = {
            'id': node.id,
            'label': node.label,
            'code': node.code,
            'dimension': node.dimension,
            'description': node.description,
            'level': node.level,
            'source': node.source,
            'children': {}
        }
        if hasattr(node, 'region_schema') and node.region_schema:
            node_dict['region_schema'] = node.region_schema
            node_dict['region_schema_reasoning'] = getattr(node, 'region_schema_reasoning', '')
        if hasattr(node, 'node_kv_schema') and node.node_kv_schema:
            node_dict['node_kv_schema'] = node.node_kv_schema
            node_dict['node_kv_schema_reasoning'] = getattr(node, 'node_kv_schema_reasoning', '')
        for child_key, child_node in node.children.items():
            node_dict['children'][child_key] = node_to_dict(child_node)
        return node_dict

    nested = {dim: node_to_dict(root) for dim, root in roots.items()}
    flat = nested_to_flat(nested)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    print(f"扁平格式分类体系已导出到: {output_path}")
    print(f"  - 总节点数: {len(flat)}")
    leaf_count = sum(1 for v in flat.values() if v.get('is_leaf'))
    print(f"  - 叶子节点数: {leaf_count}")


def print_taxonomy_structure(roots: Dict[str, Node], max_depth: int = None):
    """
    打印分类体系结构
    
    Args:
        roots: 各维度的根节点字典
        max_depth: 最大打印深度（None表示全部打印）
    """
    def print_node(node: Node, indent: int = 0, max_d: int = None):
        """递归打印节点"""
        if max_d is not None and indent > max_d:
            return
        
        prefix = "  " * indent
        print(f"{prefix}- {node.label} ({node.code}) [level={node.level}, children={len(node.children)}]")
        
        if node.description:
            print(f"{prefix}  描述: {node.description[:50]}...")
        
        for child in node.children.values():
            print_node(child, indent + 1, max_d)
    
    print("\n分类体系结构:")
    print("=" * 80)
    for dim, root in roots.items():
        print(f"\n【维度: {dim}】")
        print_node(root, 0, max_depth)
    print("=" * 80)
