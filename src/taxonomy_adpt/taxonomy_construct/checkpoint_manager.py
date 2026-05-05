"""
Checkpoint管理模块
用于保存和恢复企业文档分类系统的状态
避免大模型调用卡住导致需要从头开始

所有数据均以人类可阅读的JSON格式保存
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Set
from loguru import logger


class CheckpointManager:
    """Checkpoint管理器"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        初始化Checkpoint管理器
        
        Args:
            checkpoint_dir: checkpoint保存目录
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"[CheckpointManager] 初始化完成, 保存目录: {checkpoint_dir}")
    
    def _serialize_node_iterative(self, root_node) -> Dict:
        """
        将Node对象序列化为JSON可保存的字典（非递归版本）
        使用栈来避免递归深度限制
        
        完整序列化每个节点，不使用引用机制
        如果检测到同一节点被多次序列化，说明存在不合理的共享引用
        
        Args:
            root_node: 根节点
            
        Returns:
            序列化后的字典
        """
        # 用于检测当前维度内的循环引用
        serializing_path = set()  # 当前正在序列化的路径（检测循环）
        
        # 使用栈进行深度优先遍历
        stack = [(root_node, None, None, False)]  # (node, parent_dict, child_label, is_exit)
        result_root = None
        
        while stack:
            node, parent_dict, child_label, is_exit = stack.pop()
            
            if is_exit:
                # 退出节点，从路径中移除
                serializing_path.discard(node.id)
                continue
            
            # 检测循环引用
            if node.id in serializing_path:
                logger.warning(f"[CheckpointManager] 检测到循环引用: 节点 {node.id} ({node.label}) 在序列化路径中已存在")
                logger.warning(f"[CheckpointManager] 这可能导致数据异常，建议检查树结构构建逻辑")
                # 创建一个标记节点而不是引用
                error_dict = {
                    "id": node.id,
                    "label": f"[CIRCULAR_REF] {node.label}",
                    "code": node.code,
                    "description": "检测到循环引用，已跳过",
                    "dimension": node.dimension,
                    "level": node.level,
                    "source": node.source,
                    "parent_ids": [p.id for p in node.parents],
                    "children": {},
                    "paper_ids": []
                }
                # 即使是错误节点，也保留 schema 信息（如果有）
                if hasattr(node, 'element_schema') and node.element_schema:
                    error_dict['element_schema'] = node.element_schema
                    error_dict['schema_reasoning'] = getattr(node, 'schema_reasoning', '')
                    error_dict['schema_complexity'] = getattr(node, 'schema_complexity', 'medium')
                    error_dict['should_distinct'] = getattr(node, 'should_distinct', True)
                    error_dict['refinement_rounds'] = getattr(node, 'refinement_rounds', 0)
                    error_dict['total_changes'] = getattr(node, 'total_changes', 0)
                if hasattr(node, 'region_schema') and node.region_schema:
                    error_dict['region_schema'] = node.region_schema
                    error_dict['region_schema_reasoning'] = getattr(node, 'region_schema_reasoning', '')
                if hasattr(node, 'node_kv_schema') and node.node_kv_schema:
                    error_dict['node_kv_schema'] = node.node_kv_schema
                    error_dict['node_kv_schema_reasoning'] = getattr(node, 'node_kv_schema_reasoning', '')
                if parent_dict is not None and child_label is not None:
                    parent_dict["children"][child_label] = error_dict
                continue
            
            serializing_path.add(node.id)
            
            # 创建节点字典
            node_dict = {
                "id": node.id,
                "label": node.label,
                "code": node.code,
                "description": node.description,
                "dimension": node.dimension,
                "level": node.level,
                "source": node.source,
                "parent_ids": [p.id for p in node.parents],
                "children": {},
                "paper_ids": list(node.papers.keys()) if hasattr(node, 'papers') and node.papers else []
            }
            
            # 序列化 element_schema 相关信息
            if hasattr(node, 'element_schema') and node.element_schema:
                node_dict['element_schema'] = node.element_schema
                node_dict['schema_reasoning'] = getattr(node, 'schema_reasoning', '')
                node_dict['schema_complexity'] = getattr(node, 'schema_complexity', 'medium')
                node_dict['should_distinct'] = getattr(node, 'should_distinct', True)
                node_dict['refinement_rounds'] = getattr(node, 'refinement_rounds', 0)
                node_dict['total_changes'] = getattr(node, 'total_changes', 0)
            
            # 序列化 region_schema 相关信息
            if hasattr(node, 'region_schema') and node.region_schema:
                node_dict['region_schema'] = node.region_schema
                node_dict['region_schema_reasoning'] = getattr(node, 'region_schema_reasoning', '')
            
            # 序列化 node_kv_schema 相关信息
            if hasattr(node, 'node_kv_schema') and node.node_kv_schema:
                node_dict['node_kv_schema'] = node.node_kv_schema
                node_dict['node_kv_schema_reasoning'] = getattr(node, 'node_kv_schema_reasoning', '')
            
            # 如果有父节点，将当前节点添加到父节点的children中
            if parent_dict is not None and child_label is not None:
                parent_dict["children"][child_label] = node_dict
            else:
                # 这是根节点
                result_root = node_dict
            
            # 添加退出标记（用于从路径中移除）
            stack.append((node, None, None, True))
            
            # 将所有子节点压入栈中（逆序以保持遍历顺序）
            for child_label, child_node in reversed(list(node.children.items())):
                stack.append((child_node, node_dict, child_label, False))
        
        return result_root
    
    def _serialize_node(self, node, serializing_path=None) -> Dict:
        """
        将Node对象序列化为JSON可保存的字典（递归版本）
        
        完整序列化每个节点，不使用引用机制
        如果检测到循环引用，会报警并跳过
        
        Args:
            node: Node对象
            serializing_path: 当前序列化路径（用于检测循环引用）
            
        Returns:
            序列化后的字典
        """
        if serializing_path is None:
            serializing_path = set()
        
        # 检测循环引用
        if node.id in serializing_path:
            logger.warning(f"[CheckpointManager] 检测到循环引用: 节点 {node.id} ({node.label}) 在序列化路径中已存在")
            logger.warning(f"[CheckpointManager] 这可能导致数据异常，建议检查树结构构建逻辑")
            error_dict = {
                "id": node.id,
                "label": f"[CIRCULAR_REF] {node.label}",
                "code": node.code,
                "description": "检测到循环引用，已跳过",
                "dimension": node.dimension,
                "level": node.level,
                "source": node.source,
                "parent_ids": [p.id for p in node.parents],
                "children": {},
                "paper_ids": []
            }
            # 即使是错误节点，也保留 schema 信息（如果有）
            if hasattr(node, 'element_schema') and node.element_schema:
                error_dict['element_schema'] = node.element_schema
                error_dict['schema_reasoning'] = getattr(node, 'schema_reasoning', '')
                error_dict['schema_complexity'] = getattr(node, 'schema_complexity', 'medium')
                error_dict['should_distinct'] = getattr(node, 'should_distinct', True)
                error_dict['refinement_rounds'] = getattr(node, 'refinement_rounds', 0)
                error_dict['total_changes'] = getattr(node, 'total_changes', 0)
            if hasattr(node, 'region_schema') and node.region_schema:
                error_dict['region_schema'] = node.region_schema
                error_dict['region_schema_reasoning'] = getattr(node, 'region_schema_reasoning', '')
            if hasattr(node, 'node_kv_schema') and node.node_kv_schema:
                error_dict['node_kv_schema'] = node.node_kv_schema
                error_dict['node_kv_schema_reasoning'] = getattr(node, 'node_kv_schema_reasoning', '')
            return error_dict
        
        serializing_path.add(node.id)
        
        # 序列化节点基本信息
        node_dict = {
            "id": node.id,
            "label": node.label,
            "code": node.code,
            "description": node.description,
            "dimension": node.dimension,
            "level": node.level,
            "source": node.source,
            "parent_ids": [p.id for p in node.parents],  # 只保存父节点ID
            "children": {},  # 递归序列化子节点
            "paper_ids": list(node.papers.keys()) if hasattr(node, 'papers') and node.papers else []
        }
        
        # 序列化 element_schema 相关信息
        if hasattr(node, 'element_schema') and node.element_schema:
            node_dict['element_schema'] = node.element_schema
            node_dict['schema_reasoning'] = getattr(node, 'schema_reasoning', '')
            node_dict['schema_complexity'] = getattr(node, 'schema_complexity', 'medium')
            node_dict['should_distinct'] = getattr(node, 'should_distinct', True)
            node_dict['refinement_rounds'] = getattr(node, 'refinement_rounds', 0)
            node_dict['total_changes'] = getattr(node, 'total_changes', 0)
        
        # 序列化 region_schema 相关信息
        if hasattr(node, 'region_schema') and node.region_schema:
            node_dict['region_schema'] = node.region_schema
            node_dict['region_schema_reasoning'] = getattr(node, 'region_schema_reasoning', '')
        
        # 序列化 node_kv_schema 相关信息
        if hasattr(node, 'node_kv_schema') and node.node_kv_schema:
            node_dict['node_kv_schema'] = node.node_kv_schema
            node_dict['node_kv_schema_reasoning'] = getattr(node, 'node_kv_schema_reasoning', '')
        
        # 递归序列化子节点
        for child_label, child_node in node.children.items():
            node_dict["children"][child_label] = self._serialize_node(child_node, serializing_path)
        
        # 从路径中移除当前节点
        serializing_path.discard(node.id)
        
        return node_dict
    
    def _deserialize_node(self, node_dict: Dict, id2node: Dict, parent_nodes: list = None):
        """
        从字典反序列化Node对象
        
        Args:
            node_dict: 序列化的节点字典
            id2node: ID到节点的映射（用于恢复引用）
            parent_nodes: 父节点列表
            
        Returns:
            Node对象
        """
        # 如果是引用，直接返回已存在的节点
        if "__ref__" in node_dict:
            return id2node.get(node_dict["__ref__"])
        
        from src.taxonomy_adpt.taxonomy_construct.taxonomy import Node
        
        # 创建或获取节点
        node_id = node_dict["id"]
        if node_id in id2node:
            node = id2node[node_id]
        else:
            node = Node(
                id=node_id,
                label=node_dict["label"],
                dimension=node_dict["dimension"],
                code=node_dict.get("code"),
                description=node_dict.get("description"),
                source=node_dict.get("source")
            )
            node.level = node_dict.get("level", 0)
            node.papers = {}  # 文档稍后恢复
            
            # 恢复 element_schema 相关信息
            if 'element_schema' in node_dict:
                node.element_schema = node_dict['element_schema']
                node.schema_reasoning = node_dict.get('schema_reasoning', '')
                node.schema_complexity = node_dict.get('schema_complexity', 'medium')
                node.should_distinct = node_dict.get('should_distinct', True)
                node.refinement_rounds = node_dict.get('refinement_rounds', 0)
                node.total_changes = node_dict.get('total_changes', 0)
            
            # 恢复 region_schema（自动迁移旧格式 → JSON Schema）
            if 'region_schema' in node_dict:
                from src.taxonomy_adpt.taxonomy_construct.taxonomy_io import migrate_legacy_region_schema
                rs = migrate_legacy_region_schema(node_dict['region_schema'])
                if rs:
                    node.region_schema = rs
                    node.region_schema_reasoning = node_dict.get('region_schema_reasoning', '')
            
            # 恢复 node_kv_schema（自动迁移旧格式 → JSON Schema）
            if 'node_kv_schema' in node_dict:
                from src.taxonomy_adpt.taxonomy_construct.taxonomy_io import migrate_legacy_kv_schema
                kv = migrate_legacy_kv_schema(node_dict['node_kv_schema'])
                if kv:
                    node.node_kv_schema = kv
                    node.node_kv_schema_reasoning = node_dict.get('node_kv_schema_reasoning', '')
            
            id2node[node_id] = node
        
        # 设置父节点
        if parent_nodes:
            node.parents = parent_nodes
        
        # 递归反序列化子节点
        for child_label, child_dict in node_dict.get("children", {}).items():
            child_node = self._deserialize_node(child_dict, id2node, parent_nodes=[node])
            node.children[child_label] = child_node
        
        return node
    
    def _serialize_document(self, doc) -> Dict:
        """
        将Document对象序列化为JSON可保存的字典
        
        Args:
            doc: EnterpriseDocument对象
            
        Returns:
            序列化后的字典
        """
        return {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "metadata": doc.metadata,
            "labels": doc.labels,
            "image_url_list": doc.image_url_list
        }
    
    def _deserialize_document(self, doc_dict: Dict):
        """
        从字典反序列化Document对象
        
        Args:
            doc_dict: 序列化的文档字典
            
        Returns:
            EnterpriseDocument对象
        """
        from src.taxonomy_adpt.taxonomy_construct.document import EnterpriseDocument
        
        doc = EnterpriseDocument(
            doc_id=doc_dict["id"],
            title=doc_dict["title"],
            content=doc_dict["content"],
            metadata=doc_dict.get("metadata", {}),
            label_opts=list(doc_dict.get("labels", {}).keys()),
            image_url_list=doc_dict.get("image_url_list", [])
        )
        doc.labels = doc_dict.get("labels", {})
        return doc
    
    def save_checkpoint(
        self,
        roots: Dict,
        id2node: Dict,
        label2node: Dict,
        documents: Dict,
        visited: Set,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        保存当前状态的checkpoint
        
        Args:
            roots: 根节点字典
            id2node: ID到节点的映射
            label2node: 标签到节点的映射
            documents: 文档字典
            visited: 已访问节点集合
            iteration: 当前迭代次数
            metadata: 额外的元数据
            
        Returns:
            checkpoint文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_iter{iteration}_{timestamp}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # 创建checkpoint目录
        os.makedirs(checkpoint_path, exist_ok=True)
        
        try:
            # 1. 序列化并保存根节点（JSON格式）
            # 每个维度独立完整序列化，不使用引用机制
            roots_dict = {}
            
            logger.info(f"[CheckpointManager] 开始序列化{len(roots)}个维度的根节点...")
            logger.info(f"[CheckpointManager] 注意：使用完整序列化模式，不使用 __ref__ 引用")
            
            # 检测树的深度，决定使用哪种序列化方式
            max_depth = max(node.level for node in id2node.values()) if id2node else 0
            use_iterative = max_depth > 50  # 如果树深度超过50层，直接使用迭代方式
            
            if use_iterative:
                logger.info(f"[CheckpointManager] 检测到深层树结构（最大深度: {max_depth}），使用迭代序列化方式")
            
            for dim, root_node in roots.items():
                logger.info(f"[CheckpointManager] 序列化维度: {dim}, 根节点: {root_node.label}")
                
                if use_iterative:
                    # 直接使用迭代方式
                    roots_dict[dim] = self._serialize_node_iterative(root_node)
                    logger.info(f"[CheckpointManager] 维度 {dim} 序列化完成（迭代模式）")
                else:
                    # 尝试递归方式，失败则切换到迭代
                    try:
                        roots_dict[dim] = self._serialize_node(root_node)
                        logger.info(f"[CheckpointManager] 维度 {dim} 序列化完成")
                    except RecursionError:
                        logger.warning(f"[CheckpointManager] 递归深度超限，切换到非递归模式")
                        roots_dict[dim] = self._serialize_node_iterative(root_node)
                        logger.info(f"[CheckpointManager] 维度 {dim} 序列化完成（迭代模式）")
            
            with open(os.path.join(checkpoint_path, "roots.json"), 'w', encoding='utf-8') as f:
                json.dump(roots_dict, f, ensure_ascii=False, indent=2)
            
            # 2. 保存id2node映射（JSON格式）
            # 只保存ID列表，因为完整节点信息已在roots中
            id2node_dict = {
                node_id: {
                    "label": node.label,
                    "dimension": node.dimension,
                    "code": node.code
                } for node_id, node in id2node.items()
            }
            with open(os.path.join(checkpoint_path, "id2node.json"), 'w', encoding='utf-8') as f:
                json.dump(id2node_dict, f, ensure_ascii=False, indent=2)
            
            # 3. 保存label2node映射（JSON格式）
            label2node_dict = {
                label: node.id for label, node in label2node.items()
            }
            with open(os.path.join(checkpoint_path, "label2node.json"), 'w', encoding='utf-8') as f:
                json.dump(label2node_dict, f, ensure_ascii=False, indent=2)
            
            # 4. 序列化并保存文档（JSON格式）
            documents_dict = {
                doc_id: self._serialize_document(doc) 
                for doc_id, doc in documents.items()
            }
            with open(os.path.join(checkpoint_path, "documents.json"), 'w', encoding='utf-8') as f:
                json.dump(documents_dict, f, ensure_ascii=False, indent=2)
            
            # 5. 保存visited集合（JSON格式）
            visited_list = list(visited)
            with open(os.path.join(checkpoint_path, "visited.json"), 'w', encoding='utf-8') as f:
                json.dump(visited_list, f, ensure_ascii=False, indent=2)
            
            # 6. 保存元数据（JSON格式）
            checkpoint_metadata = {
                "iteration": iteration,
                "timestamp": timestamp,
                "num_documents": len(documents),
                "num_nodes": len(id2node),
                "num_visited": len(visited),
                "dimensions": list(roots.keys()) if roots else [],
                "format_version": "2.0",  # JSON格式版本标识
                "storage_format": "json"
            }
            
            if metadata:
                checkpoint_metadata.update(metadata)
            
            with open(os.path.join(checkpoint_path, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2)
            
            # 7. 保存分类体系快照（文本格式，便于调试）
            for dim, root in roots.items():
                snapshot_path = os.path.join(checkpoint_path, f"taxonomy_snapshot_{dim}.txt")
                try:
                    with open(snapshot_path, 'w', encoding='utf-8') as f:
                        f.write(f"=== 分类体系快照 - {dim} 维度 ===\n")
                        f.write(f"时间: {timestamp}\n")
                        f.write(f"迭代: {iteration}\n")
                        f.write(f"节点数: {len([n for n in id2node.values() if n.dimension == dim])}\n\n")
                        # 简单的树形展示 - 使用迭代方式避免递归深度限制
                        self._write_tree_snapshot_iterative(f, root)
                except RecursionError:
                    logger.warning(f"[CheckpointManager] 快照生成递归深度超限，跳过维度 {dim} 的快照")
                    # 快照不是必需的，失败了也没关系
                except Exception as e:
                    logger.warning(f"[CheckpointManager] 快照生成失败: {e}")
                    # 快照不是必需的，失败了也没关系
            
            logger.info(f"[CheckpointManager] 已保存 checkpoint: {checkpoint_path}")
            logger.info(f"[CheckpointManager] 当前状态: 迭代{iteration}, {len(documents)}份文档, {len(id2node)}个节点, {len(visited)}个已访问")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"[CheckpointManager] 保存checkpoint失败: {e}")
            raise
    
    def _write_tree_snapshot_iterative(self, f, root_node):
        """
        使用迭代方式写入树形结构快照，避免递归深度限制
        
        Args:
            f: 文件对象
            root_node: 根节点
        """
        # 使用栈来模拟递归，栈中存储 (node, indent_level)
        stack = [(root_node, 0)]
        
        while stack:
            node, indent_level = stack.pop()
            
            # 写入当前节点信息
            indent = "  " * indent_level
            doc_count = len(node.papers) if hasattr(node, 'papers') and node.papers else 0
            f.write(f"{indent}- {node.label} (ID: {node.id}, 文档数: {doc_count})\n")
            
            # 将子节点压入栈（逆序以保持正确的输出顺序）
            if hasattr(node, 'children') and node.children:
                for child_label in reversed(list(node.children.keys())):
                    child = node.children[child_label]
                    stack.append((child, indent_level + 1))
    
    def _write_tree_snapshot(self, f, node, indent_level):
        """递归写入树形结构快照（保留用于向后兼容）"""
        indent = "  " * indent_level
        doc_count = len(node.papers) if hasattr(node, 'papers') else 0
        f.write(f"{indent}- {node.label} (ID: {node.id}, 文档数: {doc_count})\n")
        
        if hasattr(node, 'children') and node.children:
            for child_label, child in node.children.items():
                self._write_tree_snapshot(f, child, indent_level + 1)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        从checkpoint恢复状态
        
        Args:
            checkpoint_path: checkpoint目录路径
            
        Returns:
            包含所有状态的字典
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
        
        try:
            logger.info(f"[CheckpointManager] 正在加载 checkpoint: {checkpoint_path}")
            
            # 检查是否为新版本JSON格式
            metadata_path = os.path.join(checkpoint_path, "metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            storage_format = metadata.get("storage_format", "pickle")
            
            if storage_format == "json":
                # 新版本JSON格式
                logger.info(f"[CheckpointManager] 检测到JSON格式checkpoint")
                
                # 1. 加载并反序列化根节点
                with open(os.path.join(checkpoint_path, "roots.json"), 'r', encoding='utf-8') as f:
                    roots_dict = json.load(f)
                
                id2node = {}
                roots = {}
                for dim, root_dict in roots_dict.items():
                    roots[dim] = self._deserialize_node(root_dict, id2node)
                
                # 2. 构建label2node映射
                with open(os.path.join(checkpoint_path, "label2node.json"), 'r', encoding='utf-8') as f:
                    label2node_dict = json.load(f)
                
                label2node = {
                    label: id2node[node_id] 
                    for label, node_id in label2node_dict.items()
                }
                
                # 3. 加载并反序列化文档
                with open(os.path.join(checkpoint_path, "documents.json"), 'r', encoding='utf-8') as f:
                    documents_dict = json.load(f)
                
                documents = {
                    doc_id: self._deserialize_document(doc_dict)
                    for doc_id, doc_dict in documents_dict.items()
                }
                
                # 4. 恢复节点的papers关联
                for node in id2node.values():
                    node.papers = {}
                
                for doc_id, doc in documents.items():
                    for dim, labels in doc.labels.items():
                        for label in labels:
                            full_label = f"{label}_{dim}"
                            if full_label in label2node:
                                label2node[full_label].papers[doc_id] = doc
                
                # 5. 加载visited集合
                with open(os.path.join(checkpoint_path, "visited.json"), 'r', encoding='utf-8') as f:
                    visited_list = json.load(f)
                visited = set(visited_list)
                
            else:
                # 旧版本pickle格式（向后兼容）
                logger.info(f"[CheckpointManager] 检测到pickle格式checkpoint（旧版本）")
                import pickle
                
                with open(os.path.join(checkpoint_path, "roots.pkl"), 'rb') as f:
                    roots = pickle.load(f)
                
                with open(os.path.join(checkpoint_path, "id2node.pkl"), 'rb') as f:
                    id2node = pickle.load(f)
                
                with open(os.path.join(checkpoint_path, "label2node.pkl"), 'rb') as f:
                    label2node = pickle.load(f)
                
                with open(os.path.join(checkpoint_path, "documents.pkl"), 'rb') as f:
                    documents = pickle.load(f)
                
                with open(os.path.join(checkpoint_path, "visited.pkl"), 'rb') as f:
                    visited = pickle.load(f)
            
            logger.info(f"[CheckpointManager] 成功加载 checkpoint")
            logger.info(f"[CheckpointManager] 恢复状态: 迭代{metadata.get('iteration', 0)}, "
                       f"{len(documents)}份文档, {len(id2node)}个节点, {len(visited)}个已访问")
            
            return {
                "roots": roots,
                "id2node": id2node,
                "label2node": label2node,
                "documents": documents,
                "visited": visited,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"[CheckpointManager] 加载checkpoint失败: {e}")
            raise
    
    def list_checkpoints(self) -> list:
        """
        列出所有可用的checkpoint
        
        Returns:
            checkpoint列表，按时间倒序排列
        """
        checkpoints = []
        
        if not os.path.exists(self.checkpoint_dir):
            return checkpoints
        
        for item in os.listdir(self.checkpoint_dir):
            checkpoint_path = os.path.join(self.checkpoint_dir, item)
            if os.path.isdir(checkpoint_path):
                metadata_path = os.path.join(checkpoint_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        checkpoints.append({
                            "name": item,
                            "path": checkpoint_path,
                            "metadata": metadata
                        })
                    except Exception as e:
                        logger.warning(f"[CheckpointManager] 无法读取checkpoint元数据: {checkpoint_path}, {e}")
        
        # 按时间戳倒序排列
        checkpoints.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新的checkpoint路径
        
        Returns:
            最新checkpoint的路径，如果没有则返回None
        """
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]['path']
        return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        清理旧的checkpoint，只保留最近的N个
        
        Args:
            keep_last_n: 保留的checkpoint数量
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            logger.info(f"[CheckpointManager] 当前有{len(checkpoints)}个checkpoint，无需清理")
            return
        
        # 删除多余的checkpoint
        to_delete = checkpoints[keep_last_n:]
        for checkpoint in to_delete:
            try:
                import shutil
                shutil.rmtree(checkpoint['path'])
                logger.info(f"[CheckpointManager] 已删除旧checkpoint: {checkpoint['name']}")
            except Exception as e:
                logger.warning(f"[CheckpointManager] 删除checkpoint失败: {checkpoint['path']}, {e}")
        
        logger.info(f"[CheckpointManager] 清理完成，保留了{keep_last_n}个最新checkpoint")


def auto_save_checkpoint(
    checkpoint_manager: CheckpointManager,
    roots: Dict,
    id2node: Dict,
    label2node: Dict,
    documents: Dict,
    visited: Set,
    iteration: int,
    save_interval: int = 10,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    根据迭代间隔自动保存checkpoint
    
    Args:
        checkpoint_manager: checkpoint管理器
        roots, id2node, label2node, documents, visited: 系统状态
        iteration: 当前迭代次数
        save_interval: 保存间隔（每N次迭代保存一次）
        metadata: 额外元数据
        
    Returns:
        如果保存了checkpoint，返回路径；否则返回None
    """
    if iteration % save_interval == 0 or iteration == 1:
        try:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                roots=roots,
                id2node=id2node,
                label2node=label2node,
                documents=documents,
                visited=visited,
                iteration=iteration,
                metadata=metadata
            )
            return checkpoint_path
        except Exception as e:
            logger.error(f"[CheckpointManager] 自动保存失败: {e}")
            return None
    return None
