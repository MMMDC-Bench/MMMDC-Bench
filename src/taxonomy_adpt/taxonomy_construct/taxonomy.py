from collections import deque
from src.taxonomy_adpt.taxonomy_construct.enrichment import enrich_node_prompt
from src.taxonomy_adpt.taxonomy_construct.classification import classify_prompt
from src.taxonomy_adpt.llm_client.llm_adapter import promptLLM
from src.taxonomy_adpt.taxonomy_construct.prompts import EnrichSchema
import json
from src.taxonomy_adpt.taxonomy_construct.utils import clean_json_string, safe_json_loads_list
try:
    from unidecode import unidecode
except ImportError:
    def unidecode(text):
        return str(text)

from src.taxonomy_adpt.taxonomy_construct.classification import ClassifySchema


class Node:
    def __init__(self, id, label, dimension, code=None, description=None, children=None, parents=None, source=None):
        """
        Initialize a Node based on the provided JSON schema.

        Args:
        id (int): Unique identifier for the node.
        label (str): The label/name for the node (can be Chinese).
        dimension (str): Type of node. Examples are: task, dataset, methodology, evaluation method, application
        code (str, optional): Unique code for the node (lowercase with underscores). If not provided, will be generated from label.
        description (str): Description of the node.
        children (dict, optional): A dictionary of children nodes, where keys are labels and values are Node instances.
        parents (list of Node, optional): A list of parent nodes of the current node.
        source (str, optional): Source of the node.
        """
        self.id = id
        self.label = label
        self.code = code if code else self._generate_code_from_label(label)
        self.description = description
        self.dimension = dimension
        self.children = children if children else {}
        self.parents = parents if parents else []
        self.level = 0 if not self.parents else max(parent.level for parent in self.parents) + 1

        self.papers = {}
        self.source = source
        
        # 企业文档专用：要素Schema
        self.element_schema = None  # 该节点对应文档类型的核心要素Schema
        self.schema_reasoning = None  # Schema设计思路
        self.schema_complexity = None  # Schema复杂度
        self.should_distinct = True  # 是否需要与兄弟节点区分
        
        # 迭代式Schema生成的元信息
        self.refinement_rounds = 0  # Schema经过的修正轮数
        self.total_changes = 0  # 修正过程中的总改动次数
        
        # 语义 Region Schema：JSON Schema 格式的区域结构定义（仅叶子节点）
        # {"type": "object", "properties": {区域名: {区域schema...}, ...}}
        self.region_schema = None
        self.region_schema_reasoning = None

        # 节点级抽取要素 Schema：JSON Schema 格式（所有节点都有）
        # 叶子节点：从 region_schema 扁平化得到
        # 非叶子节点：从子节点的 node_kv_schema 抽象概括得到
        self.node_kv_schema = None
        self.node_kv_schema_reasoning = None
    
    # ------ region_schema 辅助方法 ------

    def get_flat_kv_schema(self):
        """
        从 region_schema（JSON Schema 格式）递归收集所有叶子区域的字段定义，
        返回 dict[region_path, properties_dict]，其中 region_path 形如 "协议方信息/甲方信息"。
        若节点尚未生成 region_schema 则返回空 dict。

        兼容旧格式（list of dicts）和新格式（JSON Schema object）。
        """
        if not self.region_schema:
            return {}

        result = {}
        schema = self.region_schema

        # 新格式：JSON Schema object
        if isinstance(schema, dict) and schema.get("type") == "object" and "properties" in schema:
            def _walk_json_schema(properties, prefix=""):
                for name, prop_schema in properties.items():
                    path = f"{prefix}/{name}" if prefix else name
                    if not isinstance(prop_schema, dict):
                        continue
                    sub_props = prop_schema.get("properties", {})
                    if not sub_props:
                        continue
                    has_sub_region = any(
                        isinstance(v, dict) and v.get("type") == "object" and "properties" in v
                        for v in sub_props.values()
                    )
                    if has_sub_region:
                        _walk_json_schema(sub_props, path)
                    else:
                        result[path] = sub_props
            _walk_json_schema(schema.get("properties", {}))
            return result

        # 旧格式兼容：list of region dicts
        if isinstance(schema, dict) and "regions" in schema:
            schema = schema["regions"]

        if isinstance(schema, list):
            def _walk_legacy(regions, prefix=""):
                for region in regions:
                    if not isinstance(region, dict):
                        continue
                    name = region.get("name") or region.get("region_name", "")
                    path = f"{prefix}/{name}" if prefix else name
                    children = region.get("children")
                    kv = region.get("kv_schema")
                    if kv and not children:
                        result[path] = kv
                    if children:
                        _walk_legacy(children, path)
            _walk_legacy(schema)

        return result

    def _generate_code_from_label(self, label):
        """Generate code from label using utils function"""
        from src.taxonomy_adpt.taxonomy_construct.utils import generate_code_from_name
        return generate_code_from_name(label)

    def add_child(self, label, child_node):
        """
        Add a child node to the current node.

        Args:
        label (str): The label for the child node.
        child_node (Node): The child Node to be added.
        """
        if child_node in self.parents:
            print("CANNOT ADD! THIS WOULD ADD A CYCLE!")
        else:
            child_node.add_parent(self)
            child_node.level = min(parent.level for parent in child_node.parents) + 1
            self.children[label] = child_node

    def add_parent(self, parent_node):
        """
        Add a parent node to the current node.

        Args:
        parent_node (Node): The parent Node to be added.
        """
        if parent_node not in self.parents:
            self.parents.append(parent_node)
            self.level = min(parent.level for parent in self.parents) + 1

    def get_parents(self):
        """
        Get the parent nodes of the current node.

        Returns:
        list: A list of parent nodes.
        """
        return self.parents
    
    def get_ancestors(self):
        """
        Get all ancestor nodes of the current node.

        Returns:
        list: A list of ancestor nodes from the root to the current node.
        """
        ancestors = []
        nodes_to_visit = list(self.parents)
        while nodes_to_visit:
            current = nodes_to_visit.pop()
            if current not in ancestors:
                ancestors.append(current)
                nodes_to_visit.extend(current.parents)
        return ancestors
    
    def get_siblings(self):
        """
        Get the siblings of the current node (nodes that share at least one parent).

        Returns:
        set: A set of sibling nodes.
        """
        siblings = set()
        for parent in self.parents:
            for sibling in parent.get_children().values():
                if sibling is not self:
                    siblings.add(sibling)
        return siblings

    def get_children(self):
        """
        Get the children nodes of the current node.

        Returns:
        dict: A dictionary of children nodes where keys are labels and values are Node instances.
        """
        return self.children
    
    def get_phrases(self):
        """
        Get all phrases of the current node and its descendant nodes.

        Returns:
        list: A list of unique phrases from the current node and all of its descendants.
        """
        unique_phrases = set(self.phrases)
        nodes_to_visit = list(self.children.values())
        
        while nodes_to_visit:
            current_node = nodes_to_visit.pop()
            unique_phrases.update(current_node.phrases)
            nodes_to_visit.extend(current_node.children.values())
        
        return list(unique_phrases)

    def get_sentences(self):
        """
        Get all sentences of the current node and its descendant nodes.

        Returns:
        list: A list of unique sentences from the current node and all of its descendants.
        """
        unique_sentences = set(self.sentences)
        nodes_to_visit = list(self.children.values())
        
        while nodes_to_visit:
            current_node = nodes_to_visit.pop()
            unique_sentences.update(current_node.sentences)
            nodes_to_visit.extend(current_node.children.values())
        
        return list(unique_sentences)
    
    def classify_node(self, args, label2node, visited):

        for child_label, child in self.get_children().items():
            if child.id not in visited:
                child.papers = {}

        # Which papers are classified to the current node?
        prompts = []
        for paper_id, paper in self.papers.items():
            prompts.append(classify_prompt(self, paper))

        output = promptLLM(args, prompts, schema=ClassifySchema, max_new_tokens=3000, timeout_per_request=getattr(args, 'timeout_per_request', 120.0))
        output_dict = safe_json_loads_list(output, log_error=True)
        class_options = [c for c in self.get_children()]
        class_map = {c:0 for c in self.get_children()}
        class_map['unlabeled'] = 0

        for (paper_id, paper), out_labels in zip(self.papers.items(), output_dict):
            # 获取单个类别标签（单标签分类）
            label = out_labels.get('class_label', None)
            if label is None or label == -1 or label == "None":
                class_map['unlabeled'] += 1
                continue
            
            full_label = label + f'_{self.dimension}'
            if "None" in str(label):
                class_map['unlabeled'] += 1
                continue
            elif (full_label in label2node) and (label in class_options):
                label2node[full_label].papers[paper_id] = paper
                class_map[label] += 1
                paper.labels[self.dimension] = [label]  # 单标签分类：直接设置
            else:
                class_map['unlabeled'] += 1
        
        print(f'classification: {str(class_map)}')
        return output_dict
    
    def display(self, level=0, indent_multiplier=2, visited=None):
        """
        Display the node and its children in a structured manner, handling nodes with multiple parents.

        Args:
        level (int): The current level of the node for indentation purposes.
        indent_multiplier (int): The number of spaces used for indentation, multiplied by the level.
        visited (set): A set of visited node IDs to handle cycles in the directed acyclic graph.
        """
        indent = " " * (level * indent_multiplier)
        
        if visited is None:
            visited = set()
        if self.id in visited:
            print(f"{indent}Label (Visited): {self.label}")
            return
        
        output_dict = {"label": self.label,
                       "code": self.code,
                       "description": self.description,
                       "level":self.level,
                       "source":"initial" if self.source is None else self.source
                       }
        
        visited.add(self.id)

        print(f"{indent}Label: {self.label}")
        print(f"{indent}Code: {self.code}")
        print(f"{indent}Dimension: {self.dimension}")
        print(f"{indent}Description: {self.description}")
        print(f"{indent}Level: {self.level}")
        print(f"{indent}Source: {'Initial' if self.source is None else self.source}")

        if len(self.papers) > 0:
            example_papers = [(p.id, unidecode(p.title)) for p in self.papers.values()]
            output_dict['example_papers'] = example_papers[:10]
            output_dict['paper_ids'] = list(self.papers.keys())

            print(f"{indent}# of Papers: {len(self.papers)}")
            print(f"{indent}Example Papers: {str(example_papers[:3])}")
        if self.children:
            print(f"{indent}{'-'*40}")
            print(f"{indent}Children:")
            output_dict['children'] = []

            for child in self.children.values():
                sub_dict = child.display(level + 1, indent_multiplier, visited)
                if sub_dict is not None:
                    output_dict['children'].append(sub_dict)
            
        print(f"{indent}{'-'*40}")
        return output_dict

    def __repr__(self):
        return f"Node(label={self.label}, dim={self.dimension}, description={self.description}, level={self.level})"
    

class DAG:
    def __init__(self, root, dim):
        """
        Initialize a DAG with a root node.

        Args:
        root (Node): The root node of the DAG.
        """
        self.root = root
        self.dimension = dim

    def enrich_dag(self, args, id2node, use_element_schema=True, use_iterative=True, schema_log_callback=None):
        """
        Iterate through the DAG starting from the root node and call enrich_node on each node.
        
        Args:
            args: 参数对象
            id2node: id到节点的映射
            use_element_schema: 是否使用企业文档的要素Schema模式（新）还是传统的phrases/sentences模式（旧）
            use_iterative: 是否使用迭代式Schema生成（先生成种子，再基于文档修正）
        """
        from src.taxonomy_adpt.taxonomy_construct.prompts import ElementSchemaEnrichment
        from src.taxonomy_adpt.taxonomy_construct.enrichment import generate_schema_iteratively
        
        visited = set()
        nodes_to_visit = [(self.root, [])]
        node_info = {}  # 存储每个节点的信息（ancestors和sample_docs）

        # 第一遍遍历：收集所有节点和它们的示例文档
        while nodes_to_visit:
            current_node, ancestors = nodes_to_visit.pop()
            if current_node.id in visited:
                continue
            visited.add(current_node.id)
            
            # 为该节点收集示例文档（根据有无文档决定数量）
            sample_docs = []
            if current_node.papers and len(current_node.papers) > 0:
                # 如果使用迭代模式，收集更多文档（最多9份，用于3轮修正）
                max_samples = 9 if use_iterative else 3
                sample_docs = list(current_node.papers.values())[:max_samples]
            
            node_info[current_node.id] = {
                'node': current_node,
                'ancestors': ancestors,
                'sample_docs': sample_docs
            }
            
            # Add children to visit next with updated ancestors
            new_ancestors = ancestors + [current_node]
            for child in current_node.get_children().values():
                nodes_to_visit.append((child, new_ancestors))

        # 根据模式选择不同的Schema和处理逻辑
        if use_element_schema:
            schemas_generated = 0
            potential_merges = []  # 记录可能需要合并的节点对
            
            # 选择生成方式
            if use_iterative:
                # 迭代式生成：种子Schema + 文档修正
                print(f"  使用迭代式Schema生成（种子+修正）...")
                
                for node_id, info in node_info.items():
                    current_node = info['node']
                    ancestors = info['ancestors']
                    sample_docs = info['sample_docs']
                    
                    print(f"\n  处理节点: {current_node.label} ({len(sample_docs)} 份示例文档)")
                    
                    try:
                        # 迭代式生成Schema
                        result = generate_schema_iteratively(
                            args, 
                            current_node, 
                            ancestors, 
                            sample_docs,
                            max_rounds=getattr(args, 'schema_refinement_rounds', 3)
                        )
                        
                        if result:
                            # 存储要素Schema
                            current_node.element_schema = result.get('element_schema', {})
                            current_node.schema_reasoning = result.get('schema_reasoning', '')
                            current_node.schema_complexity = result.get('schema_complexity', 'medium')
                            current_node.should_distinct = result.get('should_distinct', True)
                            
                            # 记录迭代信息
                            if 'refinement_rounds' in result:
                                current_node.refinement_rounds = result['refinement_rounds']
                                current_node.total_changes = result['total_changes']
                            
                            schemas_generated += 1
                            
                            # 调用日志回调
                            if schema_log_callback:
                                schema_log_callback(current_node)
                            
                            # 如果LLM认为不应该与兄弟节点区分，记录下来
                            if not current_node.should_distinct:
                                siblings = current_node.get_siblings()
                                if siblings:
                                    potential_merges.append({
                                        'node': current_node.label,
                                        'siblings': [s.label for s in siblings],
                                        'reasoning': current_node.schema_reasoning
                                    })
                            
                            print(f"    ✓ Schema生成完成: 复杂度={current_node.schema_complexity}, "
                                  f"应区分={current_node.should_distinct}")
                        else:
                            print(f"    ✗ Schema生成失败")
                            
                    except Exception as e:
                        print(f"    ✗ 处理节点失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            else:
                # 单次生成模式：直接生成完整Schema
                print(f"  使用单次生成Schema模式...")
                
                prompts = {}
                for node_id, info in node_info.items():
                    from src.taxonomy_adpt.taxonomy_construct.enrichment import enrich_node_prompt
                    prompts[node_id] = enrich_node_prompt(
                        args, 
                        info['node'], 
                        info['ancestors'], 
                        info['sample_docs']
                    )
                
                output = promptLLM(
                    args, 
                    list(prompts.values()), 
                    schema=ElementSchemaEnrichment, 
                    max_new_tokens=2000,
                    timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
                    temperature=0.3,
                )
                output_dict = safe_json_loads_list(output, log_error=True)

                for node_id, out in zip(prompts.keys(), output_dict):
                    node = id2node[node_id]
                    
                    if isinstance(out, dict):
                        # 存储要素Schema
                        node.element_schema = out.get('element_schema', {})
                        node.schema_reasoning = out.get('reasoning', '')
                        node.schema_complexity = out.get('schema_complexity', 'medium')
                        node.should_distinct = out.get('should_distinct_from_siblings', True)
                        
                        schemas_generated += 1
                        
                        # 调用日志回调
                        if schema_log_callback:
                            schema_log_callback(node)
                        
                        # 如果LLM认为不应该与兄弟节点区分，记录下来
                        if not node.should_distinct:
                            siblings = node.get_siblings()
                            if siblings:
                                potential_merges.append({
                                    'node': node.label,
                                    'siblings': [s.label for s in siblings],
                                    'reasoning': node.schema_reasoning
                                })
                        
                        print(f"    ✓ {node.label}: Schema复杂度={node.schema_complexity}, "
                              f"应区分={node.should_distinct}")
            
            # 报告可能需要合并的节点
            if potential_merges:
                print(f"\n  ⚠️ 发现 {len(potential_merges)} 个可能需要合并的节点：")
                for merge_info in potential_merges:
                    print(f"    - '{merge_info['node']}' 与 {merge_info['siblings']} 的Schema本质相同")
                    print(f"      原因: {merge_info['reasoning']}")
            
            return schemas_generated, potential_merges
        
        else:
            # 旧模式：生成phrases和sentences（保持向后兼容）
            print(f"  使用传统phrases/sentences模式进行富化...")
            output = promptLLM(
                args, 
                list(prompts.values()), 
                schema=EnrichSchema, 
                max_new_tokens=1500, 
                timeout_per_request=getattr(args, 'timeout_per_request', 120.0)
            )
            output_dict = safe_json_loads_list(output, log_error=True)

            all_phrases = []
            all_sentences = []
            
            for node_id, out in zip(prompts.keys(), output_dict):
                node = id2node[node_id]

                node.phrases = [p.lower().replace(' ', '_') for p in out['commonsense_key_phrases']]
                all_phrases.extend(node.phrases)

                node.sentences = [p.lower() for p in out['commonsense_sentences']]
                all_sentences.extend(node.sentences)
            
            return all_phrases, all_sentences
    
    def classify_dag(self, args, label2node, start_node=None):
        visited = set()
        # self.root.papers = collection
        if start_node is None:
            nodes_to_visit = [(self.root, self.root.papers)]
        else:
            nodes_to_visit = [(start_node, start_node.papers)]

        while nodes_to_visit:
            current_node, papers = nodes_to_visit.pop()
            if (current_node.id in visited) or len(current_node.get_children()) == 0:
                continue
            for child_label, child in current_node.get_children().items():
                if child.id not in visited:
                    child.papers = {}

            print(f'visiting: {current_node.label}; # of papers: {len(papers)}')

            visited.add(current_node.id)

            # Which papers are classified to the current node?
            prompts = []
            for paper_id, paper in papers.items():
                prompts.append(classify_prompt(current_node, paper))

            output = promptLLM(args, prompts, schema=ClassifySchema, max_new_tokens=1500, timeout_per_request=getattr(args, 'timeout_per_request', 120.0))
            output_dict = safe_json_loads_list(output, log_error=True)
            class_options = [c for c in current_node.get_children()]

            for (paper_id, paper), out_labels in zip(papers.items(), output_dict):
                # 获取单个类别标签（单标签分类）
                label = out_labels.get('class_label', None)
                if label is None or label == -1 or label == "None":
                    continue
                if "None" in str(label):
                    continue
                elif (label in label2node) and (label in class_options):
                    label2node[label].papers[paper_id] = paper
                    paper.labels[current_node.dimension] = [label]  # 单标签分类：直接设置
            
            # Add children to visit next with updated ancestors
            for child in current_node.get_children().values():
                nodes_to_visit.append((child, child.papers))
        
        return output_dict
