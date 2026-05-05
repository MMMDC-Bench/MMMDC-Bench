"""
企业文档树的扩展逻辑(深度扩展和宽度扩展)
"""

import json
import random
from collections import Counter

from src.taxonomy_adpt.taxonomy_construct.taxonomy import Node
from src.taxonomy_adpt.taxonomy_construct.utils import clean_json_string, safe_json_loads
# 使用新的 LLM 适配器
from src.taxonomy_adpt.llm_client.llm_adapter import constructPrompt, promptLLM
from src.taxonomy_adpt.taxonomy_construct.prompts import (
    width_system_instruction, width_main_prompt, WidthExpansionSchema,
    width_cluster_system_instruction, width_cluster_main_prompt, WidthClusterListSchema,
    depth_system_instruction, depth_main_prompt, DepthExpansionSchema,
    depth_cluster_system_instruction, depth_cluster_main_prompt, DepthClusterListSchema,
    width_group_system_instruction, width_group_main_prompt, GroupPseudoLabelSchema,
    depth_group_system_instruction, depth_group_main_prompt,
    depth_expansion_evaluation_system_instruction, depth_expansion_evaluation_prompt, DepthExpansionEvaluationSchema
)


def safe_parse_subtopic_label(output_str):
    """
    安全地从LLM输出中解析子主题标签
    
    Args:
        output_str: LLM的输出字符串
        
    Returns:
        解析后的标签字符串（小写，下划线分隔），如果解析失败则返回None
    """
    parsed = safe_json_loads(output_str, default=None, log_error=False)
    
    if parsed is None:
        return None
    
    # 提取标签
    if 'new_subtopic_label' not in parsed:
        print(f"  警告: JSON中缺少 'new_subtopic_label' 字段，跳过此结果")
        return None
    
    label = parsed['new_subtopic_label']
    if not label or not isinstance(label, str):
        print(f"  警告: 标签为空或不是字符串，跳过此结果")
        return None
    
    # 标准化标签格式
    normalized_label = label.replace(' ', '_').lower()
    return normalized_label


def _expandNodeWidthTraditional(args, node, id2node, label2node, unlabeled_docs, ancestors):
    """
    传统的宽度扩展方法: 为所有文档独立生成伪标签，然后聚类
    """
    # 步骤1: 为每个未分类文档生成候选主题
    print(f'  为 {len(unlabeled_docs)} 份文档生成候选主题...')
    exp_prompts = [
        constructPrompt(args, width_system_instruction, width_main_prompt(doc, node, ancestors))
        for doc in unlabeled_docs.values()
    ]
    
    exp_outputs = promptLLM(
        args=args,
        prompts=exp_prompts,
        schema=WidthExpansionSchema,
        max_new_tokens=300,
        json_mode=True,
        timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
        temperature=0.6,
        top_p=0.99
    )
    
    parsed_outputs = []
    _parse_fail_count = 0
    for c in exp_outputs:
        parsed = safe_parse_subtopic_label(c)
        if parsed is not None:
            parsed_outputs.append(parsed)
        else:
            _parse_fail_count += 1
    if _parse_fail_count > 0:
        print(f'  警告: {_parse_fail_count}/{len(exp_outputs)} 个LLM响应解析失败（空响应或格式错误）')
    
    exp_outputs = parsed_outputs
    
    # 过滤掉已存在的主题
    exp_outputs = [w for w in exp_outputs if w + f"_{node.dimension}" not in label2node]
    
    if len(exp_outputs) == 0:
        print(f'  没有发现新的候选主题')
        return []
    
    freq_options = dict(Counter(exp_outputs))
    print(f'  候选主题频率分布: {freq_options}')
    
    # 步骤2: 使用LLM对候选主题进行聚类
    return _clusterAndCreateNodes(args, node, id2node, label2node, freq_options, ancestors, 'width')


def _expandNodeWidthWithClusters(args, node, id2node, label2node, unlabeled_docs, ancestors):
    """
    基于cluster_label的宽度扩展: 先按文档向量聚类分组，再以组为单位生成伪标签
    让模型综合判断整个组应该归属于哪些类别
    """
    # 获取聚类标签列名
    cluster_label_col = getattr(args, 'cluster_label_col', 'cluster_label')
    
    # 步骤1: 按cluster_label分组
    cluster_groups = {}
    docs_without_cluster = []
    
    for idx, doc in unlabeled_docs.items():
        cluster_label = doc.metadata.get(cluster_label_col)
        if cluster_label is not None:
            if cluster_label not in cluster_groups:
                cluster_groups[cluster_label] = []
            cluster_groups[cluster_label].append((idx, doc))
        else:
            docs_without_cluster.append((idx, doc))
    
    print(f'  文档分组: {len(cluster_groups)} 个cluster组, {len(docs_without_cluster)} 个无{cluster_label_col}的文档')
    
    # 步骤2: 对每个cluster组生成伪标签（组级别）
    all_pseudo_labels = []
    
    for cluster_id, docs in cluster_groups.items():
        print(f'  处理cluster组 {cluster_id}: {len(docs)} 份文档')
        
        # 采样文档（如果组太大）
        sample_size = min(len(docs), 10)  # 每组最多采样10份文档
        sampled_docs = random.sample(docs, sample_size) if len(docs) > sample_size else docs
        
        # 提取文档对象（去掉索引）
        doc_objects = [doc for idx, doc in sampled_docs]
        
        # 为整个组生成伪标签（将所有文档作为上下文）
        # width_group_main_prompt 现在返回 (prompt_text, image_base64_list)
        prompt_text, image_base64_list = width_group_main_prompt(doc_objects, node, ancestors)
        group_prompt = constructPrompt(
            args, 
            width_group_system_instruction, 
            prompt_text
        )
        
        # 如果有图片，将图片列表组合成一个字符串（用特殊分隔符）
        # 注意：这里需要根据实际的 LLM API 来调整
        # 对于支持多模态的 API（如 GPT-4V），可以传递图片列表
        images_for_llm = None
        if image_base64_list:
            # 暂时使用第一张图片作为代表（可以后续改进为传递所有图片）
            images_for_llm = [image_base64_list[0] if image_base64_list else None]
            print(f'    cluster组 {cluster_id} 包含 {len(image_base64_list)} 张图片')
        
        try:
            group_outputs = promptLLM(
                args=args,
                prompts=[group_prompt],
                schema=GroupPseudoLabelSchema,
                max_new_tokens=500,
                json_mode=True,
                timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
                temperature=0.6,
                top_p=0.99,
                images_base64=images_for_llm  # 传递图片
            )
            
            # 解析输出
            if len(group_outputs) > 0:
                result = group_outputs[0]
                if isinstance(result, str):
                    result = safe_json_loads(result, default={}, log_error=True)
                
                if isinstance(result, dict):
                    labels = result.get('subtopic_labels', [])
                    reasoning = result.get('reasoning', '')
                    
                    # 标准化标签格式并过滤已存在的
                    valid_labels = []
                    for label in labels:
                        if isinstance(label, str) and label:
                            normalized_label = label.replace(' ', '_').lower()
                            if normalized_label + f"_{node.dimension}" not in label2node:
                                valid_labels.append(normalized_label)
                    
                    if valid_labels:
                        print(f'    cluster组 {cluster_id} 生成标签: {valid_labels}')
                        if reasoning:
                            print(f'    原因: {reasoning}')
                        all_pseudo_labels.extend(valid_labels)
                    else:
                        print(f'    cluster组 {cluster_id} 未生成新标签（已存在或无效）')
                        
        except Exception as e:
            print(f'    cluster组 {cluster_id} 处理失败: {str(e)}')
            continue
    
    # 步骤3: 处理没有cluster_label的文档（使用传统方法）
    if len(docs_without_cluster) > 0:
        print(f'  为 {len(docs_without_cluster)} 份无{cluster_label_col}的文档生成候选主题...')
        exp_prompts = [
            constructPrompt(args, width_system_instruction, width_main_prompt(doc, node, ancestors))
            for idx, doc in docs_without_cluster
        ]
        
        exp_outputs = promptLLM(
            args=args,
            prompts=exp_prompts,
            schema=WidthExpansionSchema,
            max_new_tokens=300,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.6,
            top_p=0.99
        )
        
        parsed_outputs = []
        _parse_fail_count = 0
        for c in exp_outputs:
            parsed = safe_parse_subtopic_label(c)
            if parsed is not None:
                parsed_outputs.append(parsed)
            else:
                _parse_fail_count += 1
        if _parse_fail_count > 0:
            print(f'  警告: {_parse_fail_count}/{len(exp_outputs)} 个LLM响应解析失败（空响应或格式错误）')
        
        parsed_outputs = [w for w in parsed_outputs if w + f"_{node.dimension}" not in label2node]
        all_pseudo_labels.extend(parsed_outputs)
    
    if len(all_pseudo_labels) == 0:
        print(f'  没有发现新的候选主题')
        return []
    
    # 步骤4: 统计所有伪标签并进行最终聚类
    freq_options = dict(Counter(all_pseudo_labels))
    print(f'  总体候选主题频率分布: {freq_options}')
    
    # 步骤5: 使用LLM对候选主题进行聚类
    return _clusterAndCreateNodes(args, node, id2node, label2node, freq_options, ancestors, 'width')


def _clusterAndCreateNodes(args, node, id2node, label2node, freq_options, ancestors, source):
    """
    共用的聚类和节点创建逻辑
    
    Args:
        args: 参数
        node: 当前节点
        id2node: id到节点的映射
        label2node: label到节点的映射
        freq_options: 候选主题频率字典
        ancestors: 祖先路径字符串
        source: 节点来源 ('width' 或 'depth')
    """
    all_node_labels = ", ".join(list(label2node.keys()))
    
    args.llm = 'gpt'  # 聚类使用更强的模型
    
    if source == 'width':
        clustered_prompt = [
            constructPrompt(args, width_cluster_system_instruction,
                           width_cluster_main_prompt(freq_options, node, ancestors, all_node_labels))
        ]
        schema = WidthClusterListSchema
    else:  # depth
        clustered_prompt = [
            constructPrompt(args, depth_cluster_system_instruction,
                           depth_cluster_main_prompt(freq_options, node, ancestors, all_node_labels))
        ]
        schema = DepthClusterListSchema
    
    success = False
    attempts = 0
    cluster_outputs = None
    
    while (not success) and (attempts < 5):
        try:
            cluster_topics = promptLLM(
                args=args,
                prompts=clustered_prompt,
                schema=schema,
                max_new_tokens=3000,
                json_mode=True,
                timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
                temperature=0.6,
                top_p=0.99
            )[0]
            
            if isinstance(cluster_topics, str):
                cluster_outputs = safe_json_loads(cluster_topics, default={}, log_error=True)
            else:
                from src.taxonomy_adpt.taxonomy_construct.utils import safe_json_loads_list
                cluster_outputs = safe_json_loads_list(cluster_topics, log_error=True)
                # 如果列表解析失败，返回空字典
                if not cluster_outputs:
                    cluster_outputs = {}
            
            success = True
            
        except Exception as e:
            success = False
            attempts += 1
            print(f'  {source}扩展聚类失败(尝试 {attempts}/5): {str(e)}')
    
    args.llm = 'vllm'  # 恢复使用vllm
    
    if not success:
        print(f'  警告: {source}扩展失败')
        return []
    
    # 根据聚类结果创建新节点
    cluster_outputs = cluster_outputs.get('new_cluster_topics', [])
    if cluster_outputs:
        print(f'  [{source}扩展] LLM 聚类结果: {len(cluster_outputs)} 个类别')
        for ct in cluster_outputs:
            core_features = ct.get('core_features', '')
            feature_str = f" | 核心特征: {core_features}" if core_features else ""
            print(f'    · {ct.get("label", "?")} — {ct.get("description", "")}{feature_str}')

    final_expansion = []
    dim = node.dimension
    
    # 收集已存在的code，确保唯一性
    existing_codes = {n.code for n in id2node.values()}
    
    # 找到当前最大的id，确保新节点的id不会重复
    max_id = max(id2node.keys()) if id2node else -1
    
    for subtopic_cluster in cluster_outputs:
        child_label = subtopic_cluster['label']
        child_desc = subtopic_cluster['description']
        
        # 获取LLM生成的code，如果没有则自动生成
        child_code = subtopic_cluster.get('code', None)
        if not child_code:
            from src.taxonomy_adpt.taxonomy_construct.utils import generate_code_from_name
            child_code = generate_code_from_name(child_label, existing_codes)
        else:
            # 确保code符合规范
            child_code = child_code.lower().replace(' ', '_').replace('-', '_')
            if child_code in existing_codes:
                from src.taxonomy_adpt.taxonomy_construct.utils import generate_code_from_name
                child_code = generate_code_from_name(child_label, existing_codes)
        
        existing_codes.add(child_code)
        mod_full_key = child_code + f"_{dim}"
        
        if mod_full_key not in label2node:
            max_id += 1  # 递增id
            child_node = Node(
                id=max_id,
                label=child_label,
                code=child_code,
                dimension=dim,
                description=child_desc,
                parents=[node],
                source=source
            )
            node.add_child(child_code, child_node)
            id2node[child_node.id] = child_node
            label2node[mod_full_key] = child_node
            final_expansion.append(child_label)
            
        elif node.code + f"_{dim}" in label2node and label2node[mod_full_key] in label2node[node.code + f"_{dim}"].get_ancestors():
            # 避免循环依赖
            print(f"  - 跳过: {child_label} 会形成循环")
            continue
        else:
            # 节点已存在,添加父子关系
            child_node = label2node[mod_full_key]
            node.add_child(child_code, child_node)
            child_node.add_parent(node)
            final_expansion.append(child_label)
    
    return final_expansion


def expandNodeWidth(args, node, id2node, label2node):
    """
    宽度扩展: 为父节点发现新的兄弟类别
    当父节点的子节点无法覆盖所有文档时触发
    
    支持两种模式:
    1. 基于cluster_label的分组扩展 (推荐): 先按文档向量聚类结果分组，再在组内定义伪标签
    2. 传统的全局扩展: 为所有文档独立生成伪标签
    """
    # 找出未被子节点覆盖的文档
    unlabeled_docs = {}
    for idx, doc in node.papers.items():
        unlabeled = True
        for c in node.children.values():
            if idx in c.papers:
                unlabeled = False
                break
        if unlabeled:
            unlabeled_docs[idx] = doc
    
    # 构建祖先路径
    node_ancestors = node.get_ancestors()
    if node_ancestors is None:
        ancestors = "无"
    else:
        node_ancestors.reverse()
        ancestors = " -> ".join([ancestor.label for ancestor in node_ancestors])
    
    existing_children = [c.label for c in node.children.values()]
    total_classified = len(node.papers) - len(unlabeled_docs)
    print(f'  [宽度扩展] 节点 "{node.label}" — 总文档={len(node.papers)}, 已分类={total_classified}, 未分类={len(unlabeled_docs)}')
    if existing_children:
        print(f'  [宽度扩展] 现有子类: {existing_children}')
    
    # 如果未分类文档数量不足阈值,不进行扩展
    if len(unlabeled_docs) <= args.max_density:
        print(f'  [宽度扩展] ✗ 未分类文档数不足阈值 ({len(unlabeled_docs)} <= {args.max_density})，不扩展')
        return []
    
    # 检查是否启用基于cluster_label的扩展
    use_cluster_based = getattr(args, 'use_cluster_based_expansion', True)
    if use_cluster_based:
        # 获取聚类标签列名
        cluster_label_col = getattr(args, 'cluster_label_col', 'cluster_label')
        
        # 检查文档是否有cluster_label
        has_cluster_label = any(
            doc.metadata.get(cluster_label_col) is not None 
            for doc in unlabeled_docs.values()
        )
        
        if has_cluster_label:
            print(f'  使用基于{cluster_label_col}的分组扩展')
            return _expandNodeWidthWithClusters(args, node, id2node, label2node, unlabeled_docs, ancestors)
    
    # 使用传统方法
    print(f'  使用传统的全局扩展方法')
    return _expandNodeWidthTraditional(args, node, id2node, label2node, unlabeled_docs, ancestors)


def _expandNodeDepthTraditional(args, node, id2node, label2node, ancestors):
    """
    传统的深度扩展方法: 为所有文档独立生成伪标签，然后聚类
    """
    # 步骤1: 为每个文档生成候选子主题
    print(f'  为 {len(node.papers)} 份文档生成候选子主题...')
    args.llm = 'vllm'
    
    subtopic_prompts = [
        constructPrompt(args, depth_system_instruction, depth_main_prompt(doc, node, ancestors))
        for doc in node.papers.values()
    ]
    
    subtopic_outputs = promptLLM(
        args=args,
        prompts=subtopic_prompts,
        schema=DepthExpansionSchema,
        max_new_tokens=300,
        json_mode=True,
        timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
        temperature=0.6,
        top_p=0.99
    )
    
    parsed_outputs = []
    _parse_fail_count = 0
    for c in subtopic_outputs:
        parsed = safe_parse_subtopic_label(c)
        if parsed is not None:
            parsed_outputs.append(parsed)
        else:
            _parse_fail_count += 1
    if _parse_fail_count > 0:
        print(f'  警告: {_parse_fail_count}/{len(subtopic_outputs)} 个LLM响应解析失败（空响应或格式错误）')
    
    subtopic_outputs = parsed_outputs
    
    # 过滤掉已存在的主题
    subtopic_outputs = [w for w in subtopic_outputs if w + f"_{node.dimension}" not in label2node]
    
    if len(subtopic_outputs) == 0:
        print(f'  没有发现新的候选子主题')
        return [], False
    
    freq_options = dict(Counter(subtopic_outputs))
    print(f'  候选子主题频率分布: {freq_options}')
    
    # 步骤2: 使用LLM对候选子主题进行聚类
    final_expansion = _clusterAndCreateNodes(args, node, id2node, label2node, freq_options, ancestors, 'depth')
    return final_expansion, len(final_expansion) > 0


def _expandNodeDepthWithClusters(args, node, id2node, label2node, ancestors):
    """
    基于cluster_label的深度扩展: 先按文档向量聚类分组，再以组为单位生成伪标签
    让模型综合判断整个组应该归属于哪些子类别
    """
    # 获取聚类标签列名
    cluster_label_col = getattr(args, 'cluster_label_col', 'cluster_label')
    
    # 步骤1: 按cluster_label分组
    cluster_groups = {}
    docs_without_cluster = []
    
    for idx, doc in node.papers.items():
        cluster_label = doc.metadata.get(cluster_label_col)
        if cluster_label is not None:
            if cluster_label not in cluster_groups:
                cluster_groups[cluster_label] = []
            cluster_groups[cluster_label].append((idx, doc))
        else:
            docs_without_cluster.append((idx, doc))
    
    print(f'  文档分组: {len(cluster_groups)} 个cluster组, {len(docs_without_cluster)} 个无{cluster_label_col}的文档')
    
    # 步骤2: 对每个cluster组生成伪标签（组级别）
    all_pseudo_labels = []
    
    for cluster_id, docs in cluster_groups.items():
        print(f'  处理cluster组 {cluster_id}: {len(docs)} 份文档')
        
        # 采样文档（如果组太大）
        sample_size = min(len(docs), 10)  # 每组最多采样10份文档
        sampled_docs = random.sample(docs, sample_size) if len(docs) > sample_size else docs
        
        # 提取文档对象（去掉索引）
        doc_objects = [doc for idx, doc in sampled_docs]
        
        # 为整个组生成伪标签（将所有文档作为上下文）
        # depth_group_main_prompt 现在返回 (prompt_text, image_base64_list)
        args.llm = 'vllm'
        prompt_text, image_base64_list = depth_group_main_prompt(doc_objects, node, ancestors)
        group_prompt = constructPrompt(
            args, 
            depth_group_system_instruction, 
            prompt_text
        )
        
        # 如果有图片，将图片列表组合成一个字符串（用特殊分隔符）
        images_for_llm = None
        if image_base64_list:
            # 暂时使用第一张图片作为代表（可以后续改进为传递所有图片）
            images_for_llm = [image_base64_list[0] if image_base64_list else None]
            print(f'    cluster组 {cluster_id} 包含 {len(image_base64_list)} 张图片')
        
        try:
            group_outputs = promptLLM(
                args=args,
                prompts=[group_prompt],
                schema=GroupPseudoLabelSchema,
                max_new_tokens=500,
                json_mode=True,
                timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
                temperature=0.6,
                top_p=0.99,
                images_base64=images_for_llm  # 传递图片
            )
            
            # 解析输出
            if len(group_outputs) > 0:
                result = group_outputs[0]
                if isinstance(result, str):
                    result = safe_json_loads(result, default={}, log_error=True)
                
                if isinstance(result, dict):
                    labels = result.get('subtopic_labels', [])
                    reasoning = result.get('reasoning', '')
                    
                    # 标准化标签格式并过滤已存在的
                    valid_labels = []
                    for label in labels:
                        if isinstance(label, str) and label:
                            normalized_label = label.replace(' ', '_').lower()
                            if normalized_label + f"_{node.dimension}" not in label2node:
                                valid_labels.append(normalized_label)
                    
                    if valid_labels:
                        print(f'    cluster组 {cluster_id} 生成标签: {valid_labels}')
                        if reasoning:
                            print(f'    原因: {reasoning}')
                        all_pseudo_labels.extend(valid_labels)
                    else:
                        print(f'    cluster组 {cluster_id} 未生成新标签（已存在或无效）')
                        
        except Exception as e:
            print(f'    cluster组 {cluster_id} 处理失败: {str(e)}')
            continue
    
    # 步骤3: 处理没有cluster_label的文档（使用传统方法）
    if len(docs_without_cluster) > 0:
        print(f'  为 {len(docs_without_cluster)} 份无{cluster_label_col}的文档生成候选子主题...')
        args.llm = 'vllm'
        subtopic_prompts = [
            constructPrompt(args, depth_system_instruction, depth_main_prompt(doc, node, ancestors))
            for idx, doc in docs_without_cluster
        ]
        
        subtopic_outputs = promptLLM(
            args=args,
            prompts=subtopic_prompts,
            schema=DepthExpansionSchema,
            max_new_tokens=300,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.6,
            top_p=0.99
        )
        
        parsed_outputs = []
        _parse_fail_count = 0
        for c in subtopic_outputs:
            parsed = safe_parse_subtopic_label(c)
            if parsed is not None:
                parsed_outputs.append(parsed)
            else:
                _parse_fail_count += 1
        if _parse_fail_count > 0:
            print(f'  警告: {_parse_fail_count}/{len(subtopic_outputs)} 个LLM响应解析失败（空响应或格式错误）')
        
        parsed_outputs = [w for w in parsed_outputs if w + f"_{node.dimension}" not in label2node]
        all_pseudo_labels.extend(parsed_outputs)
    
    if len(all_pseudo_labels) == 0:
        print(f'  没有发现新的候选子主题')
        return [], False
    
    # 步骤4: 统计所有伪标签并进行最终聚类
    freq_options = dict(Counter(all_pseudo_labels))
    print(f'  总体候选子主题频率分布: {freq_options}')
    
    # 步骤5: 使用LLM对候选子主题进行聚类
    final_expansion = _clusterAndCreateNodes(args, node, id2node, label2node, freq_options, ancestors, 'depth')
    return final_expansion, len(final_expansion) > 0


def _evaluateDepthExpansionWithRegion(args, node, ancestors):
    """
    基于 Region Schema 差异评估是否应该进行深度扩展。

    流程：
    1. 采样文档，生成典型 Region Schema
    2. 按 cluster 分组，对每组检测与典型 schema 的偏离
    3. 汇总偏离结果，调用 LLM 做最终扩展决策

    Returns:
        tuple: (should_expand, evaluation_result)
    """
    from src.taxonomy_adpt.taxonomy_construct.enrichment import (
        generate_region_schema_for_node,
        check_region_deviation,
    )
    from src.taxonomy_adpt.taxonomy_construct.prompts import (
        region_expansion_decision_system_instruction,
        region_expansion_decision_prompt,
        RegionExpansionDecision,
    )

    cluster_label_col = getattr(args, 'cluster_label_col', 'cluster_label')

    # 密度门槛：文档数低于阈值则不扩展（基础条件仍然保留）
    if len(node.papers) <= args.max_density:
        return False, {
            'should_expand': False,
            'reasoning': f'文档数未超过阈值 ({len(node.papers)} <= {args.max_density})',
        }

    # --- Step 1: 采样并生成典型 Region Schema ---
    print(f'  [Region] 为节点 "{node.label}" 生成典型 Region Schema...')

    # 按 cluster 分组
    docs_by_cluster = {}
    for doc in node.papers.values():
        cid = doc.metadata.get(cluster_label_col, '__no_cluster__')
        docs_by_cluster.setdefault(cid, []).append(doc)

    # 从各 cluster 采样代表性文档（用于生成典型 schema）
    sample_docs = []
    for cid, docs in docs_by_cluster.items():
        sample_count = min(2, len(docs))
        sample_docs.extend(random.sample(docs, sample_count))
    if len(sample_docs) > 10:
        sample_docs = random.sample(sample_docs, 10)

    node_ancestors = node.get_ancestors() or []
    node_ancestors.reverse()

    canonical_schema, canonical_kv_schema, reasoning = generate_region_schema_for_node(
        args, node, node_ancestors, sample_docs
    )

    if canonical_schema is None:
        # 回退到密度判断
        should_expand = len(node.papers) > args.max_density
        return should_expand, {
            'should_expand': should_expand,
            'reasoning': 'Region Schema 生成失败，回退到密度阈值判断',
        }

    # 保存典型 schema 到节点
    node.region_schema = canonical_schema
    node.region_schema_reasoning = reasoning

    # --- Step 2: 逐 cluster 检测偏离 ---
    print(f'  [Region] 检测 {len(docs_by_cluster)} 个 cluster 的偏离...')
    deviation_summaries = []

    saved_llm = args.llm
    args.llm = 'gpt'  # 偏离检测使用强模型

    for cid, docs in docs_by_cluster.items():
        if cid == '__no_cluster__':
            continue
        # 采样
        check_docs = docs if len(docs) <= 8 else random.sample(docs, 8)

        print(f'    cluster {cid}: {len(docs)} 份文档')
        result = check_region_deviation(args, node, canonical_schema, check_docs)
        result['cluster_id'] = str(cid)
        result['doc_count'] = len(docs)
        deviation_summaries.append(result)

        status = "符合" if result['fits'] else "偏离"
        print(f'      → {status}')
        if result.get('structural_deviations'):
            for dev in result['structural_deviations'][:3]:
                print(f'        - {dev}')

    # --- Step 3: 汇总决策 ---
    deviates_count = sum(1 for d in deviation_summaries if not d.get('fits', True))

    if deviates_count == 0:
        args.llm = saved_llm
        print(f'  [Region] 所有 cluster 均符合典型 schema，不扩展')
        return False, {
            'should_expand': False,
            'reasoning': '所有 cluster 均符合典型 Region Schema，无结构性差异',
            'canonical_region_schema': canonical_schema,
        }

    print(f'  [Region] {deviates_count}/{len(deviation_summaries)} 个 cluster 存在偏离，请求最终决策...')

    ancestors_str = " -> ".join([a.label for a in node_ancestors]) if node_ancestors else "无"
    decision_prompt_text = region_expansion_decision_prompt(
        node, canonical_schema, deviation_summaries, ancestors_str
    )
    decision_prompt = constructPrompt(
        args, region_expansion_decision_system_instruction, decision_prompt_text
    )

    try:
        decision_output = promptLLM(
            args=args,
            prompts=[decision_prompt],
            schema=RegionExpansionDecision,
            max_new_tokens=4000,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.2,
        )[0]

        if isinstance(decision_output, str):
            decision_result = safe_json_loads(decision_output, default={}, log_error=True)
        else:
            decision_result = decision_output

        should_expand = decision_result.get('should_expand', False)
        decision_reasoning = decision_result.get('reasoning', '')
        candidates = decision_result.get('candidates', [])

        print(f'  [Region] 决策: {"扩展" if should_expand else "不扩展"}')
        print(f'  [Region] 理由: {decision_reasoning}')
        if candidates:
            print(f'  [Region] 候选子类别: {[c.get("label", "?") for c in candidates]}')

        args.llm = saved_llm
        return should_expand, {
            'should_expand': should_expand,
            'reasoning': decision_reasoning,
            'canonical_region_schema': canonical_schema,
            'candidates': candidates,
            'deviation_summaries': deviation_summaries,
        }

    except Exception as e:
        args.llm = saved_llm
        print(f'  [Region] 决策调用失败: {str(e)}，回退到密度判断')
        should_expand = len(node.papers) > args.max_density
        return should_expand, {
            'should_expand': should_expand,
            'reasoning': f'Region 决策失败，回退到密度判断: {str(e)}',
            'canonical_region_schema': canonical_schema,
        }


def evaluateDepthExpansion(args, node, ancestors):
    """
    评估是否应该进行深度扩展。

    支持三种模式（按优先级）：
    1. Region Schema 驱动（use_region_based_expansion）
    2. 可解释增强（use_interpretable_expansion）—— 旧方案，保留作为 fallback
    3. 传统密度阈值
    
    Args:
        args: 参数
        node: 当前节点
        ancestors: 祖先路径字符串
    
    Returns:
        tuple: (should_expand, evaluation_result)
    """
    # 优先使用 Region Schema 驱动
    if getattr(args, 'use_region_based_expansion', False):
        return _evaluateDepthExpansionWithRegion(args, node, ancestors)

    # 检查是否启用可解释增强（旧方案）
    if not getattr(args, 'use_interpretable_expansion', True):
        # 如果未启用，使用传统的密度阈值判断
        should_expand = len(node.papers) > args.max_density
        return should_expand, {
            'should_expand': should_expand,
            'reasoning': f'使用传统阈值判断: 文档数={len(node.papers)}, 阈值={args.max_density}',
            'cluster_cohesion': 'unknown',
            'has_clear_types': False
        }
    
    # 获取聚类标签列名
    cluster_label_col = getattr(args, 'cluster_label_col', 'cluster_label')
    
    # 检查是否有聚类信息
    has_cluster_label = any(
        doc.metadata.get(cluster_label_col) is not None 
        for doc in node.papers.values()
    )
    
    if not has_cluster_label:
        # 如果没有聚类信息，使用传统判断
        should_expand = len(node.papers) > args.max_density
        return should_expand, {
            'should_expand': should_expand,
            'reasoning': f'无聚类信息，使用传统阈值判断: 文档数={len(node.papers)}, 阈值={args.max_density}',
            'cluster_cohesion': 'unknown',
            'has_clear_types': False
        }
    
    # 步骤1: 统计聚类分布
    cluster_distribution = {}
    for doc in node.papers.values():
        cluster_id = doc.metadata.get(cluster_label_col)
        if cluster_id is not None:
            cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1
    
    print(f'  评估深度扩展: {len(node.papers)} 份文档分布在 {len(cluster_distribution)} 个cluster中')
    
    # 步骤2: 采样文档（代表性样本）
    # 从不同的cluster中采样，确保多样性
    sampled_docs = []
    docs_by_cluster = {}
    for doc in node.papers.values():
        cluster_id = doc.metadata.get(cluster_label_col)
        if cluster_id not in docs_by_cluster:
            docs_by_cluster[cluster_id] = []
        docs_by_cluster[cluster_id].append(doc)
    
    # 从每个cluster采样1-2份
    for cluster_id, docs in docs_by_cluster.items():
        sample_count = min(2, len(docs))
        sampled_docs.extend(random.sample(docs, sample_count))
    
    # 限制总样本数
    if len(sampled_docs) > 15:
        sampled_docs = random.sample(sampled_docs, 15)
    
    print(f'  采样 {len(sampled_docs)} 份文档进行评估')
    
    # 步骤3: 调用LLM评估
    prompt_text, image_base64_list = depth_expansion_evaluation_prompt(
        node, cluster_distribution, sampled_docs, ancestors
    )
    
    eval_prompt = constructPrompt(
        args,
        depth_expansion_evaluation_system_instruction,
        prompt_text
    )
    
    # 如果有图片，传递第一张作为代表
    images_for_llm = None
    if image_base64_list:
        images_for_llm = [image_base64_list[0]]
        print(f'  评估包含 {len(image_base64_list)} 张图片样本')
    
    try:
        args.llm = 'gpt'  # 评估使用更强的模型
        eval_outputs = promptLLM(
            args=args,
            prompts=[eval_prompt],
            schema=DepthExpansionEvaluationSchema,
            max_new_tokens=800,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.3,
            top_p=0.95,
            images_base64=images_for_llm
        )
        
        args.llm = 'vllm'  # 恢复使用vllm
        
        if len(eval_outputs) > 0:
            result = eval_outputs[0]
            if isinstance(result, str):
                result = safe_json_loads(result, default={}, log_error=True)
            
            if isinstance(result, dict):
                should_expand = result.get('should_expand', False)
                reasoning = result.get('reasoning', '')
                cohesion = result.get('cluster_cohesion', 'unknown')
                has_clear_types = result.get('has_clear_types', False)
                
                print(f'  评估结果: {"✓ 应该扩展" if should_expand else "✗ 不应该扩展"}')
                print(f'  聚类凝聚度: {cohesion}')
                print(f'  明确类型特征: {"是" if has_clear_types else "否"}')
                print(f'  判断理由: {reasoning}')
                
                return should_expand, result
        
    except Exception as e:
        print(f'  评估失败: {str(e)}，使用传统阈值判断')
        args.llm = 'vllm'
    
    # 评估失败，使用传统判断
    should_expand = len(node.papers) > args.max_density
    return should_expand, {
        'should_expand': should_expand,
        'reasoning': f'评估失败，回退到传统判断: 文档数={len(node.papers)}, 阈值={args.max_density}',
        'cluster_cohesion': 'unknown',
        'has_clear_types': False
    }


def expandNodeDepth(args, node, id2node, label2node):
    """
    深度扩展: 为叶节点生成子类别

    流程：
    1. 评估是否应该扩展（Region Schema / 可解释增强 / 密度阈值）
    2. 如果决定扩展，生成子类别（cluster 分组 / 传统全局）

    Returns:
        tuple: (new_children_labels, success)
    """
    # 构建祖先路径
    node_ancestors = node.get_ancestors()
    if node_ancestors is None:
        ancestors = "无"
    else:
        node_ancestors.reverse()
        ancestors = " -> ".join([ancestor.label for ancestor in node_ancestors])

    # ── 步骤1: 评估是否应该扩展 ──
    use_region = getattr(args, 'use_region_based_expansion', False)
    use_interpretable = getattr(args, 'use_interpretable_expansion', False)

    if use_region or use_interpretable:
        mode_label = "Region Schema" if use_region else "可解释增强"
        print(f'  ┌─ [{mode_label}] 评估节点 "{node.label}" (level={node.level}, docs={len(node.papers)})')

        should_expand, eval_result = evaluateDepthExpansion(args, node, ancestors)
        reasoning = eval_result.get('reasoning', '')

        if not should_expand:
            print(f'  └─ [{mode_label}] ✗ 决定不扩展: {reasoning}')
            return [], False

        print(f'  │  [{mode_label}] ✓ 决定扩展: {reasoning}')
        if use_region and eval_result.get('candidates'):
            for c in eval_result['candidates']:
                print(f'  │    候选子类: {c.get("label", "?")} — {c.get("description", "")}')
        print(f'  └─ 开始生成子类别...')
    else:
        # 纯密度阈值
        if len(node.papers) <= args.max_density:
            print(f'  ✗ 文档数未超过阈值 ({len(node.papers)} <= {args.max_density})，不扩展')
            return [], False
        print(f'  ┌─ [密度阈值] 文档数={len(node.papers)} > 阈值={args.max_density}，执行扩展')
        print(f'  └─ 开始生成子类别...')

    # ── 步骤2: 生成子类别 ──
    use_cluster_based = getattr(args, 'use_cluster_based_expansion', True)
    if use_cluster_based:
        cluster_label_col = getattr(args, 'cluster_label_col', 'cluster_label')
        has_cluster_label = any(
            doc.metadata.get(cluster_label_col) is not None
            for doc in node.papers.values()
        )
        if has_cluster_label:
            print(f'  使用基于 {cluster_label_col} 的分组扩展')
            return _expandNodeDepthWithClusters(args, node, id2node, label2node, ancestors)

    print(f'  使用传统的全局扩展方法')
    return _expandNodeDepthTraditional(args, node, id2node, label2node, ancestors)
