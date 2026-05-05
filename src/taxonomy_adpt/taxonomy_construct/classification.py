"""
企业文档分类模块
"""

import json
from src.taxonomy_adpt.taxonomy_construct.utils import clean_json_string, safe_json_loads_list
from src.taxonomy_adpt.llm_client.llm_adapter import constructPrompt, promptLLM
from src.taxonomy_adpt.taxonomy_construct.prompts import classify_prompt, ClassifySchema


def classify_documents(args, node, label2node, visited):
    """
    将父节点的文档分类到子节点（支持图片辅助分类）
    
    Args:
        args: 参数对象
        node: 当前节点
        label2node: 标签到节点的映射
        visited: 已访问的节点集合
    
    Returns:
        分类结果列表
    """
    # 初始化子节点的文档集合
    for child_label, child in node.get_children().items():
        if child.id not in visited:
            child.papers = {}
    
    # 为每个文档生成分类prompt和图片
    prompts = []
    images_base64 = []
    for doc_id, doc in node.papers.items():
        prompts.append(classify_prompt(node, doc))
        # 获取文档的第一张图片（如果有）
        if hasattr(doc, 'get_image_base64'):
            images_base64.append(doc.get_image_base64(0))  # 获取第一张图片
        else:
            images_base64.append(None)
    
    if len(prompts) == 0:
        return []
    
    # 调用LLM进行分类（支持多模态输入）
    output = promptLLM(
        args, 
        prompts, 
        schema=ClassifySchema, 
        max_new_tokens=3000,
        images_base64=images_base64,
        timeout_per_request=getattr(args, 'timeout_per_request', 120.0)
    )
    
    # 使用 return_none_on_error=True 保持位置对应
    output_dict = safe_json_loads_list(output, log_error=True, return_none_on_error=True)
    
    # 检查输出数量是否匹配
    if len(output_dict) != len(node.papers):
        print(f"  警告: 分类输出数量 ({len(output_dict)}) 与文档数量 ({len(node.papers)}) 不匹配")
        # 补齐
        if len(output_dict) < len(node.papers):
            output_dict.extend([None] * (len(node.papers) - len(output_dict)))
    
    class_options = [c for c in node.get_children()]
    class_map = {c: 0 for c in node.get_children()}
    class_map['unlabeled'] = 0
    
    # 处理分类结果（单标签分类）
    failed_count = 0
    for (doc_id, doc), out_labels in zip(node.papers.items(), output_dict):
        if out_labels is None:
            failed_count += 1
            class_map['unlabeled'] += 1
            continue
        
        # 获取单个类别标签ID
        label_id = out_labels.get('class_label', -1)
        
        # 如果标签ID为-1或None，表示不属于任何类别
        if label_id is None or label_id == -1 or label_id == "None":
            class_map['unlabeled'] += 1
            continue
        
        # 找到对应的标签
        label = None
        for child_label, child in node.children.items():
            if child.id == label_id:
                label = child_label
                break
        
        if label is None:
            class_map['unlabeled'] += 1
            continue
        
        full_label = label + f'_{node.dimension}'
        
        if "None" in str(label):
            class_map['unlabeled'] += 1
            continue
        elif (full_label in label2node) and (label in class_options):
            label2node[full_label].papers[doc_id] = doc
            class_map[label] += 1
            # 单标签分类：直接设置标签（替换而非追加）
            doc.labels[node.dimension] = [label]
        else:
            class_map['unlabeled'] += 1
    
    if failed_count > 0:
        print(f'  警告: {failed_count}/{len(node.papers)} 个文档解析失败')
    
    print(f'  分类结果: {str(class_map)}')
    return output_dict
