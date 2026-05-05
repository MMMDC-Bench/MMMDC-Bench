"""
企业文档树TaxoAdapt主程序
基于TaxoAdapt框架,适配企业文档场景
"""

import os
import sys
import json
from collections import deque
from contextlib import redirect_stdout
import argparse
from tqdm import tqdm
import pandas as pd

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 上两级目录（MMMDC-Bench）
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用新的 LLM 适配器
from src.taxonomy_adpt.llm_client.llm_adapter import initializeLLM, promptLLM, constructPrompt
from src.taxonomy_adpt.taxonomy_construct.taxonomy import Node, DAG
from src.taxonomy_adpt.taxonomy_construct.utils import clean_json_string, safe_json_loads, safe_json_loads_list

# 导入企业文档专用模块
from src.taxonomy_adpt.taxonomy_construct.document import EnterpriseDocument
from src.taxonomy_adpt.taxonomy_construct.prompts import (
    multi_dim_prompt, NodeListSchema,
    type_cls_system_instruction, type_cls_main_prompt, TypeClsSchema,
    ClassifySchema
)
from src.taxonomy_adpt.taxonomy_construct.expansion import expandNodeWidth, expandNodeDepth
from src.taxonomy_adpt.taxonomy_construct.classification import classify_documents
from src.taxonomy_adpt.taxonomy_construct.checkpoint_manager import CheckpointManager, auto_save_checkpoint
from src.taxonomy_adpt.taxonomy_construct.taxonomy_io import (
    export_taxonomy_structure, import_taxonomy_structure, print_taxonomy_structure
)


def load_documents_from_json(json_file_path, dimensions, max_docs=None, sample=False, seed=42):
    """
    从JSON文件加载企业文档
    
    JSON格式示例:
    [
        {
            "id": 1,
            "title": "2024年度财务预算报告",
            "content": "本报告总结了2024年度...",
            "metadata": {
                "department": "财务部",
                "doc_type": "报告",
                "created_at": "2024-01-15",
                "author": "张三"
            },
            "image_url_list": ["path/to/image1.png", "path/to/image2.png"]  // 可选：图片路径列表
        },
        ...
    ]
    
    Args:
        json_file_path: JSON文件路径
        dimensions: 维度列表
        max_docs: 最大文档数量（None表示全部加载）
        sample: 是否随机采样（True）还是顺序截取（False）
        seed: 随机种子
    """
    documents = {}
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_docs = len(data)
    
    # 处理数据采样
    if max_docs is not None and max_docs < total_docs:
        if sample:
            # 随机采样
            import random
            random.seed(seed)
            data = random.sample(data, max_docs)
            print(f"随机采样 {max_docs}/{total_docs} 份文档（种子: {seed}）")
        else:
            # 顺序截取
            data = data[:max_docs]
            print(f"顺序加载前 {max_docs}/{total_docs} 份文档")
    else:
        print(f"加载全部 {total_docs} 份文档")
    
    for index, item in tqdm(enumerate(data), total=len(data), desc="加载文档"):
        # doc_id = item.get('id')
        doc_id = index
        title = item.get('title', '')
        content = item.get('content', '')
        metadata = item.get('metadata', {})
        
        # 处理 image_url_list（可能是字符串形式的列表）
        image_url_list = item.get('image_url_list', None)
        if image_url_list is not None:
            if isinstance(image_url_list, str):
                # 如果是字符串形式的列表，尝试解析
                image_str = image_url_list.strip()
                if image_str.startswith('[') and image_str.endswith(']'):
                    try:
                        import ast
                        image_url_list = ast.literal_eval(image_str)
                    except (ValueError, SyntaxError):
                        try:
                            image_url_list = json.loads(image_str)
                        except json.JSONDecodeError:
                            # 解析失败，当作单个路径
                            image_url_list = [image_str] if image_str else None
                elif image_str:
                    # 普通字符串，当作单个路径
                    image_url_list = [image_str]
                else:
                    image_url_list = None
            elif not isinstance(image_url_list, list):
                # 其他类型，转换为列表
                image_url_list = [str(image_url_list)]
        
        doc = EnterpriseDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            metadata=metadata,
            label_opts=dimensions,
            image_url_list=image_url_list
        )
        documents[doc_id] = doc
    
    print(f"成功加载 {len(documents)} 份企业文档")
    # 统计有图片的文档数量和图片总数
    doc_with_images = sum(1 for doc in documents.values() if doc.has_images())
    total_images = sum(doc.get_image_count() for doc in documents.values())
    if total_images > 0:
        print(f"其中 {doc_with_images} 份文档包含图片，共计 {total_images} 张图片")
    return documents


def load_documents_from_table(table_file_path, dimensions, title_col='title', content_col='content', image_col='image_url_list', max_docs=None, sample=False, seed=42):
    """
    从表格文件加载企业文档（支持CSV、Excel等）
    
    Args:
        table_file_path: 表格文件路径（.csv, .xlsx, .xls）
        dimensions: 维度列表
        title_col: 标题列名
        content_col: 内容列名
        image_col: 图片路径列名（可选，如果不存在则忽略）
    
    Returns:
        documents: 文档字典
    """
    documents = {}
    
    # 根据文件扩展名读取表格
    file_ext = os.path.splitext(table_file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            df = pd.read_csv(table_file_path, encoding='utf-8')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(table_file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}。仅支持 .csv, .xlsx, .xls")
        
        print(f"成功读取表格文件: {table_file_path}")
        print(f"  - 行数: {len(df)}")
        print(f"  - 列数: {len(df.columns)}")
        print(f"  - 列名: {list(df.columns)}")
        
        # 处理数据采样
        total_docs = len(df)
        if max_docs is not None and max_docs < total_docs:
            if sample:
                # 随机采样
                df = df.sample(n=max_docs, random_state=seed)
                print(f"  - 随机采样 {max_docs}/{total_docs} 行（种子: {seed}）")
            else:
                # 顺序截取
                df = df.head(max_docs)
                print(f"  - 顺序加载前 {max_docs}/{total_docs} 行")
        else:
            print(f"  - 加载全部 {total_docs} 行")
        
        # 检查必需列
        if title_col not in df.columns:
            raise ValueError(f"未找到标题列 '{title_col}'，可用列: {list(df.columns)}")
        if content_col not in df.columns:
            raise ValueError(f"未找到内容列 '{content_col}'，可用列: {list(df.columns)}")
        
        # 检查图片列是否存在
        has_image_col = image_col in df.columns
        if has_image_col:
            print(f"  - 检测到图片列: {image_col}")
        else:
            print(f"  - 未检测到图片列 '{image_col}'（可选）")
        
        # 处理每一行
        for index, row in tqdm(df.iterrows(), total=len(df), desc="加载文档"):
            doc_id = index
            title = str(row[title_col]) if pd.notna(row[title_col]) else ''
            content = str(row[content_col]) if pd.notna(row[content_col]) else ''
            
            # 处理图片列
            image_url_list = None
            if has_image_col and pd.notna(row[image_col]):
                image_value = row[image_col]
                
                # 如果已经是列表，直接使用
                if isinstance(image_value, list):
                    image_url_list = [str(url).strip() for url in image_value if url]
                else:
                    # 转换为字符串
                    image_str = str(image_value).strip()
                    
                    if not image_str:
                        image_url_list = None
                    else:
                        # 尝试解析为JSON列表（处理字符串形式的列表，如 "['img1.png', 'img2.png']" 或 '["img1.png"]'）
                        if image_str.startswith('[') and image_str.endswith(']'):
                            try:
                                import ast
                                # 使用 ast.literal_eval 安全地解析
                                parsed = ast.literal_eval(image_str)
                                if isinstance(parsed, list):
                                    image_url_list = [str(url).strip() for url in parsed if url]
                                else:
                                    image_url_list = [str(parsed).strip()]
                            except (ValueError, SyntaxError):
                                # 如果解析失败，尝试用 json.loads
                                try:
                                    parsed = json.loads(image_str)
                                    if isinstance(parsed, list):
                                        image_url_list = [str(url).strip() for url in parsed if url]
                                    else:
                                        image_url_list = [str(parsed).strip()]
                                except json.JSONDecodeError:
                                    # 仍然失败，当作普通字符串处理
                                    image_url_list = [image_str]
                        # 支持多种分隔符：逗号、分号、换行符
                        elif ',' in image_str:
                            image_url_list = [url.strip() for url in image_str.split(',') if url.strip()]
                        elif ';' in image_str:
                            image_url_list = [url.strip() for url in image_str.split(';') if url.strip()]
                        elif '\n' in image_str:
                            image_url_list = [url.strip() for url in image_str.split('\n') if url.strip()]
                        else:
                            # 单个图片路径
                            image_url_list = [image_str]
            
            # 所有其他列作为metadata
            metadata = {}
            for col in df.columns:
                # 跳过title、content、image_url_list列
                if col not in [title_col, content_col, image_col]:
                    value = row[col]
                    # 处理NaN值
                    if pd.notna(value):
                        # 保持原始数据类型
                        if isinstance(value, (int, float, bool)):
                            metadata[col] = value
                        else:
                            metadata[col] = str(value)
                    else:
                        metadata[col] = None
            
            doc = EnterpriseDocument(
                doc_id=doc_id,
                title=title,
                content=content,
                metadata=metadata,
                label_opts=dimensions,
                image_url_list=image_url_list
            )
            documents[doc_id] = doc
        
        print(f"成功加载 {len(documents)} 份企业文档")
        
        # 统计有图片的文档数量和图片总数
        doc_with_images = sum(1 for doc in documents.values() if doc.has_images())
        total_images = sum(doc.get_image_count() for doc in documents.values())
        if doc_with_images > 0:
            print(f"其中 {doc_with_images} 份文档包含图片，共计 {total_images} 张图片")
        
        # 打印metadata统计
        if documents:
            sample_doc = list(documents.values())[0]
            print(f"Metadata字段数量: {len(sample_doc.metadata)}")
            print(f"Metadata字段名: {list(sample_doc.metadata.keys())}")
        
        return documents
        
    except Exception as e:
        print(f"错误: 加载表格文件失败 - {str(e)}")
        raise


def load_documents_from_directory(directory_path, dimensions):
    """
    从目录加载企业文档(支持txt, md等文本文件)
    """
    documents = {}
    doc_id = 0
    
    supported_extensions = ['.txt', '.md', '.doc', '.docx']
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 使用文件名作为标题
                    title = os.path.splitext(file)[0]
                    
                    # 从路径提取部门信息(假设目录结构为: root/部门/文件)
                    path_parts = root.replace(directory_path, '').strip(os.sep).split(os.sep)
                    department = path_parts[0] if path_parts else 'unknown'
                    
                    metadata = {
                        'department': department,
                        'file_path': file_path,
                        'doc_type': 'unknown'
                    }
                    
                    doc = EnterpriseDocument(
                        doc_id=doc_id,
                        title=title,
                        content=content,
                        metadata=metadata,
                        label_opts=dimensions
                    )
                    documents[doc_id] = doc
                    doc_id += 1
                    
                except Exception as e:
                    print(f"警告: 无法加载文件 {file_path}: {str(e)}")
    
    print(f"成功从目录加载 {len(documents)} 份企业文档")
    return documents


def initialize_enterprise_DAG(args):
    """初始化企业文档的多维度DAG"""
    roots = {}
    id2node = {}
    label2node = {}
    idx = 0
    
    print(f"\n开始初始化企业文档树的 {len(args.dimensions)} 个维度...")
    
    # 检查是否有预定义的doc_type配置文件
    predefined_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'predefined_doc_type.json'
    )
    use_predefined_doc_type = 'doc_type' in args.dimensions and os.path.exists(predefined_config_path)
    
    if use_predefined_doc_type:
        print(f"\n发现预定义的doc_type配置文件: {predefined_config_path}")
        try:
            from src.taxonomy_adpt.taxonomy_construct.taxonomy_io import import_taxonomy_structure
            predefined_roots, predefined_id2node, predefined_label2node = import_taxonomy_structure(predefined_config_path)
            
            if 'doc_type' in predefined_roots:
                roots['doc_type'] = predefined_roots['doc_type']
                # 合并到id2node和label2node
                id2node.update(predefined_id2node)
                label2node.update(predefined_label2node)
                idx = max(id2node.keys()) + 1 if id2node else 0
                print(f"  成功加载预定义的doc_type树（{len(predefined_id2node)}个节点）")
        except Exception as e:
            print(f"  警告: 加载预定义doc_type配置失败: {str(e)}")
            print(f"  将使用LLM生成doc_type维度")
            use_predefined_doc_type = False
    
    # 初始化其他维度的根节点
    for dim in args.dimensions:
        if use_predefined_doc_type and dim == 'doc_type':
            # doc_type已经从配置文件加载，跳过
            continue
            
        mod_topic = args.topic  # 保持原始中文名称
        mod_topic_code = args.topic.replace(' ', '_').replace('企业', 'enterprise').lower()
        mod_full_topic = mod_topic_code + f"_{dim}"
        
        root = Node(
            id=idx,
            label=mod_topic,
            code=mod_topic_code,
            dimension=dim
        )
        roots[dim] = root
        id2node[idx] = root
        label2node[mod_full_topic] = root
        idx += 1
    
    # 为需要扩展的根节点创建队列（排除已经从配置加载的doc_type）
    queue = deque()
    for node in roots.values():
        # 如果是doc_type且已经从配置文件加载，检查是否需要继续扩展
        if use_predefined_doc_type and node.dimension == 'doc_type':
            # 如果预定义的doc_type根节点已有子节点，将子节点加入队列以便继续扩展
            if len(node.children) > 0 and args.init_levels > 1:
                for child in node.children.values():
                    if child.level < args.init_levels:
                        queue.append(child)
            continue
        queue.append(node)
    
    # 为每个根节点生成初始子节点
    while queue:
        curr_node = queue.popleft()
        label = curr_node.label
        dim = curr_node.dimension
        
        print(f"\n正在扩展节点: {label} (维度: {dim}, 层级: {curr_node.level})")
        
        # 使用LLM生成子节点
        system_instruction, main_prompt, json_output_format = multi_dim_prompt(curr_node)
        prompts = [constructPrompt(args, system_instruction, main_prompt + "\n\n" + json_output_format)]
        
        try:
            outputs = promptLLM(
                args=args, 
                prompts=prompts, 
                schema=NodeListSchema, 
                max_new_tokens=3000, 
                json_mode=True, 
                temperature=0.01, 
                top_p=1.0,
                timeout_per_request=args.timeout_per_request
            )[0]
            
            outputs = safe_json_loads(outputs, default={}, log_error=True)
            outputs = outputs.get('root_topic', outputs.get(label, {}))
        except Exception as e:
            print(f"错误: 调用LLM生成节点时失败: {str(e)}")
            outputs = {}
        
        try:
            # 收集已存在的code，确保唯一性
            existing_codes = {node.code for node in id2node.values()}
            
            # 找到当前最大的id，确保新节点的id不会重复
            max_id = max(id2node.keys()) if id2node else -1
            
            # 添加所有子节点
            for key, value in outputs.items():
                # key是中文名称，保持不变
                child_label = key
                
                # 获取LLM生成的code，如果没有则自动生成
                child_code = value.get('code', None)
                if not child_code:
                    from src.taxonomy_adpt.taxonomy_construct.utils import generate_code_from_name
                    child_code = generate_code_from_name(child_label, existing_codes)
                else:
                    # 确保LLM生成的code符合规范（小写、下划线）
                    child_code = child_code.lower().replace(' ', '_').replace('-', '_')
                    # 确保唯一性
                    if child_code in existing_codes:
                        from src.taxonomy_adpt.taxonomy_construct.utils import generate_code_from_name
                        child_code = generate_code_from_name(child_label, existing_codes)
                
                existing_codes.add(child_code)
                
                # 使用code作为完整标识
                mod_full_key = child_code + f"_{dim}"
                
                if mod_full_key not in label2node:
                    max_id += 1  # 递增id
                    child_node = Node(
                        id=max_id,
                        label=child_label,
                        code=child_code,
                        dimension=dim,
                        description=value.get('description', ''),
                        parents=[curr_node]
                    )
                    curr_node.add_child(child_code, child_node)  # 使用code作为key
                    id2node[child_node.id] = child_node
                    label2node[mod_full_key] = child_node
                    
                    print(f"  - 添加子节点: {child_label} (code: {child_code})")
                    
                    # 如果未达到初始层级,继续扩展
                    if child_node.level < args.init_levels:
                        queue.append(child_node)
                    
                elif curr_node.code + f"_{dim}" in label2node and label2node[mod_full_key] in label2node[curr_node.code + f"_{dim}"].get_ancestors():
                    # 检查是否会形成循环（子节点是当前节点的祖先）
                    print(f"  - 跳过: {child_label} 会形成循环")
                    continue
                else:
                    # 节点已存在,添加父子关系
                    child_node = label2node[mod_full_key]
                    curr_node.add_child(child_code, child_node)  # 使用code作为key
                    child_node.add_parent(curr_node)
                    print(f"  - 复用已存在节点: {child_label} (code: {child_code})")
                    
        except Exception as e:
            print(f"错误: 扩展节点 {label} 时失败: {str(e)}")
    
    return roots, id2node, label2node


def main(args):
    """主函数"""
    
    print("=" * 80)
    print("企业文档树 TaxoAdapt 系统")
    print("=" * 80)
    
    # 用于追踪是否正常完成
    success = False
    
    # ========== 初始化Checkpoint管理器 ==========
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints")
    )
    
    # ========== 检查是否从checkpoint恢复 ==========
    if args.resume:
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            print(f"\n【恢复模式】从checkpoint恢复: {latest_checkpoint}")
            checkpoint_data = checkpoint_manager.load_checkpoint(latest_checkpoint)
            
            roots = checkpoint_data['roots']
            id2node = checkpoint_data['id2node']
            label2node = checkpoint_data['label2node']
            documents = checkpoint_data['documents']
            visited = checkpoint_data['visited']
            start_iteration = checkpoint_data['metadata'].get('iteration', 0) + 1
            
            print(f"成功恢复状态，从迭代 {start_iteration} 继续运行")
            
            # 初始化LLM
            args = initializeLLM(args)
            
            # 跳到步骤4继续执行
            dags = {dim: DAG(root=root, dim=dim) for dim, root in roots.items()}
            
        else:
            print("\n【警告】未找到可用的checkpoint，将从头开始运行")
            args.resume = False
    
    if not args.resume:
        # ========== 步骤1: 加载文档 ==========
        print("\n【步骤1】加载企业文档集合...")
        
        # 显示数据量限制信息
        if args.max_docs is not None:
            sample_mode = "随机采样" if args.sample_docs else "顺序加载"
            print(f"【调试模式】数据量限制: 最多 {args.max_docs} 份文档（{sample_mode}）")
        
        if args.input_json:
            documents = load_documents_from_json(
                args.input_json, 
                args.dimensions,
                max_docs=args.max_docs,
                sample=args.sample_docs,
                seed=args.seed
            )
        elif args.input_table:
            documents = load_documents_from_table(
                args.input_table, 
                args.dimensions,
                title_col=args.title_col,
                content_col=args.content_col,
                image_col=args.image_col,
                max_docs=args.max_docs,
                sample=args.sample_docs,
                seed=args.seed
            )
        elif args.input_dir:
            documents = load_documents_from_directory(args.input_dir, args.dimensions)
        else:
            raise ValueError("必须指定 --input_json、--input_table 或 --input_dir 参数")
        
        # ========== 步骤2: 初始化LLM和DAG ==========
        print("\n【步骤2】初始化LLM和企业文档分类体系...")
        args = initializeLLM(args)
        
        # 检查是否导入已有的分类体系
        if args.import_taxonomy:
            print(f"  【导入模式】从文件导入分类体系: {args.import_taxonomy}")
            roots, id2node, label2node = import_taxonomy_structure(
                args.import_taxonomy, 
                format=args.taxonomy_format
            )
            print("\n  导入的分类体系结构预览（前2层）:")
            print_taxonomy_structure(roots, max_depth=2)
        else:
            print("  【构建模式】从头构建分类体系...")
            roots, id2node, label2node = initialize_enterprise_DAG(args)
    
        # 保存初始分类体系
        for dim in args.dimensions:
            output_file = os.path.join(args.output_dir, f'initial_taxonomy_{dim}.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    roots[dim].display(0, indent_multiplier=5)
            print(f"初始分类体系已保存: {output_file}")
        
        # ========== 步骤3: 文档维度分类 ==========
        print("\n【步骤3】对文档进行多维度分类...")
        
        args.llm = 'vllm'  # 使用vllm加速批量推理
        dags = {dim: DAG(root=root, dim=dim) for dim, root in roots.items()}
        
        # 使用LLM判断每个文档属于哪些维度
        prompts = [
            constructPrompt(args, type_cls_system_instruction, type_cls_main_prompt(doc)) 
            for doc in documents.values()
        ]
        
        outputs = promptLLM(
            args=args, 
            prompts=prompts, 
            schema=TypeClsSchema, 
            max_new_tokens=500, 
            json_mode=True, 
            temperature=0.1, 
            top_p=0.99,
            timeout_per_request=args.timeout_per_request
        )
        
        # 使用 return_none_on_error=True 保持位置对应
        outputs = safe_json_loads_list(outputs, log_error=True, return_none_on_error=True)
        
        # 初始化根节点的文档集合
        for r in roots:
            roots[r].papers = {}
        
        type_dist = {dim: [] for dim in args.dimensions}
        
        # 检查输出数量是否与文档数量匹配
        if len(outputs) != len(documents):
            print(f"  警告: 输出数量 ({len(outputs)}) 与文档数量 ({len(documents)}) 不匹配")
            # 补齐或截断
            if len(outputs) < len(documents):
                outputs.extend([None] * (len(documents) - len(outputs)))
            else:
                outputs = outputs[:len(documents)]
        
        failed_count = 0
        for doc_id, out in enumerate(outputs):
            if out is None:
                failed_count += 1
                # 初始化空标签，以便文档仍然在系统中
                documents[doc_id].labels = {dim: [] for dim in args.dimensions}
                continue
            
            documents[doc_id].labels = {}
            
            for key, val in out.items():
                # 只处理预定义的维度，忽略其他字段（如 'error'）
                if key not in args.dimensions:
                    if key not in ['error', 'message']:  # 忽略常见的错误字段
                        print(f"  警告: 文档 {doc_id} 返回了未定义的维度 '{key}'，已忽略")
                    continue
                
                if val:
                    type_dist[key].append(documents[doc_id])
                    documents[doc_id].labels[key] = []
                    roots[key].papers[doc_id] = documents[doc_id]
        
        if failed_count > 0:
            print(f"\n  总计: {failed_count}/{len(documents)} 个文档的初始分类失败（{failed_count/len(documents)*100:.1f}%）")
        
        print("\n维度分布统计:")
        for k, v in type_dist.items():
            print(f"  {k}: {len(v)} 份文档")
        
        visited = set()
        start_iteration = 0
        
        # 保存初始checkpoint
        print("\n【Checkpoint】保存初始状态...")
        checkpoint_manager.save_checkpoint(
            roots=roots,
            id2node=id2node,
            label2node=label2node,
            documents=documents,
            visited=visited,
            iteration=0,
            metadata={"stage": "初始化完成"}
        )
    
    # ========== 步骤4: 迭代分类和扩展 ==========
    print("\n【步骤4】迭代进行文档分类和分类体系扩展...")
    
    # 初始化队列
    if args.resume and args.refine:
        # 细化模式：找出所有需要重新分类的节点
        print("\n【细化模式】识别需要重新分类的节点...")
        queue = deque()
        refine_nodes = []
        
        for node_id, node in id2node.items():
            # 跳过没有子节点的叶子节点
            if len(node.children) == 0:
                continue
            
            # 检查是否有文档
            if len(node.papers) == 0:
                continue
            
            # 检查子节点是否有文档
            has_child_docs = False
            for child_label, child_node in node.children.items():
                if len(child_node.papers) > 0:
                    has_child_docs = True
                    break
            
            # 如果父节点有文档但子节点都没有文档，说明需要重新分类
            if not has_child_docs:
                refine_nodes.append(node)
                # 从visited中移除，允许重新分类
                if node.id in visited:
                    visited.discard(node.id)
                print(f"  - {node.label} ({node.dimension}): {len(node.papers)} 份文档, {len(node.children)} 个子节点")
        
        if refine_nodes:
            print(f"\n共发现 {len(refine_nodes)} 个需要细化的节点，加入队列...")
            queue.extend(refine_nodes)
        else:
            print("\n所有节点都已正确分类，无需细化。")
            # 如果没有需要细化的节点，仍然检查是否有节点需要继续深度扩展
            print("\n检查是否有节点需要继续扩展...")
            use_interpretable = getattr(args, 'use_interpretable_expansion', True)
            for node_id, node in id2node.items():
                if len(node.children) == 0 and node.level < args.max_depth:
                    if use_interpretable:
                        # 可解释增强模式：所有叶节点都考虑（由评估决定）
                        queue.append(node)
                        print(f"  - {node.label}: {len(node.papers)} 份文档，将进行智能评估")
                    elif len(node.papers) > args.max_density:
                        # 传统模式：只有超过阈值的才考虑
                        queue.append(node)
                        print(f"  - {node.label}: {len(node.papers)} 份文档，可继续扩展")
            
            if not queue:
                print("\n分类已经完成，无需进一步处理。")
    else:
        # 正常模式：从根节点开始
        queue = deque([roots[r] for r in roots])
    
    iteration = start_iteration
    max_iterations = 10000  # 防止无限循环（提高限制）
    
    # 跟踪队列中的节点，避免重复添加
    nodes_in_queue = {node.id for node in queue}
    
    print(f"\n开始处理，初始队列大小: {len(queue)}, 已访问节点数: {len(visited)}")
    
    while queue and iteration < max_iterations:
        iteration += 1
        curr_node = queue.popleft()
        
        # 从队列跟踪集合中移除
        nodes_in_queue.discard(curr_node.id)
        
        print(f"\n[迭代 {iteration}] 访问节点: {curr_node.label} ({curr_node.dimension})")
        print(f"  当前层级: {curr_node.level}, 文档数: {len(curr_node.papers)}, 队列剩余: {len(queue)}")
        
        try:
            if len(curr_node.children) > 0:
                # 节点已有子节点,执行分类
                # 修复：只有在未访问过时才执行分类
                if curr_node.id not in visited:
                    visited.add(curr_node.id)
                    
                    print(f"  执行文档分类...")
                    classify_documents(args, curr_node, label2node, visited)
                    # 剪枝：移除分类后仍无文档的空叶子节点
                    from src.taxonomy_adpt.taxonomy_construct.enrichment import prune_empty_leaves
                    pruned = prune_empty_leaves(curr_node, id2node, label2node)
                    
                    # 检查是否需要宽度扩展(发现未覆盖的新类别)
                    new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
                    if len(new_sibs) > 0:
                        print(f"  【宽度扩展】为 {curr_node.label} 新增 {len(new_sibs)} 个兄弟节点:")
                        for sib in new_sibs:
                            sib_node = label2node.get(sib)
                            desc = f" — {sib_node.description}" if sib_node and sib_node.description else ""
                            print(f"    + {sib}{desc}")
                        # 重新分类
                        classify_documents(args, curr_node, label2node, visited)
                        pruned = prune_empty_leaves(curr_node, id2node, label2node)
                else:
                    print(f"  节点已访问过，跳过分类")
                
                # 将符合条件的子节点加入队列
                # 修复：检查子节点是否已经访问过或已在队列中，避免重复处理
                for child_label, child_node in curr_node.children.items():
                    c_papers = label2node[child_label + f"_{curr_node.dimension}"].papers
                    
                    # 跳过已访问或已在队列中的节点
                    if child_node.id in visited:
                        pass  # 已访问，跳过
                    elif child_node.id in nodes_in_queue:
                        pass  # 已在队列中，跳过
                    # 只添加未访问过且符合条件的子节点
                    elif (child_node.level < args.max_depth) and (len(c_papers) > args.max_density):
                        queue.append(child_node)
                        nodes_in_queue.add(child_node.id)
                        print(f"    → 子节点 {child_node.label} 加入队列 (文档数: {len(c_papers)})")
                    elif child_node.level >= args.max_depth:
                        print(f"    ✗ 子节点 {child_node.label} 已达最大深度，跳过")
                    elif len(c_papers) <= args.max_density:
                        print(f"    ✓ 子节点 {child_node.label} 文档数已满足要求 ({len(c_papers)} <= {args.max_density})")
            else:
                # 叶节点,考虑执行深度扩展(生成子类别)
                if curr_node.level < args.max_depth:
                    # 标记已访问，避免重复尝试扩展
                    if curr_node.id not in visited:
                        visited.add(curr_node.id)
                    
                    new_children, success = expandNodeDepth(args, curr_node, id2node, label2node)
                    args.llm = 'vllm'
                    
                    if success and len(new_children) > 0:
                        print(f'  【深度扩展】为 "{curr_node.label}" 新增 {len(new_children)} 个子节点:')
                        for child_label in new_children:
                            child_node = label2node.get(child_label)
                            desc = f" — {child_node.description}" if child_node and child_node.description else ""
                            print(f"    + {child_label}{desc}")
                        visited.remove(curr_node.id)
                        if curr_node.id not in nodes_in_queue:
                            queue.append(curr_node)
                            nodes_in_queue.add(curr_node.id)
                    else:
                        print(f'  节点 "{curr_node.label}" 保持为叶节点（扩展被跳过或未创建子节点）')
                elif curr_node.level >= args.max_depth:
                    print(f"  已达到最大深度限制 (level={curr_node.level}, max_depth={args.max_depth})，不再扩展")
            
            # 自动保存checkpoint
            checkpoint_path = auto_save_checkpoint(
                checkpoint_manager=checkpoint_manager,
                roots=roots,
                id2node=id2node,
                label2node=label2node,
                documents=documents,
                visited=visited,
                iteration=iteration,
                save_interval=args.checkpoint_interval,
                metadata={
                    "current_node": curr_node.label,
                    "queue_size": len(queue)
                }
            )
            if checkpoint_path:
                print(f"  【Checkpoint】已保存: {os.path.basename(checkpoint_path)}")
        
        except KeyboardInterrupt:
            print(f"\n\n【用户中断】在迭代 {iteration} 时收到中断信号")
            print(f"【Checkpoint】正在保存当前状态...")
            try:
                interrupt_checkpoint = checkpoint_manager.save_checkpoint(
                    roots=roots,
                    id2node=id2node,
                    label2node=label2node,
                    documents=documents,
                    visited=visited,
                    iteration=iteration,
                    metadata={
                        "interrupted": True,
                        "current_node": curr_node.label if curr_node else "unknown",
                        "queue_size": len(queue)
                    }
                )
                print(f"【Checkpoint】已保存: {interrupt_checkpoint}")
                print(f"可以使用 --resume 参数从此checkpoint恢复")
            except Exception as save_error:
                print(f"【错误】无法保存checkpoint: {str(save_error)}")
            raise
        
        except Exception as e:
            print(f"\n【错误】迭代 {iteration} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            print(f"【Checkpoint】正在保存错误状态...")
            try:
                error_checkpoint = checkpoint_manager.save_checkpoint(
                    roots=roots,
                    id2node=id2node,
                    label2node=label2node,
                    documents=documents,
                    visited=visited,
                    iteration=iteration,
                    metadata={
                        "error": str(e),
                        "current_node": curr_node.label if curr_node else "unknown",
                        "queue_size": len(queue)
                    }
                )
                print(f"【Checkpoint】错误状态已保存: {error_checkpoint}")
                print(f"可以使用 --resume 参数从此checkpoint恢复")
            except Exception as save_error:
                print(f"【错误】无法保存checkpoint: {str(save_error)}")
            
            # 不要继续执行，直接抛出异常
            raise
    
    # 检查循环退出原因
    if queue:
        # 队列不为空但循环退出，说明达到了最大迭代次数
        print(f"\n【警告】已达到最大迭代次数限制 ({iteration}/{max_iterations})")
        print(f"  队列中还剩 {len(queue)} 个节点未处理")
        print(f"  已访问节点数: {len(visited)}")
        print(f"  总节点数: {len(id2node)}")
        
        # 显示队列中的前几个节点
        if queue:
            print(f"\n  队列中剩余节点（前5个）：")
            for i, node in enumerate(list(queue)[:5]):
                print(f"    {i+1}. {node.label} (level={node.level}, papers={len(node.papers)})")
        
        print(f"\n  建议：")
        print(f"    1. 检查是否存在死循环（同一节点被重复访问）")
        print(f"    2. 检查max_depth和max_density参数是否合理")
        print(f"    3. 可以增加max_iterations限制（当前：{max_iterations}）")
    else:
        print(f"\n【完成】所有节点已处理完毕")
        print(f"  总迭代次数: {iteration}")
        print(f"  已访问节点数: {len(visited)}")
        print(f"  总节点数: {len(id2node)}")
    
    # ========== 步骤4.5: 空节点清理 + 退化折叠 + Region Schema 富化 + 冗余兄弟合并 ==========

    # ---- 全局空叶子节点清理（无条件执行）----
    from src.taxonomy_adpt.taxonomy_construct.enrichment import prune_all_empty_leaves

    print("\n【步骤4.5 清理】移除空叶子节点（扩展后未获得任何文档的类别）...")
    total_pruned = prune_all_empty_leaves(roots, id2node, label2node)
    if total_pruned > 0:
        print(f"  ✓ 共剪枝 {total_pruned} 个空叶子节点")
        checkpoint_manager.save_checkpoint(
            roots=roots, id2node=id2node, label2node=label2node,
            documents=documents, visited=visited, iteration=iteration,
            metadata={"stage": "empty_leaves_pruned"}
        )
    else:
        print("  无空叶子节点")

    # ---- 退化折叠（无条件执行）：父子同名且为唯一子节点 → 折叠 ----
    from src.taxonomy_adpt.taxonomy_construct.enrichment import collapse_degenerate_nodes

    print("\n【步骤4.5 预处理】折叠退化节点（父子同名且为唯一子节点）...")
    collapsed = collapse_degenerate_nodes(roots, id2node, label2node)
    if collapsed > 0:
        print(f"  ✓ 共折叠 {collapsed} 个退化节点")
        checkpoint_manager.save_checkpoint(
            roots=roots, id2node=id2node, label2node=label2node,
            documents=documents, visited=visited, iteration=iteration,
            metadata={"stage": "degenerate_collapsed"}
        )
    else:
        print("  无退化节点")

    # ---- Region Schema 富化（可选） ----
    enable_region_enrichment = getattr(args, 'enable_region_enrichment', False)
    if enable_region_enrichment:
        from src.taxonomy_adpt.taxonomy_construct.enrichment import (
            enrich_all_leaf_region_schemas,
            compare_sibling_region_schemas,
            merge_sibling_nodes,
        )

        print("\n【步骤4.5a】为叶子节点生成 Region Schema...")
        saved_llm = args.llm
        args.llm = 'gpt'
        enrich_all_leaf_region_schemas(args, roots, id2node)
        args.llm = saved_llm

        # 保存中间 checkpoint
        checkpoint_manager.save_checkpoint(
            roots=roots, id2node=id2node, label2node=label2node,
            documents=documents, visited=visited, iteration=iteration,
            metadata={"stage": "region_schema_enriched"}
        )

        merge_redundant = getattr(args, 'merge_redundant_siblings', True)
        if merge_redundant:
            print("\n【步骤4.5b】自底向上迭代合并冗余兄弟节点...")
            args.llm = 'gpt'

            total_merged_all_rounds = 0
            max_merge_rounds = 10
            for merge_round in range(1, max_merge_rounds + 1):
                # 按层级从深到浅排序，自底向上处理
                parents_to_check = []
                for node in id2node.values():
                    leaf_children = [c for c in node.children.values() if len(c.children) == 0 and c.region_schema]
                    if len(leaf_children) >= 2:
                        parents_to_check.append(node)
                parents_to_check.sort(key=lambda n: n.level, reverse=True)

                if not parents_to_check:
                    break

                print(f"\n  ── 合并轮次 {merge_round}（{len(parents_to_check)} 个父节点待检查）──")

                round_merged = 0
                newly_became_leaf = []  # 本轮合并后变成叶子的节点

                for parent in parents_to_check:
                    if parent.id not in id2node:
                        continue
                    leaf_children = [c for c in parent.children.values() if len(c.children) == 0 and c.region_schema]
                    if len(leaf_children) < 2:
                        continue

                    print(f"\n  检查 \"{parent.label}\"（{len(leaf_children)} 个叶子子节点）:")
                    for lc in leaf_children:
                        region_count = len(lc.region_schema) if lc.region_schema else 0
                        print(f"    · {lc.label} — {region_count} 个顶层区域")

                    result = compare_sibling_region_schemas(args, parent)
                    if result.get('has_redundancy'):
                        print(f"  → 发现冗余: {result.get('reasoning', '')}")
                        for mg in result.get('merge_groups', []):
                            merged = merge_sibling_nodes(parent, mg, id2node, label2node)
                            if merged:
                                round_merged += 1
                        # 合并后检查：如果该父节点只剩 0 或 1 个子节点，它自身变成了"准叶子"
                        remaining = [c for c in parent.children.values() if len(c.children) == 0]
                        if len(parent.children) <= 1:
                            newly_became_leaf.append(parent)
                    else:
                        print(f"  → 无冗余")

                total_merged_all_rounds += round_merged
                print(f"\n  轮次 {merge_round} 完成: 合并 {round_merged} 组")

                if round_merged == 0:
                    break

                # 为本轮新产生的叶子节点补充 Region Schema，供下一轮对比
                from src.taxonomy_adpt.taxonomy_construct.enrichment import generate_region_schema_for_node
                import random as _rand

                for node in newly_became_leaf:
                    if node.region_schema:
                        continue
                    if len(node.children) == 1:
                        only_child = list(node.children.values())[0]
                        if only_child.region_schema:
                            node.region_schema = only_child.region_schema
                            node.region_schema_reasoning = f'继承自唯一子节点 "{only_child.label}"'
                            node.node_kv_schema = only_child.node_kv_schema
                            node.node_kv_schema_reasoning = f'继承自唯一子节点 "{only_child.label}"'
                            print(f'  ↑ "{node.label}" 继承子节点 "{only_child.label}" 的 schema')
                    elif len(node.children) == 0 and len(node.papers) > 0:
                        sample = list(node.papers.values())
                        if len(sample) > 8:
                            sample = _rand.sample(sample, 8)
                        anc = node.get_ancestors() or []
                        anc.reverse()
                        schema, kv_schema, reasoning = generate_region_schema_for_node(args, node, anc, sample)
                        if schema:
                            node.region_schema = schema
                            node.region_schema_reasoning = reasoning
                            node.node_kv_schema = kv_schema or {}
                            node.node_kv_schema_reasoning = reasoning
                            print(f'  + 为新叶子 "{node.label}" 生成了 Region Schema + node_kv_schema')

            args.llm = saved_llm

            if total_merged_all_rounds > 0:
                print(f"\n  迭代合并完成: 共 {merge_round} 轮，合并 {total_merged_all_rounds} 组")
                checkpoint_manager.save_checkpoint(
                    roots=roots, id2node=id2node, label2node=label2node,
                    documents=documents, visited=visited, iteration=iteration,
                    metadata={"stage": "siblings_merged"}
                )
            else:
                print(f"\n  无需合并")

        # ---- 步骤4.5c: 自底向上为非叶子节点生成抽象 node_kv_schema ----
        from src.taxonomy_adpt.taxonomy_construct.enrichment import propagate_node_kv_schemas_bottom_up

        print("\n【步骤4.5c】自底向上为非叶子节点生成抽象 node_kv_schema...")
        saved_llm_abs = args.llm
        args.llm = 'gpt'
        abstracted = propagate_node_kv_schemas_bottom_up(args, roots, id2node)
        args.llm = saved_llm_abs
        print(f"  ✓ 共 {abstracted} 个非叶子节点获得了抽象 node_kv_schema")

        checkpoint_manager.save_checkpoint(
            roots=roots, id2node=id2node, label2node=label2node,
            documents=documents, visited=visited, iteration=iteration,
            metadata={"stage": "node_kv_schema_propagated"}
        )

    # ========== 步骤5: 保存最终分类体系 ==========
    print("\n【步骤5】保存最终企业文档分类体系...")
    
    # 保存最终checkpoint
    print("\n【Checkpoint】保存最终状态...")
    checkpoint_manager.save_checkpoint(
        roots=roots,
        id2node=id2node,
        label2node=label2node,
        documents=documents,
        visited=visited,
        iteration=iteration,
        metadata={"stage": "完成"}
    )
    
    # 清理旧checkpoint
    if args.cleanup_checkpoints:
        checkpoint_manager.cleanup_old_checkpoints(keep_last_n=args.keep_checkpoints)
    
    for dim in args.dimensions:
        # 保存文本格式
        txt_file = os.path.join(args.output_dir, f'final_taxonomy_{dim}.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                taxo_dict = roots[dim].display(0, indent_multiplier=5)
        
        # 保存JSON格式
        json_file = os.path.join(args.output_dir, f'final_taxonomy_{dim}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(taxo_dict, f, ensure_ascii=False, indent=4)
        
        print(f"  维度 {dim} 分类体系已保存:")
        print(f"    - 文本格式: {txt_file}")
        print(f"    - JSON格式: {json_file}")
    
    # 保存文档分类结果
    doc_labels_file = os.path.join(args.output_dir, 'document_labels.json')
    
    # 过滤标签，只保留路径末端节点
    def filter_leaf_labels(labels_dict, all_label2node):
        """
        过滤文档标签，只保留路径末端的节点（移除中间的祖先节点）
        注意：在单标签分类模式下，每个维度最多只有一个标签
        
        Args:
            labels_dict: 文档的标签字典，格式为 {dimension: [label1, label2, ...]}
            all_label2node: 标签到节点的映射字典，key格式为 'label_dimension'
        
        Returns:
            过滤后的标签字典
        """
        filtered_labels = {}
        
        for dimension, labels in labels_dict.items():
            if not labels:
                filtered_labels[dimension] = []
                continue
            
            # 获取所有标签对应的节点
            label_nodes = []
            for label in labels:
                full_key = f"{label}_{dimension}"
                if full_key in all_label2node:
                    label_nodes.append((label, all_label2node[full_key]))
            
            # 过滤掉是其他节点祖先的节点
            leaf_labels = []
            seen_labels = set()  # 用于去重
            for label, node in label_nodes:
                # 跳过已经处理过的重复标签
                if label in seen_labels:
                    continue
                    
                is_ancestor = False
                # 检查这个节点是否是其他任何节点的祖先
                for other_label, other_node in label_nodes:
                    if label != other_label:
                        # 如果当前节点在other_node的祖先列表中，说明是祖先节点
                        if node in other_node.get_ancestors():
                            is_ancestor = True
                            break
                
                if not is_ancestor:
                    leaf_labels.append(label)
                    seen_labels.add(label)
            
            filtered_labels[dimension] = leaf_labels
        
        return filtered_labels
    
    doc_labels = {
        doc_id: {
            'title': doc.title,
            'labels': filter_leaf_labels(doc.labels, label2node),
            'metadata': doc.metadata
        }
        for doc_id, doc in documents.items()
    }
    with open(doc_labels_file, 'w', encoding='utf-8') as f:
        json.dump(doc_labels, f, ensure_ascii=False, indent=4)
    
    print(f"\n文档分类结果已保存（已过滤中间节点，仅保留路径末端节点）: {doc_labels_file}")
    
    # ========== 步骤6: 导出分类体系结构（如果指定）==========
    if args.export_taxonomy:
        print("\n【步骤6】导出分类体系结构...")
        export_taxonomy_structure(
            roots, 
            args.export_taxonomy, 
            format=args.taxonomy_format
        )
    
    success = True
    print("\n" + "=" * 80)
    print("企业文档树构建完成!")
    print("=" * 80)
    
    return success

def read_config():
    parser = argparse.ArgumentParser(description='企业文档树TaxoAdapt系统')
    
    # 输入参数
    parser.add_argument('--input_json', type=str, default=None,
                        help='输入文档的JSON文件路径')
    parser.add_argument('--input_table', type=str, default=None,
                        help='输入文档的表格文件路径（支持CSV、Excel）')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='输入文档的目录路径')
    
    # 表格列映射参数
    parser.add_argument('--title_col', type=str, default='title',
                        help='表格中标题列的列名（默认: title）')
    parser.add_argument('--content_col', type=str, default='content',
                        help='表格中内容列的列名（默认: content）')
    parser.add_argument('--image_col', type=str, default='image_url_list',
                        help='表格中图片路径列的列名（默认: image_url_list，可选）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output/enterprise_taxonomy',
                        help='输出目录')
    
    # 主题和维度
    parser.add_argument('--topic', type=str, default='企业文档',
                        help='文档树的根主题')
    parser.add_argument('--dimensions', type=str, nargs='+',
                        default=['doc_type', 'topic'],
                        help='文档的维度列表')
    
    # LLM参数
    parser.add_argument('--llm', type=str, default='gpt', choices=['gpt', 'vllm', 'claude'],
                        help='使用的LLM类型')
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                        help='LLM模型名称')
    parser.add_argument('--api_protocol', type=str, default='openai',
                        choices=['openai', 'anthropic', 'dashscope'],
                        help='大模型API协议类型（openai/anthropic/dashscope）')
    parser.add_argument('--base_url', type=str, default='',
                        help='大模型API基础地址（也可通过环境变量 BASE_URL 指定）')
    parser.add_argument('--api_key', type=str, default='',
                        help='大模型API Key（也可通过环境变量 API_KEY 指定）')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='单次请求最大重试次数')
    parser.add_argument('--retry_delay', type=float, default=2.0,
                        help='请求失败后的重试间隔（秒）')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='采样温度')
    parser.add_argument('--max_new_tokens', type=int, default=3000,
                        help='单次请求最大输出token数')
    parser.add_argument('--timeout_per_request', type=float, default=120.0,
                        help='每个LLM请求的超时时间（秒），默认120秒')
    parser.add_argument('--max_workers', type=int, default=3,
                        help='LLM批量请求的最大并发数（默认3，降低可减轻网关压力）')
    parser.add_argument('--max_qps', type=float, default=None,
                        help='LLM请求的每秒最大请求数（默认不限制，如网关不稳定建议设为2~5）')
    parser.add_argument('--max_batch_retries', type=int, default=2,
                        help='批量LLM请求中失败项的最大重试轮数（默认2）')
    
    # 扩展参数
    parser.add_argument('--max_depth', type=int, default=3,
                        help='分类体系的最大深度')
    parser.add_argument('--init_levels', type=int, default=1,
                        help='初始化时生成的层级数')
    parser.add_argument('--max_density', type=int, default=30,
                        help='触发扩展的文档密度阈值')
    parser.add_argument('--use_cluster_based_expansion', action='store_true', default=True,
                        help='启用基于cluster_label的分组扩展（推荐，可避免相似文档被赋予不同标签）')
    parser.add_argument('--no_cluster_based_expansion', action='store_false', dest='use_cluster_based_expansion',
                        help='禁用基于cluster_label的分组扩展，使用传统方法')
    parser.add_argument('--cluster_label_col', type=str, default='cluster_label',
                        help='聚类标签列的列名（默认: cluster_label）')
    parser.add_argument('--use_interpretable_expansion', action='store_true', default=False,
                        help='[已废弃] 启用可解释增强的深度扩展评估')
    parser.add_argument('--no_interpretable_expansion', action='store_false', dest='use_interpretable_expansion',
                        help='禁用可解释增强评估')
    parser.add_argument('--use_region_based_expansion', action='store_true', default=False,
                        help='[已废弃] 启用 Region Schema 驱动的深度扩展前评估（建议改用 --enable_region_enrichment）')
    parser.add_argument('--no_region_based_expansion', action='store_false', dest='use_region_based_expansion',
                        help='禁用 Region Schema 驱动的深度扩展前评估')
    
    # Region Schema 后置富化 + 合并参数
    parser.add_argument('--enable_region_enrichment', action='store_true',
                        help='扩展完成后为所有叶子节点生成 Region Schema（推荐）')
    parser.add_argument('--merge_redundant_siblings', action='store_true', default=True,
                        help='基于 Region Schema 对比合并冗余的兄弟节点（需配合 --enable_region_enrichment）')
    parser.add_argument('--no_merge_redundant_siblings', action='store_false', dest='merge_redundant_siblings',
                        help='禁用兄弟节点冗余合并')
    
    # [已废弃] Element Schema 参数 — 已被 Region Schema (--enable_region_enrichment) 取代
    parser.add_argument('--enable_schema_enrichment', action='store_true',
                        help='[已废弃] 旧的要素Schema富化，请改用 --enable_region_enrichment')
    parser.add_argument('--use_iterative_schema', action='store_true', default=True,
                        help='[已废弃] 迭代式Schema生成')
    parser.add_argument('--no_iterative_schema', action='store_false', dest='use_iterative_schema')
    parser.add_argument('--schema_refinement_rounds', type=int, default=3)
    parser.add_argument('--check_schema_redundancy', action='store_true')
    parser.add_argument('--schema_similarity_threshold', type=float, default=0.8)
    parser.add_argument('--min_schema_keys', type=int, default=3)
    
    # Checkpoint参数
    parser.add_argument('--resume', action='store_true',
                        help='从最新的checkpoint恢复运行')
    parser.add_argument('--refine', action='store_true',
                        help='细化模式：从checkpoint恢复并自动识别需要重新分类的节点（必须配合--resume使用）')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='checkpoint保存间隔（每N次迭代保存一次）')
    parser.add_argument('--cleanup_checkpoints', action='store_true',
                        help='运行结束后清理旧的checkpoint')
    parser.add_argument('--keep_checkpoints', type=int, default=5,
                        help='保留最近的N个checkpoint')
    
    # 调试参数
    parser.add_argument('--max_docs', type=int, default=None,
                        help='限制加载的最大文档数量（用于调试）')
    parser.add_argument('--sample_docs', action='store_true',
                        help='随机采样文档而非顺序截取（配合--max_docs使用）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机采样的种子（默认：42）')
    
    # 树结构复用参数
    parser.add_argument('--import_taxonomy', type=str, default=None,
                        help='导入已有的分类体系结构（JSON文件）')
    parser.add_argument('--export_taxonomy', type=str, default=None,
                        help='导出分类体系结构到文件（JSON格式）')
    parser.add_argument('--taxonomy_format', type=str, default='json', choices=['json', 'pickle'],
                        help='分类体系导入/导出格式（默认：json，pickle仅用于向后兼容）')
    
    args = parser.parse_args()
    
    # Region-based 与 interpretable 扩展互斥
    if args.use_region_based_expansion and args.use_interpretable_expansion:
        print("提示：region_based_expansion 与 interpretable_expansion 互斥，"
              "已自动禁用 interpretable_expansion")
        args.use_interpretable_expansion = False
    
    return args


if __name__ == "__main__":
    args = read_config()
    
    # 参数验证
    if args.refine and not args.resume:
        print("错误：--refine 必须配合 --resume 使用")
        sys.exit(1)
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        success = main(args)
        if success:
            print("\n程序执行成功！")
            sys.exit(0)
        else:
            print("\n程序执行失败！")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n用户中断程序执行（Ctrl+C）")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n正在清理资源并退出...")
        # 确保所有资源被释放
        import gc
        gc.collect()
