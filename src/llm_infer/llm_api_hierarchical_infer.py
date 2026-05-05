"""
基于大模型API的层级文档分类推理脚本

通过调用大模型API（如 Qwen-VL-Max）进行层级文档分类推理，
支持多种Prompt策略：DP, DL, DH, TMH, DH-CoT, Few-Shot 等。

核心功能：
1. 读取SFT格式的输入数据（包含instruction和images字段）
2. 通过统一API客户端调用大模型API进行推理
3. 支持并发批量推理
4. 支持断点续推（跳过已处理的样本）
5. 支持分片输出（用于并行处理）
6. 记录用量统计

Usage:
    python src/llm_infer/llm_api_hierarchical_infer.py \
        --input /path/to/test.json \
        --output /path/to/test_predicted.json \
        --model qwen_vl_max_latest \
        --max_workers 3 \
        --batch_size 10
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_infer.llm_api_client import LlmApiClient


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于大模型API的层级文档分类推理"
    )
    
    # 输入输出配置
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入数据文件路径（JSON格式）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出结果文件路径"
    )
    
    # 模型配置
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen-vl-max",
        help="模型名称（由API服务端识别）"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("BASE_URL", ""),
        help="LLM API基础地址（可通过环境变量 BASE_URL 配置）"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("API_KEY", ""),
        help="LLM API Key（可通过环境变量 API_KEY 配置）"
    )
    parser.add_argument(
        "--api_protocol",
        type=str,
        default=os.environ.get("API_PROTOCOL", "openai"),
        choices=["openai", "anthropic"],
        help="API协议类型"
    )
    
    # 并发配置
    parser.add_argument(
        "--max_workers",
        type=int,
        default=3,
        help="最大并发数（默认3）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="批处理大小（每批处理多少样本后保存一次）"
    )
    
    # 重试配置
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="单次请求最大重试次数"
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=2.0,
        help="重试间隔（秒）"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="单次请求超时时间（秒）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="单次生成最大token数"
    )
    
    # 字段配置
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="instruction",
        help="输入数据中的Prompt字段名"
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="images",
        help="输入数据中的图片字段名"
    )
    parser.add_argument(
        "--output_column",
        type=str,
        default="output",
        help="输入数据中的标签字段名（用于保留）"
    )
    parser.add_argument(
        "--predict_column",
        type=str,
        default="predict",
        help="输出数据中的预测结果字段名"
    )
    
    # 断点续推配置
    parser.add_argument(
        "--resume",
        action="store_true",
        help="是否启用断点续推（跳过已处理的样本）"
    )
    
    # 分片配置（用于并行推理）
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="分片ID（用于并行处理）"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="总分片数（用于并行处理）"
    )
    
    # 调试配置
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式（只处理少量样本）"
    )
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=10,
        help="调试模式下处理的样本数"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
    return parser.parse_args()


def load_input_data(input_path: str) -> List[Dict]:
    """加载输入数据"""
    logger.info(f"Loading input data from: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # 如果是字典格式，尝试获取数据列表
        if "data" in data:
            data = data["data"]
        else:
            raise ValueError("Input data is a dict but doesn't have 'data' key")
    
    logger.info(f"Loaded {len(data)} samples")
    return data


def load_existing_results(output_path: str) -> Dict[str, Dict]:
    """加载已存在的结果（用于断点续推）"""
    if not os.path.exists(output_path):
        return {}
    
    logger.info(f"Loading existing results from: {output_path}")
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        
        # 建立索引（按instruction或sample_id）
        result_map = {}
        for item in existing_data:
            # 优先使用 sample_id，其次使用 instruction
            key = item.get("sample_id") or item.get("instruction", "")
            if key and "predict" in item:
                result_map[key] = item
        
        logger.info(f"Found {len(result_map)} existing results")
        return result_map
    except Exception as e:
        logger.warning(f"Failed to load existing results: {e}")
        return {}


def save_results(
    results: List[Dict],
    output_path: str,
    stats: Optional[Dict] = None
):
    """保存推理结果"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(results)} results to: {output_path}")
    
    # 保存统计信息
    if stats:
        stats_path = output_path.replace(".json", "_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved stats to: {stats_path}")


def get_image_urls(item: Dict, image_column: str) -> Optional[List[str]]:
    """从数据项中提取图片URL列表"""
    images = item.get(image_column)
    
    if images is None:
        return None
    
    if isinstance(images, str):
        # 单个图片URL
        return [images] if images.strip() else None
    
    if isinstance(images, list):
        # 图片URL列表
        valid_urls = [url for url in images if isinstance(url, str) and url.strip()]
        return valid_urls if valid_urls else None
    
    return None


def process_batch(
    client: LlmApiClient,
    batch: List[Tuple[int, Dict]],
    prompt_column: str,
    image_column: str,
    output_column: str,
    predict_column: str,
    max_workers: int,
    verbose: bool = False,
) -> List[Dict]:
    """
    处理一批数据
    
    Args:
        client: LLM API客户端
        batch: (原始索引, 数据项) 的列表
        prompt_column: prompt字段名
        image_column: 图片字段名
        output_column: 标签字段名
        predict_column: 预测结果字段名
        max_workers: 最大并发数
        verbose: 是否详细输出
        
    Returns:
        处理后的结果列表
    """
    # 准备请求
    requests = []
    for idx, item in batch:
        prompt = item.get(prompt_column, "")
        images = get_image_urls(item, image_column)
        requests.append((prompt, images))
    
    # 批量调用
    responses = client.batch_chat_with_images(
        requests,
        max_workers=max_workers,
        show_progress=False,
    )
    
    # 构建结果
    results = []
    for (idx, item), response in zip(batch, responses):
        result = {
            prompt_column: item.get(prompt_column, ""),
            output_column: item.get(output_column, ""),
            predict_column: response
        }
        
        # 保留其他字段（如 sample_id, images 等）
        for key, value in item.items():
            if key not in [prompt_column, output_column]:
                result[key] = value
        
        results.append(result)
        
        if verbose:
            logger.debug(f"[{idx}] Predict: {response[:100]}...")
    
    return results


def run_inference(args):
    """执行推理主流程"""
    start_time = time.time()
    
    # 配置日志
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    logger.info("=" * 60)
    logger.info("LLM API Hierarchical Document Classification Inference")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Protocol: {args.api_protocol}")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max Workers: {args.max_workers}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info("=" * 60)

    if not args.base_url:
        raise ValueError("base_url is required. Use --base_url or env BASE_URL")
    if not args.api_key:
        raise ValueError("api_key is required. Use --api_key or env API_KEY")
    
    # 加载输入数据
    data = load_input_data(args.input)
    
    # 分片处理
    if args.shard_id is not None and args.num_shards is not None:
        shard_size = len(data) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size if args.shard_id < args.num_shards - 1 else len(data)
        data = data[start_idx:end_idx]
        
        # 修改输出路径
        base_path = args.output
        if not base_path.endswith(f"_{args.shard_id}"):
            args.output = f"{base_path}_{args.shard_id}"
        
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: processing samples [{start_idx}, {end_idx})")
    
    # 调试模式
    if args.debug:
        data = data[:args.debug_samples]
        logger.info(f"Debug mode: processing {len(data)} samples only")
    
    # 断点续推
    existing_results = {}
    if args.resume:
        existing_results = load_existing_results(args.output)
    
    # 过滤已处理的样本
    pending_data = []
    completed_results = []
    
    for idx, item in enumerate(data):
        key = item.get("sample_id") or item.get(args.prompt_column, "")
        if key in existing_results:
            completed_results.append(existing_results[key])
        else:
            pending_data.append((idx, item))
    
    if existing_results:
        logger.info(f"Skipping {len(completed_results)} already processed samples")
    
    if not pending_data:
        logger.info("All samples already processed!")
        return
    
    logger.info(f"Processing {len(pending_data)} samples...")
    
    # 初始化LLM API客户端
    logger.info("Initializing LLM API client...")
    client = LlmApiClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_protocol=args.api_protocol,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # 批量处理
    all_results = list(completed_results)
    total_batches = (len(pending_data) + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=len(pending_data), desc="Inference") as pbar:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(pending_data))
            batch = pending_data[batch_start:batch_end]
            
            # 处理当前批次
            batch_results = process_batch(
                client=client,
                batch=batch,
                prompt_column=args.prompt_column,
                image_column=args.image_column,
                output_column=args.output_column,
                predict_column=args.predict_column,
                max_workers=args.max_workers,
                verbose=args.verbose,
            )
            
            all_results.extend(batch_results)
            pbar.update(len(batch))
            
            # 每批保存一次（防止中断丢失）
            save_results(all_results, args.output)
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                stats = client.get_stats()
                logger.info(
                    f"Batch {batch_idx + 1}/{total_batches} | "
                    f"Success Rate: {stats['success_rate']} | "
                    f"Avg Latency: {stats['avg_latency_ms']} ms"
                )
    
    # 获取最终统计
    final_stats = client.get_stats()
    final_stats["total_samples"] = len(all_results)
    final_stats["inference_time_seconds"] = time.time() - start_time
    final_stats["timestamp"] = datetime.now().isoformat()
    
    # 保存最终结果
    save_results(all_results, args.output, final_stats)
    
    # 打印统计信息
    logger.info("=" * 60)
    logger.info("Inference Completed!")
    logger.info("=" * 60)
    client.print_stats()
    logger.info(f"Total Time: {final_stats['inference_time_seconds']:.2f} seconds")
    logger.info(f"Samples/Second: {len(all_results) / final_stats['inference_time_seconds']:.2f}")
    logger.info("=" * 60)


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
