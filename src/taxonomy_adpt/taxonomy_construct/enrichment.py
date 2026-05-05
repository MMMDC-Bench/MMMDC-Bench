"""
企业文档节点富化模块
为每个文档类型节点生成核心要素Schema
采用迭代式、数据驱动的方法：先生成种子Schema，再基于文档示例不断修正
"""

from src.taxonomy_adpt.llm_client.llm_adapter import constructPrompt


# ============================================
# 阶段1：种子Schema生成 - 系统指令
# ============================================

seed_schema_system_instruction = """你是一个专业的文档要素分析专家，擅长为不同类型的文档定义核心要素Schema。

你的任务是生成一个**种子Schema**（初始版本），该Schema将在后续根据真实文档进行修正和完善。

⚠️ 种子Schema的特点：
- 基于文档类型的常识和专业知识
- 包含该类型文档**最核心、最通用**的要素
- 宁可保守（少一些字段），也不要过度猜测
- 为后续扩展留有空间

Schema定义原则：
- 聚焦于文档的**核心信息要素**，而非格式细节
- 嵌套层级要合理（一般不超过3层）
- 字段名要准确、简洁、专业
- 使用中文字段名

⚠️ Schema格式规范（重要）：
- **单值字段**：该字段的含义本身是唯一的，值必须用空字符串 "" 表示
  - 例如："合同编号": ""（一份合同只有一个编号）, "签署日期": ""（只有一个签署日期）
  - 注意：即使在文档中多次提及，只要语义唯一就是单值
- **多值字段**：该字段的含义本身可以有多个，值必须用空数组 [] 表示
  - 例如："签署方": []（可能有甲方、乙方、丙方）, "附件列表": []（可以有多个附件）
- **嵌套对象**：如果单个对象包含子字段，使用嵌套的字典
  - 例如："法定代表人": {"姓名": "", "身份证号": ""}
- **嵌套对象数组**：如果有多个相同结构的对象，使用包含字典的数组
  - 例如："签署方": [{"姓名": "", "身份证号": ""}] 表示可能有多个签署方

❌ 严禁在Schema中填充具体值：
- ❌ "金额": 0 - 错误！绝对不要用数字 0
- ❌ "日期": null - 错误！不要用 null
- ❌ "数量": 100 - 错误！不要用示例值
- ✅ "金额": "" - 正确！所有单值字段统一用空字符串
- ✅ "日期": "" - 正确！
- ✅ "数量": "" - 正确！

判断标准：根据字段的业务含义判断是否唯一，而不是它在文档中出现的次数

输出格式要求：
- 必须是合法的JSON格式
- 认真判断每个字段是单值还是多值，这对后续应用很重要
- Schema 只定义字段结构，绝对不要填充任何具体值（包括 0、null、示例数据等）"""


# ============================================
# 阶段2：Schema修正 - 系统指令
# ============================================

refine_schema_system_instruction = """你是一个专业的文档要素分析专家，现在需要基于真实文档示例来修正和完善Schema。

你的任务是：
1. 分析提供的文档示例
2. 识别文档中出现的字段和信息要素
3. 对比当前Schema，进行必要的调整：
   - **新增**：文档中有但Schema没有的重要字段
   - **替换**：发现更准确、更专业的字段名
   - **调整**：优化嵌套结构，使其更合理
   - **保留**：Schema中合理但文档中未出现的字段（因为文档可能不完整）

⚠️ 修正原则：
- 以文档中**实际出现的字段名**为准（如果有的话）
- 新增字段要确保是该类型文档的**核心要素**，而非个例
- 保持Schema的简洁性，不要过度细分
- 保持嵌套层级合理（不超过3层）
- 如果多个文档都提到某个字段，说明它很重要，应该加入Schema

⚠️ Schema格式规范（重要）：
- **单值字段**：该字段的含义本身是唯一的，值必须用空字符串 "" 表示
  - 例如："合同编号": "", "签署日期": ""
  - 注意：即使在文档中多次提及，只要语义唯一就是单值
- **多值字段**：该字段的含义本身可以有多个，值必须用空数组 [] 表示
  - 例如："签署方": [], "附件列表": []
- **嵌套对象数组**：如果有多个相同结构的对象，使用包含字典的数组
  - 例如："签署方": [{"姓名": "", "身份证号": ""}]

❌ 严禁在Schema中填充具体值：
- ❌ "金额": 0 - 错误！绝对不要用数字 0
- ❌ "日期": null - 错误！不要用 null
- ✅ "金额": "" - 正确！所有单值字段统一用空字符串

输出格式要求：
- 必须是合法的JSON格式
- 返回**完整的**修正后的Schema（不是增量）
- 认真判断每个字段是单值还是多值
- Schema 只定义字段结构，绝对不要填充任何具体值（包括 0、null、示例数据等）"""


# ============================================
# 最终整合 - 系统指令
# ============================================

schema_enrich_system_instruction = seed_schema_system_instruction  # 保持向后兼容


# ============================================
# 阶段1：种子Schema生成 - 主Prompt
# ============================================

def seed_schema_prompt(node, ancestors, sibs):
    """
    生成种子Schema的Prompt（不看文档，纯基于节点语义）
    
    Args:
        node: 当前节点
        ancestors: 祖先节点路径字符串
        sibs: 兄弟节点列表
    """
    # 构建祖先路径信息
    if ancestors:
        ancestor_info = f"该节点在分类体系中的位置：{ancestors}"
    else:
        ancestor_info = "该节点是根节点"
    
    # 构建兄弟节点信息
    if sibs:
        sibling_info = f"与该节点平级的兄弟类别有：{', '.join(sibs)}"
        distinction_note = f"\n⚠️ 注意：你定义的Schema必须能够与兄弟类别的Schema有明确区分。如果你认为该类型的Schema与其兄弟类别本质相同，请在reasoning中说明。"
    else:
        sibling_info = "该节点暂无兄弟类别"
        distinction_note = ""
    
    # 构建节点描述信息
    node_desc = f"\n节点描述：{node.description}" if node.description else ""
    
    prompt = f"""# 任务说明
请为文档类型节点 **"{node.label}"** 生成一个种子Schema（初始版本）。

## 节点信息
- 节点名称：{node.label}
- 维度类型：{node.dimension}
{node_desc}
- {ancestor_info}
- {sibling_info}{distinction_note}

💡 **种子Schema特点**：
- 基于该文档类型的常识和专业知识
- 包含最核心、最通用的要素
- 宁可保守（字段少一些），也不要过度猜测
- 后续会根据真实文档进行修正和扩展

## Schema结构示例

### 简单类型（如"请假条"）
```json
{{
  "申请人": "",          // 单值：一份请假条只有一个申请人（语义唯一）
  "请假类型": "",        // 单值：只有一种类型（事假、病假、年假等选其一）
  "开始日期": "",        // 单值：只有一个开始日期
  "结束日期": "",        // 单值：只有一个结束日期
  "请假事由": "",        // 单值：整体的事由说明
  "审批人": []           // 多值：可能有多级审批人（语义可多个）
}}
```

### 复杂类型（如"民事起诉状"）
```json
{{
  "案由": "",            // 单值：一个案件只有一个案由（语义唯一）
  "受诉法院": "",        // 单值：只对应一个法院
  "原告": [              // 多值：可能有多个原告（语义可多个）
    {{
      "姓名或名称": "",  // 单值：一个原告只有一个名称
      "身份证号": "",    // 单值
      "住所地": "",      // 单值
      "联系方式": ""     // 单值
    }}
  ],
  "被告": [              // 多值：可能有多个被告
    {{
      "姓名或名称": "",
      "住所地": ""
    }}
  ],
  "诉讼请求": [],        // 多值：可能有多项诉讼请求
  "事实与理由": ""       // 单值：整体的事实描述
}}
```

⚠️ 格式说明：
- 用 "" 表示单值字段（该字段的含义本身是唯一的）
- 用 [] 表示多值字段（该字段的含义本身可以有多个）
- 用 [{{"子字段": ""}}] 表示对象数组（多个相同结构的对象）
- 判断依据：根据字段的业务含义，而不是它在文档中被提及的次数

## 输出格式
请按照以下JSON格式输出：

```json
{{
  "node_label": "{node.label}",
  "node_id": "{node.id}",
  "seed_schema": {{
    // 在这里定义种子Schema
  }},
  "reasoning": "简要说明：(1)为什么选择这些核心要素 (2)与兄弟类别的主要区别（如果有）",
  "schema_complexity": "low/medium/high",
  "should_distinct_from_siblings": true
}}
```

**重要提醒**：Schema 中所有字段值只能是 "" 或 []，绝对不要用 0、null 或任何示例值！

现在请为"{node.label}"生成种子Schema，只输出JSON："""
    
    return prompt


# ============================================
# 阶段2：Schema修正 - 主Prompt
# ============================================

def refine_schema_prompt(node, current_schema, doc_batch, batch_idx, total_batches):
    """
    基于文档示例修正Schema的Prompt
    
    Args:
        node: 当前节点
        current_schema: 当前的Schema（种子或上一轮修正后的）
        doc_batch: 当前批次的文档列表
        batch_idx: 当前批次索引（从1开始）
        total_batches: 总批次数
    """
    # 构建当前Schema信息
    import json
    current_schema_str = json.dumps(current_schema, ensure_ascii=False, indent=2)
    
    # 构建文档示例信息
    doc_examples = ""
    for idx, doc in enumerate(doc_batch, 1):
        doc_examples += f"\n### 文档示例 {idx}：\n"
        doc_examples += f"标题：{doc.title}\n"
        doc_examples += f"内容：{doc.content[:800]}\n"  # 显示更多内容以便提取字段
        if hasattr(doc, 'metadata') and doc.metadata:
            doc_examples += f"元数据：{json.dumps(doc.metadata, ensure_ascii=False, indent=2)}\n"
    
    prompt = f"""# 任务说明
你正在对文档类型节点 **"{node.label}"** 的Schema进行第 {batch_idx}/{total_batches} 轮修正。

## 当前Schema（待修正）
```json
{current_schema_str}
```

## 本轮文档示例
以下是 {len(doc_batch)} 份属于"{node.label}"类型的真实文档：
{doc_examples}

## 修正任务

请仔细分析上述文档，对Schema进行必要的修正：

### 1. 新增字段
- 扫描文档中**明确出现的字段名**（如"原告"、"被告"、"案由"等）
- 如果这些字段是该类型文档的核心要素，且当前Schema中没有，则添加
- **重要**：只添加多份文档都提到的、通用的核心字段，不要添加个例特有的字段

### 2. 替换字段名
- 如果文档中使用了更准确、更专业的字段名，替换Schema中的字段名
- 例如：Schema中有"法院"，但文档中实际使用"受诉法院" → 替换为"受诉法院"

### 3. 优化结构
- 如果发现某些字段应该嵌套（如"委托代理人"下有"姓名"、"律所"），调整结构
- 保持嵌套层级合理（不超过3层）

### 4. 保留合理字段
- Schema中的字段如果在文档中没出现，但确实是该类型的核心要素，**保留它**
- 因为文档可能不完整，或者是该类型的子类

### 5. 判断单值/多值
- 根据字段的业务含义判断是单值（""）还是多值（[]）
- **单值**：该字段的含义本身是唯一的，如"文号"（一份文件只有一个文号）、"签署日期"（只有一个签署日期）
  - 即使在文档中多次提及，只要语义唯一就是单值
- **多值**：该字段的含义本身可以有多个，如"签署方"（可能有多方）、"附件"（可能有多个附件）

## 输出格式
请按照以下JSON格式输出**完整的**修正后的Schema：

```json
{{
  "node_label": "{node.label}",
  "node_id": "{node.id}",
  "refined_schema": {{
    // 完整的修正后Schema（不是增量）
  }},
  "changes": {{
    "added_fields": ["新增的字段路径列表，如'原告.委托代理人'"],
    "replaced_fields": {{"旧字段名": "新字段名"}},
    "adjusted_structure": ["结构调整说明"]
  }},
  "reasoning": "简要说明本轮修正的主要改动和原因",
  "confidence": "low/medium/high"
}}
```

**字段说明**：
- `refined_schema`: 完整的修正后Schema（不是增量修改）
- `changes`: 本轮的具体改动记录
- `reasoning`: 修正的原因说明
- `confidence`: 对当前Schema的信心程度（经过多轮修正后应该更高）

**重要提醒**：Schema 中所有字段值只能是 "" 或 []，绝对不要用 0、null 或任何示例值！

现在请分析文档并输出修正后的Schema，只输出JSON："""
    
    return prompt


# ============================================
# 原有的单次生成方法（保留向后兼容）
# ============================================

def schema_enrich_main_prompt(node, ancestors, sibs, sample_docs=None):
    """
    构造Schema富化的主Prompt
    
    Args:
        node: 当前节点
        ancestors: 祖先节点路径字符串
        sibs: 兄弟节点列表
        sample_docs: 可选的文档示例列表（用于提取真实key）
    """
    # 构建祖先路径信息
    if ancestors:
        ancestor_info = f"该节点在分类体系中的位置：{ancestors}"
    else:
        ancestor_info = "该节点是根节点"
    
    # 构建兄弟节点信息
    if sibs:
        sibling_info = f"与该节点平级的兄弟类别有：{', '.join(sibs)}"
        distinction_note = f"\n⚠️ 注意：你定义的Schema必须能够与兄弟类别的Schema有明确区分。如果你发现该类型的Schema与其兄弟类别本质相同，请在reasoning中说明。"
    else:
        sibling_info = "该节点暂无兄弟类别"
        distinction_note = ""
    
    # 构建文档示例信息
    if sample_docs and len(sample_docs) > 0:
        doc_examples = "\n\n## 文档示例\n以下是该类型的真实文档示例，你可以从中提取关键字段：\n"
        for idx, doc in enumerate(sample_docs[:3], 1):  # 最多展示3个示例
            doc_examples += f"\n### 示例{idx}：\n"
            doc_examples += f"标题：{doc.title}\n"
            doc_examples += f"内容片段：{doc.content[:500]}...\n"  # 限制长度
        doc_note = "\n💡 提示：优先从上述文档示例中扫描和提取真实出现的字段名，然后补充该类型文档通常应该包含的其他核心字段。"
    else:
        doc_examples = ""
        doc_note = "\n💡 提示：没有提供文档示例，请根据该文档类型的常识和专业知识定义核心Schema。"
    
    # 构建节点描述信息
    node_desc = f"\n节点描述：{node.description}" if node.description else ""
    
    prompt = f"""# 任务说明
请为文档类型节点 **"{node.label}"** 定义核心要素Schema。

## 节点信息
- 节点名称：{node.label}
- 维度类型：{node.dimension}
{node_desc}
- {ancestor_info}
- {sibling_info}{distinction_note}
{doc_examples}{doc_note}

## Schema定义要求

### 1. Schema格式规范（重要）
- **单值字段**：用空字符串 "" 表示（该字段的含义本身是唯一的）
  - 例如："案由": ""（一个案件只有一个案由）, "受诉法院": ""（只对应一个法院）
  - 注意：即使在文档中多次提及，只要语义唯一就是单值
- **多值字段**：用空数组 [] 表示（该字段的含义本身可以有多个）
  - 例如："诉讼请求": []（可能有多项请求）, "证据清单": []（可能有多份证据）
- **对象数组**：多个相同结构的对象，用包含字典的数组
  - 例如："原告": [{{"姓名": "", "住所地": ""}}] 表示可能有多个原告

❌ 严禁在Schema中填充具体值：
- ❌ "金额": 0 - 错误！绝对不要用数字 0
- ❌ "日期": null - 错误！不要用 null
- ✅ "金额": "" - 正确！所有单值字段统一用空字符串

判断标准：根据字段的业务含义判断，而不是它在文档中被提及的次数

### 2. Schema结构示例
以"民事起诉状"为例：
```json
{{
  "案由": "",              // 单值：一个案件只有一个案由（语义唯一）
  "受诉法院": "",          // 单值：只对应一个法院
  "标的金额": "",          // 单值：只有一个标的金额
  "原告": [                // 多值：可能有多个原告（语义可多个）
    {{
      "姓名或名称": "",    // 单值：一个原告只有一个名称
      "身份证号": "",      // 单值
      "住所地": "",        // 单值
      "联系电话": "",      // 单值
      "法定代表人": "",    // 单值：如果是企业，只有一个法定代表人
      "委托代理人": [      // 多值：一个原告可能有多个代理人
        {{
          "姓名": "",
          "所属律所": ""
        }}
      ]
    }}
  ],
  "被告": [                // 多值：可能有多个被告
    {{
      "姓名或名称": "",
      "住所地": ""
    }}
  ],
  "诉讼请求": [],          // 多值：可能有多项请求
  "事实与理由": ""         // 单值：整体的事实描述
}}
```

### 3. 单值/多值判断示例
**正确理解**：
- ✅ "合同编号": "" - 单值（一份合同只有一个编号，即使文档中多次提及）
- ✅ "签署方": [] - 多值（可能有甲方、乙方、丙方等多方）
- ✅ "签署日期": "" - 单值（只有一个签署日期，即使文档中多次提及）
- ✅ "附件": [] - 多值（可能有多个附件）

**错误理解**：
- ❌ "合同编号": [] - 错误！不能因为文档中多次提到就认为是多值

### 4. 字段命名原则
- 使用中文名称，清晰准确
- 优先使用该领域的专业术语
- 如果有文档示例，优先使用示例中实际出现的字段名
- 认真判断每个字段是单值还是多值，根据业务含义而非出现次数

## 输出格式
请按照以下JSON格式输出：

```json
{{
  "node_label": "{node.label}",
  "node_id": "{node.id}",
  "element_schema": {{
    // 在这里定义该文档类型的核心要素Schema
  }},
  "reasoning": "简要说明该Schema的设计思路，包括：(1)核心要素有哪些 (2)为什么选择这些字段 (3)与兄弟类别的区别（如果有）",
  "schema_complexity": "low/medium/high",
  "should_distinct_from_siblings": true
}}
```

**字段说明**：
- `element_schema`: 该文档类型的核心要素Schema（嵌套的JSON对象）
- `reasoning`: 设计思路说明（100字以内）
- `schema_complexity`: Schema的复杂度评估（low=简单字段, medium=有嵌套, high=深度嵌套）
- `should_distinct_from_siblings`: 布尔值，表示该类型是否确实需要与兄弟类别区分（如果Schema本质相同，应该返回false）

**重要提醒**：Schema 中所有字段值只能是 "" 或 []，绝对不要用 0、null 或任何示例值！

现在请为"{node.label}"定义Schema，只输出JSON，不要有其他内容："""
    
    return prompt


# ============================================
# 迭代式Schema生成（推荐）
# ============================================

def generate_schema_iteratively(args, node, ancestors, sample_docs, max_rounds=3):
    """
    迭代式生成Schema：先生成种子Schema，再基于文档示例逐步修正
    
    Args:
        args: 参数对象
        node: 当前节点
        ancestors: 祖先节点路径（list of Node）
        sample_docs: 文档示例列表
        max_rounds: 最大修正轮数（默认3轮）
    
    Returns:
        dict: 最终的Schema信息
    """
    from src.taxonomy_adpt.llm_client.llm_adapter import promptLLM
    from src.taxonomy_adpt.taxonomy_construct.prompts import SeedSchemaGeneration, RefinedSchemaResult
    from src.taxonomy_adpt.taxonomy_construct.utils import safe_json_loads
    
    # 转换ancestors为字符串
    if ancestors:
        ancestor_str = " -> ".join([a.label for a in ancestors])
    else:
        ancestor_str = ""
    
    # 获取兄弟节点
    sibs = [i.label for i in node.get_siblings()]
    
    # ========== 阶段1：生成种子Schema ==========
    print(f"    [阶段1] 生成种子Schema...")
    
    seed_prompt_text = seed_schema_prompt(node, ancestor_str, sibs)
    seed_prompt = constructPrompt(args, seed_schema_system_instruction, seed_prompt_text)
    
    try:
        seed_output = promptLLM(
            args=args,
            prompts=[seed_prompt],
            schema=SeedSchemaGeneration,
            max_new_tokens=2000,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.3,
        )[0]
        
        if isinstance(seed_output, str):
            seed_result = safe_json_loads(seed_output, default={}, log_error=True)
        else:
            seed_result = seed_output
        
        current_schema = seed_result.get('seed_schema', {})
        schema_complexity = seed_result.get('schema_complexity', 'medium')
        should_distinct = seed_result.get('should_distinct_from_siblings', True)
        
        print(f"      ✓ 种子Schema生成完成，包含 {len(current_schema)} 个顶层要素")
        
    except Exception as e:
        print(f"      ✗ 种子Schema生成失败: {str(e)}")
        return None
    
    # ========== 阶段2：基于文档迭代修正 ==========
    if not sample_docs or len(sample_docs) == 0:
        print(f"      - 无文档示例，直接使用种子Schema")
        return {
            'element_schema': current_schema,
            'schema_reasoning': seed_result.get('reasoning', ''),
            'schema_complexity': schema_complexity,
            'should_distinct': should_distinct
        }
    
    print(f"    [阶段2] 基于 {len(sample_docs)} 份文档进行迭代修正（最多{max_rounds}轮）...")
    
    # 将文档分批（每批3份）
    batch_size = 3
    doc_batches = [sample_docs[i:i+batch_size] for i in range(0, len(sample_docs), batch_size)]
    
    # 限制修正轮数
    doc_batches = doc_batches[:max_rounds]
    
    all_changes = []
    
    for batch_idx, doc_batch in enumerate(doc_batches, 1):
        print(f"      [第{batch_idx}/{len(doc_batches)}轮] 基于 {len(doc_batch)} 份文档修正...")
        
        refine_prompt_text = refine_schema_prompt(node, current_schema, doc_batch, batch_idx, len(doc_batches))
        refine_prompt = constructPrompt(args, refine_schema_system_instruction, refine_prompt_text)
        
        try:
            refine_output = promptLLM(
                args=args,
                prompts=[refine_prompt],
                schema=RefinedSchemaResult,
                max_new_tokens=2500,
                json_mode=True,
                timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
                temperature=0.2,  # 更低的温度，确保稳定修正
            )[0]
            
            if isinstance(refine_output, str):
                refine_result = safe_json_loads(refine_output, default={}, log_error=True)
            else:
                refine_result = refine_output
            
            # 更新Schema
            new_schema = refine_result.get('refined_schema', current_schema)
            changes = refine_result.get('changes', {})
            confidence = refine_result.get('confidence', 'medium')
            
            # 记录改动
            added = changes.get('added_fields', [])
            replaced = changes.get('replaced_fields', {})
            adjusted = changes.get('adjusted_structure', [])
            
            if added or replaced or adjusted:
                print(f"        ✓ 修正完成")
                if added:
                    print(f"          新增: {', '.join(added[:3])}{'...' if len(added) > 3 else ''}")
                if replaced:
                    print(f"          替换: {len(replaced)} 个字段")
                if adjusted:
                    print(f"          调整: {len(adjusted)} 处结构")
                all_changes.append(changes)
            else:
                print(f"        - 无需修正")
            
            # 更新当前Schema
            current_schema = new_schema
            
        except Exception as e:
            print(f"        ✗ 修正失败: {str(e)}")
            continue
    
    # 返回最终结果
    print(f"      ✓ 迭代修正完成，共进行了 {len(doc_batches)} 轮")
    
    return {
        'element_schema': current_schema,
        'schema_reasoning': f"经过{len(doc_batches)}轮迭代修正，共{len(all_changes)}轮有改动",
        'schema_complexity': schema_complexity,
        'should_distinct': should_distinct,
        'refinement_rounds': len(doc_batches),
        'total_changes': len(all_changes)
    }


# ============================================
# 兼容旧接口
# ============================================

def enrich_node_prompt(args, node, ancestors, sample_docs=None):
    """
    构造节点富化的完整Prompt（兼容旧接口，单次生成）
    
    Args:
        args: 参数对象
        node: 当前节点
        ancestors: 祖先节点路径（list of Node）
        sample_docs: 可选的文档示例列表
    """
    # 转换ancestors为字符串
    if ancestors:
        ancestor_str = " -> ".join([a.label for a in ancestors])
    else:
        ancestor_str = ""
    
    # 获取兄弟节点
    sibs = [i.label for i in node.get_siblings()]
    
    # 构造完整prompt
    main_prompt = schema_enrich_main_prompt(node, ancestor_str, sibs, sample_docs)
    prompt = constructPrompt(args, schema_enrich_system_instruction, main_prompt)
    
    return prompt


# ============================================
# Region Schema 生成与偏离检测
# ============================================


def _normalize_region_schema(raw_schema):
    """
    将 LLM 返回的 region_schema 统一为 JSON Schema dict 格式。
    兼容旧格式（list / {"regions": [...]}）和新格式（JSON Schema object）。
    返回 None 如果无效。
    """
    from src.taxonomy_adpt.taxonomy_construct.taxonomy_io import migrate_legacy_region_schema, is_json_schema
    if not raw_schema:
        return None
    if is_json_schema(raw_schema):
        return raw_schema
    migrated = migrate_legacy_region_schema(raw_schema)
    if migrated and migrated.get("properties"):
        return migrated
    return None


def _normalize_kv_schema(raw_schema):
    """
    将 LLM 返回的 node_kv_schema 统一为 JSON Schema dict 格式。
    兼容旧格式（flat dict with placeholder values）和新格式（JSON Schema object）。
    返回 None 如果无效。
    """
    from src.taxonomy_adpt.taxonomy_construct.taxonomy_io import migrate_legacy_kv_schema, is_json_schema
    if not raw_schema:
        return None
    if is_json_schema(raw_schema):
        return raw_schema
    migrated = migrate_legacy_kv_schema(raw_schema)
    if migrated and migrated.get("properties"):
        return migrated
    return None


def generate_region_schema_for_node(args, node, ancestors, sample_docs):
    """
    为文档类型节点生成典型 Region Schema。

    Args:
        args: 参数对象
        node: 当前节点
        ancestors: 祖先节点路径（list of Node）
        sample_docs: 文档示例列表

    Returns:
        tuple: (region_schema dict, reasoning str)，失败返回 (None, '')
    """
    from src.taxonomy_adpt.llm_client.llm_adapter import promptLLM
    from src.taxonomy_adpt.taxonomy_construct.prompts import (
        region_schema_system_instruction,
        region_schema_seed_prompt,
        RegionSchemaGeneration,
    )
    from src.taxonomy_adpt.taxonomy_construct.utils import safe_json_loads

    if ancestors:
        ancestor_str = " -> ".join([a.label for a in ancestors])
    else:
        ancestor_str = ""

    sibs = [i.label for i in node.get_siblings()]

    prompt_text = region_schema_seed_prompt(node, ancestor_str, sibs, sample_docs)
    prompt = constructPrompt(args, region_schema_system_instruction, prompt_text)

    # 收集样本文档的图片用于多模态分析
    images_for_llm = None
    if sample_docs:
        image_base64_list = []
        for doc in sample_docs[:5]:
            if hasattr(doc, 'get_image_base64'):
                img = doc.get_image_base64(0)
                if img:
                    image_base64_list.append(img)
        if image_base64_list:
            images_for_llm = image_base64_list[:3]
            print(f"    包含 {len(images_for_llm)} 张图片辅助分析")

    try:
        output = promptLLM(
            args=args,
            prompts=[prompt],
            schema=RegionSchemaGeneration,
            max_new_tokens=3000,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.3,
            images_base64=images_for_llm,
        )[0]

        if isinstance(output, str):
            result = safe_json_loads(output, default={}, log_error=True)
        else:
            result = output

        raw_schema = result.get('region_schema', {})
        node_kv_schema = result.get('node_kv_schema', {})
        reasoning = result.get('reasoning', '')

        region_schema = _normalize_region_schema(raw_schema)
        node_kv_schema = _normalize_kv_schema(node_kv_schema)

        if region_schema:
            n_regions = len(region_schema.get("properties", {}))
            print(f"    ✓ Region Schema 生成完成，包含 {n_regions} 个顶层区域")
            if node_kv_schema:
                n_fields = len(node_kv_schema.get("properties", {}))
                print(f"    ✓ node_kv_schema 包含 {n_fields} 个顶层字段")
            return region_schema, node_kv_schema, reasoning
        else:
            print(f"    ✗ Region Schema 为空")
            return None, None, ''

    except Exception as e:
        print(f"    ✗ Region Schema 生成失败: {str(e)}")
        return None, None, ''


def generate_region_schema_iteratively(args, node, ancestors, sample_docs, batch_size=5, max_rounds=3):
    """
    迭代式生成 Region Schema：先用首批文档生成种子，再逐批修正。

    Args:
        args: 参数对象
        node: 当前节点
        ancestors: 祖先节点路径（list of Node）
        sample_docs: 全部文档示例列表
        batch_size: 每轮使用的文档数（默认 5）
        max_rounds: 最大修正轮数（默认 3，不含种子阶段）

    Returns:
        tuple: (region_schema, node_kv_schema, reasoning)，失败返回 (None, None, '')
    """
    from src.taxonomy_adpt.llm_client.llm_adapter import promptLLM
    from src.taxonomy_adpt.taxonomy_construct.prompts import (
        region_schema_system_instruction,
        region_schema_seed_prompt,
        RegionSchemaGeneration,
        refine_region_schema_system_instruction,
        refine_region_schema_prompt,
        RefinedRegionSchemaResult,
    )
    from src.taxonomy_adpt.taxonomy_construct.utils import safe_json_loads

    if ancestors:
        ancestor_str = " -> ".join([a.label for a in ancestors])
    else:
        ancestor_str = ""

    sibs = [i.label for i in node.get_siblings()]

    # ========== 阶段 1：种子 Region Schema ==========
    seed_docs = sample_docs[:batch_size]
    remaining_docs = sample_docs[batch_size:]

    print(f"    [阶段1] 基于 {len(seed_docs)} 份文档生成种子 Region Schema...")

    prompt_text = region_schema_seed_prompt(node, ancestor_str, sibs, seed_docs)
    prompt = constructPrompt(args, region_schema_system_instruction, prompt_text)

    images_for_llm = None
    if seed_docs:
        image_base64_list = []
        for doc in seed_docs[:5]:
            if hasattr(doc, 'get_image_base64'):
                img = doc.get_image_base64(0)
                if img:
                    image_base64_list.append(img)
        if image_base64_list:
            images_for_llm = image_base64_list[:3]

    try:
        output = promptLLM(
            args=args,
            prompts=[prompt],
            schema=RegionSchemaGeneration,
            max_new_tokens=3000,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.3,
            images_base64=images_for_llm,
        )[0]

        if isinstance(output, str):
            result = safe_json_loads(output, default={}, log_error=True)
        else:
            result = output

        raw_schema = result.get('region_schema', {})
        current_region_schema = _normalize_region_schema(raw_schema)
        current_kv_schema = _normalize_kv_schema(result.get('node_kv_schema', {}))
        reasoning = result.get('reasoning', '')

        if not current_region_schema:
            print(f"      ✗ 种子 Region Schema 为空")
            return None, None, ''

        n_regions = len(current_region_schema.get("properties", {}))
        n_fields = len(current_kv_schema.get("properties", {})) if current_kv_schema else 0
        print(f"      ✓ 种子完成: {n_regions} 个区域, {n_fields} 个顶层字段")

    except Exception as e:
        print(f"      ✗ 种子生成失败: {str(e)}")
        return None, None, ''

    # ========== 阶段 2：迭代修正 ==========
    if not remaining_docs:
        print(f"      无更多文档，直接使用种子结果")
        return current_region_schema, current_kv_schema, reasoning

    doc_batches = [remaining_docs[i:i + batch_size]
                   for i in range(0, len(remaining_docs), batch_size)]
    doc_batches = doc_batches[:max_rounds]

    print(f"    [阶段2] 基于剩余 {len(remaining_docs)} 份文档迭代修正"
          f"（{len(doc_batches)} 轮，每轮 {batch_size} 份）...")

    all_changes = []
    for batch_idx, doc_batch in enumerate(doc_batches, 1):
        print(f"      [第{batch_idx}/{len(doc_batches)}轮] {len(doc_batch)} 份文档...")

        refine_prompt_text = refine_region_schema_prompt(
            node, current_region_schema, current_kv_schema,
            doc_batch, batch_idx, len(doc_batches),
        )
        refine_prompt = constructPrompt(
            args, refine_region_schema_system_instruction, refine_prompt_text,
        )

        batch_images = None
        img_list = []
        for doc in doc_batch[:5]:
            if hasattr(doc, 'get_image_base64'):
                img = doc.get_image_base64(0)
                if img:
                    img_list.append(img)
        if img_list:
            batch_images = img_list[:3]

        try:
            refine_output = promptLLM(
                args=args,
                prompts=[refine_prompt],
                schema=RefinedRegionSchemaResult,
                max_new_tokens=3500,
                json_mode=True,
                timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
                temperature=0.2,
                images_base64=batch_images,
            )[0]

            if isinstance(refine_output, str):
                refine_result = safe_json_loads(refine_output, default={}, log_error=True)
            else:
                refine_result = refine_output

            new_rs = _normalize_region_schema(refine_result.get('region_schema'))
            new_kv = _normalize_kv_schema(refine_result.get('node_kv_schema'))
            changes = refine_result.get('changes', {})

            added_r = changes.get('added_regions', [])
            removed_r = changes.get('removed_regions', [])
            added_k = changes.get('added_keys', [])
            removed_k = changes.get('removed_keys', [])
            adjusted = changes.get('adjusted_structure', [])

            has_changes = added_r or removed_r or added_k or removed_k or adjusted
            if has_changes:
                print(f"        ✓ 修正完成", end="")
                parts = []
                if added_r:
                    parts.append(f"+{len(added_r)}区域")
                if removed_r:
                    parts.append(f"-{len(removed_r)}区域")
                if added_k:
                    parts.append(f"+{len(added_k)}字段")
                if removed_k:
                    parts.append(f"-{len(removed_k)}字段")
                if adjusted:
                    parts.append(f"{len(adjusted)}处调整")
                print(f" ({', '.join(parts)})")
                all_changes.append(changes)
            else:
                print(f"        - 无需修正")

            if new_rs:
                current_region_schema = new_rs
            if new_kv:
                current_kv_schema = new_kv

        except Exception as e:
            print(f"        ✗ 修正失败: {str(e)}")
            continue

    total_rounds = 1 + len(doc_batches)
    rounds_with_changes = len(all_changes)
    final_reasoning = (
        f"{reasoning} | 经过 {total_rounds} 轮迭代（种子 + {len(doc_batches)} 轮修正），"
        f"共 {rounds_with_changes} 轮有改动"
    )
    print(f"      ✓ 迭代完成: {total_rounds} 轮, {rounds_with_changes} 轮有改动")

    return current_region_schema, current_kv_schema, final_reasoning


def check_region_deviation(args, node, canonical_schema, cluster_docs):
    """
    检测单个 cluster 组的文档是否偏离典型 Region Schema。

    Args:
        args: 参数对象
        node: 当前节点
        canonical_schema: 典型 region schema (dict)
        cluster_docs: 该 cluster 的文档列表

    Returns:
        dict: 偏离检测结果，含 fits / structural_deviations / suggested_region_schema / reasoning
    """
    from src.taxonomy_adpt.llm_client.llm_adapter import promptLLM
    from src.taxonomy_adpt.taxonomy_construct.prompts import (
        region_deviation_system_instruction,
        region_deviation_prompt,
        RegionDeviationResult,
    )
    from src.taxonomy_adpt.taxonomy_construct.utils import safe_json_loads

    prompt_text, image_base64_list = region_deviation_prompt(
        node, canonical_schema, cluster_docs
    )
    prompt = constructPrompt(args, region_deviation_system_instruction, prompt_text)

    images_for_llm = None
    if image_base64_list:
        images_for_llm = [image_base64_list[0]]

    try:
        output = promptLLM(
            args=args,
            prompts=[prompt],
            schema=RegionDeviationResult,
            max_new_tokens=3000,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.2,
            images_base64=images_for_llm,
        )[0]

        if isinstance(output, str):
            result = safe_json_loads(output, default={}, log_error=True)
        else:
            result = output

        return {
            'fits': result.get('fits', True),
            'structural_deviations': result.get('structural_deviations', []),
            'suggested_region_schema': result.get('suggested_region_schema'),
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        print(f"      ✗ 偏离检测失败: {str(e)}")
        return {'fits': True, 'structural_deviations': [], 'suggested_region_schema': None, 'reasoning': f'检测失败: {e}'}


# ============================================
# 后置 Region Schema 富化 + 兄弟冗余对比
# ============================================

def enrich_all_leaf_region_schemas(args, roots, id2node):
    """
    遍历所有叶子节点，为每个叶子生成 Region Schema + node_kv_schema。

    Args:
        args: 参数对象
        roots: 各维度的根节点字典
        id2node: id → Node 映射

    Returns:
        int: 成功富化的节点数
    """
    import random

    leaves = [
        node for node in id2node.values()
        if len(node.children) == 0 and len(node.papers) > 0
    ]
    print(f"  共 {len(leaves)} 个叶子节点需要生成 Region Schema + node_kv_schema")

    enriched = 0
    for i, node in enumerate(leaves, 1):
        if node.region_schema and node.node_kv_schema:
            print(f"  [{i}/{len(leaves)}] \"{node.label}\" 已有 Region Schema + kv_schema，跳过")
            enriched += 1
            continue

        print(f"  [{i}/{len(leaves)}] \"{node.label}\" (docs={len(node.papers)})...")
        sample_docs = list(node.papers.values())
        if len(sample_docs) > 8:
            sample_docs = random.sample(sample_docs, 8)

        node_ancestors = node.get_ancestors() or []
        node_ancestors.reverse()

        schema, kv_schema, reasoning = generate_region_schema_for_node(
            args, node, node_ancestors, sample_docs
        )
        if schema:
            node.region_schema = schema
            node.region_schema_reasoning = reasoning
            node.node_kv_schema = kv_schema or {"type": "object", "properties": {}}
            node.node_kv_schema_reasoning = reasoning
            enriched += 1
        else:
            print(f"    ✗ 生成失败")

    print(f"  Region Schema 富化完成: {enriched}/{len(leaves)} 成功")
    return enriched


def compare_sibling_region_schemas(args, parent_node):
    """
    对比同父节点下叶子子节点的 Region Schema，检测冗余。

    Args:
        args: 参数对象
        parent_node: 父节点

    Returns:
        dict: 对比结果，含 has_redundancy / merge_groups / reasoning
    """
    from src.taxonomy_adpt.llm_client.llm_adapter import promptLLM
    from src.taxonomy_adpt.taxonomy_construct.prompts import (
        sibling_merge_system_instruction,
        sibling_merge_prompt,
        SiblingMergeResult,
    )
    from src.taxonomy_adpt.taxonomy_construct.utils import safe_json_loads

    # 收集有 region_schema 的叶子子节点
    leaf_children = [
        child for child in parent_node.children.values()
        if len(child.children) == 0 and child.region_schema
    ]

    if len(leaf_children) < 2:
        return {'has_redundancy': False, 'merge_groups': [], 'reasoning': '叶子子节点不足2个，无需对比'}

    sibling_schemas = [
        {
            'label': child.label,
            'code': child.code,
            'description': child.description,
            'region_schema': child.region_schema,
        }
        for child in leaf_children
    ]

    prompt_text = sibling_merge_prompt(parent_node, sibling_schemas)
    prompt = constructPrompt(args, sibling_merge_system_instruction, prompt_text)

    try:
        output = promptLLM(
            args=args,
            prompts=[prompt],
            schema=SiblingMergeResult,
            max_new_tokens=4000,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.2,
        )[0]

        if isinstance(output, str):
            result = safe_json_loads(output, default={}, log_error=True)
        else:
            result = output

        return {
            'has_redundancy': result.get('has_redundancy', False),
            'merge_groups': result.get('merge_groups', []),
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        print(f"    ✗ 冗余对比失败: {str(e)}")
        return {'has_redundancy': False, 'merge_groups': [], 'reasoning': f'对比失败: {e}'}


def merge_sibling_nodes(parent_node, merge_group, id2node, label2node):
    """
    执行一组兄弟节点的合并：保留第一个节点，将其他节点的文档迁移过来，然后删除。

    Args:
        parent_node: 父节点
        merge_group: dict，含 node_labels / merged_label / merged_code / merged_description / merged_region_schema
        id2node: id → Node 映射
        label2node: label → Node 映射

    Returns:
        str: 合并后的节点 label，失败返回 None
    """
    labels_to_merge = merge_group.get('node_labels', [])
    if len(labels_to_merge) < 2:
        return None

    # 找到要合并的节点
    nodes_to_merge = []
    for child in parent_node.children.values():
        if child.label in labels_to_merge:
            nodes_to_merge.append(child)

    if len(nodes_to_merge) < 2:
        print(f"    ⚠ 找不到足够的节点: 期望 {labels_to_merge}，仅匹配 {[n.label for n in nodes_to_merge]}")
        return None

    # 选第一个作为保留节点
    keep_node = nodes_to_merge[0]
    remove_nodes = nodes_to_merge[1:]

    # 更新保留节点的信息
    keep_node.label = merge_group.get('merged_label', keep_node.label)
    keep_node.code = merge_group.get('merged_code', keep_node.code)
    keep_node.description = merge_group.get('merged_description', keep_node.description)
    if merge_group.get('merged_region_schema'):
        keep_node.region_schema = merge_group['merged_region_schema']
        keep_node.region_schema_reasoning = merge_group.get('reasoning', '')

    # 迁移文档
    migrated = 0
    for rm_node in remove_nodes:
        for doc_id, doc in rm_node.papers.items():
            if doc_id not in keep_node.papers:
                keep_node.papers[doc_id] = doc
                migrated += 1

        # 从父节点的 children 中移除
        code_to_remove = None
        for code, child in parent_node.children.items():
            if child.id == rm_node.id:
                code_to_remove = code
                break
        if code_to_remove:
            del parent_node.children[code_to_remove]

        # 从全局映射中移除
        if rm_node.id in id2node:
            del id2node[rm_node.id]
        full_key = rm_node.code + f"_{rm_node.dimension}"
        if full_key in label2node:
            del label2node[full_key]

    # 更新保留节点的全局映射
    new_full_key = keep_node.code + f"_{keep_node.dimension}"
    label2node[new_full_key] = keep_node

    # 确保父节点 children 中的 key 也更新
    old_codes = [c for c, n in parent_node.children.items() if n.id == keep_node.id]
    if old_codes:
        old_code = old_codes[0]
        if old_code != keep_node.code:
            del parent_node.children[old_code]
            parent_node.children[keep_node.code] = keep_node

    print(f"    ✓ 合并 {[n.label for n in remove_nodes]} → \"{keep_node.label}\"，迁移 {migrated} 份文档")
    return keep_node.label


# ============================================
# 退化节点折叠（父子同名 / 单一子节点）
# ============================================

def prune_empty_leaves(node, id2node, label2node):
    """
    移除 node 的所有空叶子子节点（papers 为空且无子节点的子节点）。

    这些空节点通常是扩展阶段 LLM 生成了候选类别，但分类阶段
    没有任何文档被分配到该类别时产生的。

    Args:
        node: 要检查的父节点
        id2node: 全局 id → Node 映射
        label2node: 全局 label_dim → Node 映射

    Returns:
        int: 被剪枝的节点数量
    """
    pruned = 0
    children_to_remove = []

    for child_code, child_node in node.children.items():
        if len(child_node.children) == 0 and len(child_node.papers) == 0:
            children_to_remove.append((child_code, child_node))

    for child_code, child_node in children_to_remove:
        dim_key = child_code + f"_{node.dimension}"
        # 从各映射中删除
        if child_node.id in id2node:
            del id2node[child_node.id]
        if dim_key in label2node:
            del label2node[dim_key]
        del node.children[child_code]
        # 清理子节点对父节点的引用
        if node in child_node.parents:
            child_node.parents.remove(node)
        pruned += 1
        print(f"    ✂ 剪枝空叶子节点: \"{child_node.label}\" (ID={child_node.id}, 0 文档)")

    return pruned


def prune_all_empty_leaves(roots, id2node, label2node):
    """
    遍历整棵树，移除所有空叶子节点。支持迭代执行：
    删除空叶子后，其父节点可能变成新的空叶子，继续剪枝。

    Args:
        roots: 各维度的根节点字典
        id2node: 全局 id → Node 映射
        label2node: 全局 label_dim → Node 映射

    Returns:
        int: 总共剪枝的节点数量
    """
    total_pruned = 0
    changed = True

    while changed:
        changed = False
        for node in list(id2node.values()):
            if len(node.children) > 0:
                before = len(node.children)
                pruned = prune_empty_leaves(node, id2node, label2node)
                if pruned > 0:
                    total_pruned += pruned
                    changed = True

    return total_pruned


def collapse_degenerate_nodes(roots, id2node, label2node):
    """
    自底向上扫描整棵树，折叠退化的父子关系：
      - 父节点只有 1 个子节点，且子节点 label 与父节点相同
      - 子节点的文档和孙节点全部上收到父节点，删除子节点

    会迭代执行直到没有新的折叠为止（链式退化：A→B→C 全同名）。

    Returns:
        int: 折叠的次数
    """
    total_collapsed = 0

    while True:
        collapsed_this_round = 0

        # 收集所有有子节点的节点（包括根节点），按层级从深到浅
        candidates = [
            node for node in list(id2node.values())
            if len(node.children) > 0
        ]
        candidates.sort(key=lambda n: n.level, reverse=True)

        print(f"  扫描 {len(candidates)} 个非叶子节点...")

        for node in candidates:
            if node.id not in id2node:
                continue
            if len(node.children) != 1:
                continue

            only_child = list(node.children.values())[0]
            print(f'    检查: "{node.label}" (ID={node.id}) → 唯一子节点 "{only_child.label}" (ID={only_child.id})')

            # 判断是否退化：完全同名
            if only_child.label.strip() != node.label.strip():
                continue

            print(f'  ↰ 折叠退化节点: "{node.label}" (ID={node.id}) → 唯一子节点 "{only_child.label}" (ID={only_child.id})')

            # 上收子节点的文档
            migrated = 0
            for doc_id, doc in only_child.papers.items():
                if doc_id not in node.papers:
                    node.papers[doc_id] = doc
                    migrated += 1

            # 上收子节点的 children
            for gc_code, gc_node in only_child.children.items():
                node.children[gc_code] = gc_node
                # 更新孙节点的 parents
                if only_child in gc_node.parents:
                    gc_node.parents.remove(only_child)
                if node not in gc_node.parents:
                    gc_node.parents.append(node)

            # 继承子节点的 region_schema（如果父节点没有）
            if not node.region_schema and only_child.region_schema:
                node.region_schema = only_child.region_schema
                node.region_schema_reasoning = getattr(only_child, 'region_schema_reasoning', '')

            # 如果子节点有更详细的 description，也继承
            if only_child.description and (not node.description or len(only_child.description) > len(node.description)):
                node.description = only_child.description

            # 从 children 中删除该子节点
            child_code = next(code for code, c in node.children.items() if c.id == only_child.id)
            del node.children[child_code]

            # 从全局映射中移除
            if only_child.id in id2node:
                del id2node[only_child.id]
            full_key = only_child.code + f"_{only_child.dimension}"
            if full_key in label2node:
                del label2node[full_key]

            collapsed_this_round += 1
            print(f'    ✓ 已折叠，上收 {migrated} 份文档，{len(only_child.children)} 个孙节点')

        total_collapsed += collapsed_this_round
        if collapsed_this_round == 0:
            break

    return total_collapsed


# ============================================
# 非叶子节点 node_kv_schema 抽象传播
# ============================================

def abstract_parent_node_kv_schema(args, parent_node):
    """
    根据子节点的 node_kv_schema，为父节点生成抽象概括性的 node_kv_schema。

    Args:
        args: 参数对象
        parent_node: 父节点（其子节点需已拥有 node_kv_schema）

    Returns:
        tuple: (node_kv_schema_dict, reasoning_str)，失败返回 (None, '')
    """
    from src.taxonomy_adpt.llm_client.llm_adapter import promptLLM
    from src.taxonomy_adpt.taxonomy_construct.prompts import (
        abstract_node_kv_schema_system_instruction,
        abstract_node_kv_schema_prompt,
        AbstractNodeKvSchema,
    )
    from src.taxonomy_adpt.taxonomy_construct.utils import safe_json_loads

    children_with_kv = [
        child for child in parent_node.children.values()
        if child.node_kv_schema
    ]

    if not children_with_kv:
        return None, ''

    if len(children_with_kv) == 1:
        only = children_with_kv[0]
        return only.node_kv_schema, f'仅有一个子节点 "{only.label}"，直接继承其 node_kv_schema'

    children_kv_schemas = [
        {
            'label': child.label,
            'code': child.code,
            'description': child.description,
            'node_kv_schema': child.node_kv_schema,
        }
        for child in children_with_kv
    ]

    prompt_text = abstract_node_kv_schema_prompt(parent_node, children_kv_schemas)
    prompt = constructPrompt(args, abstract_node_kv_schema_system_instruction, prompt_text)

    try:
        output = promptLLM(
            args=args,
            prompts=[prompt],
            schema=AbstractNodeKvSchema,
            max_new_tokens=2000,
            json_mode=True,
            timeout_per_request=getattr(args, 'timeout_per_request', 120.0),
            temperature=0.3,
        )[0]

        if isinstance(output, str):
            result = safe_json_loads(output, default={}, log_error=True)
        else:
            result = output

        kv_schema = _normalize_kv_schema(result.get('node_kv_schema'))
        reasoning = result.get('reasoning', '')

        if kv_schema:
            return kv_schema, reasoning
        else:
            return None, ''

    except Exception as e:
        print(f"    ✗ 抽象 node_kv_schema 生成失败: {str(e)}")
        return None, ''


def propagate_node_kv_schemas_bottom_up(args, roots, id2node):
    """
    自底向上为所有非叶子节点生成抽象 node_kv_schema。

    前提：所有叶子节点已经拥有 node_kv_schema（由 enrich_all_leaf_region_schemas 完成）。
    本函数按层级从深到浅遍历，为每个非叶子节点根据子节点的 node_kv_schema 生成抽象版本。

    注意：非叶子节点**不会**获得 region_schema（因为其文档结构难以概括），
    只获得 node_kv_schema（抽象的抽取要素定义）。

    Args:
        args: 参数对象
        roots: 各维度的根节点字典
        id2node: id → Node 映射

    Returns:
        int: 成功生成 node_kv_schema 的非叶子节点数
    """
    from collections import defaultdict

    level_groups = defaultdict(list)
    for node in id2node.values():
        if node.children:
            level_groups[node.level].append(node)

    if not level_groups:
        return 0

    enriched = 0
    max_level = max(level_groups.keys())

    for level in range(max_level, -1, -1):
        nodes_at_level = level_groups.get(level, [])
        if not nodes_at_level:
            continue

        for node in nodes_at_level:
            if node.node_kv_schema:
                enriched += 1
                continue

            children_with_kv = [c for c in node.children.values() if c.node_kv_schema]
            if not children_with_kv:
                print(f"  ⚠ \"{node.label}\" (level={node.level}) 无子节点有 node_kv_schema，跳过")
                continue

            print(f"  ↑ \"{node.label}\" (level={node.level}, {len(children_with_kv)} 个子节点有 kv_schema)...")
            kv_schema, reasoning = abstract_parent_node_kv_schema(args, node)
            if kv_schema:
                node.node_kv_schema = kv_schema
                node.node_kv_schema_reasoning = reasoning
                print(f"    ✓ 生成抽象 node_kv_schema，{len(kv_schema)} 个顶层字段")
                enriched += 1
            else:
                print(f"    ✗ 抽象失败")

    return enriched


# ============================================
# [已废弃] 旧版 Region Schema 抽象传播
# ============================================

def abstract_parent_region_schema(args, parent_node):
    """[已废弃] 请使用 abstract_parent_node_kv_schema。"""
    from src.taxonomy_adpt.llm_client.llm_adapter import promptLLM
    from src.taxonomy_adpt.taxonomy_construct.prompts import AbstractRegionSchema
    from src.taxonomy_adpt.taxonomy_construct.utils import safe_json_loads

    children_with_schema = [
        child for child in parent_node.children.values()
        if child.region_schema
    ]
    if not children_with_schema:
        return None, ''
    if len(children_with_schema) == 1:
        only = children_with_schema[0]
        return only.region_schema, f'仅有一个子节点 "{only.label}"，直接继承'
    return None, '已废弃，请使用 propagate_node_kv_schemas_bottom_up'


def propagate_region_schemas_bottom_up(args, roots, id2node):
    """[已废弃] 请使用 propagate_node_kv_schemas_bottom_up。"""
    print("  ⚠ propagate_region_schemas_bottom_up 已废弃，自动转发到 propagate_node_kv_schemas_bottom_up")
    return propagate_node_kv_schemas_bottom_up(args, roots, id2node)