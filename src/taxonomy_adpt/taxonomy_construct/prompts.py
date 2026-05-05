"""
企业文档树的Prompt定义
适配中文场景和企业文档特征
"""

from pydantic import BaseModel, conset, conlist, StringConstraints, Field
from typing_extensions import Annotated
from typing import Dict, List, Optional

# ============================================
# Schema定义
# ============================================

class NodeSchema(BaseModel):
    code: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r'^[a-z0-9_]+$')]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]

class NodeListSchema(BaseModel):
    root_topic: Dict[str, NodeSchema]

class WidthExpansionSchema(BaseModel):
    new_subtopic_label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]

class DepthExpansionSchema(BaseModel):
    new_subtopic_label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]

class WidthClusterSchema(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True)]
    code: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r'^[a-z0-9_]+$')]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]
    covered_doc_topics: conlist(str, min_length=1, max_length=20)
    core_features: Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)] = ""  # 新增，默认为空

class WidthClusterListSchema(BaseModel):
    new_cluster_topics: conlist(WidthClusterSchema, min_length=1, max_length=10)

class DepthClusterSchema(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]
    code: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r'^[a-z0-9_]+$')]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]
    covered_doc_topics: conlist(str, min_length=1, max_length=20)
    core_features: Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)] = ""  # 新增，默认为空

class DepthClusterListSchema(BaseModel):
    new_cluster_topics: conlist(DepthClusterSchema, min_length=1, max_length=10)

# 组级别伪标签生成的Schema
class GroupPseudoLabelSchema(BaseModel):
    """组级别伪标签生成的返回格式"""
    subtopic_labels: conlist(str, min_length=1, max_length=5)  # 1-5个标签
    reasoning: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] = ""  # 生成原因

# 深度扩展评估的Schema
class DepthExpansionEvaluationSchema(BaseModel):
    """深度扩展评估的返回格式"""
    should_expand: bool  # 是否应该进行深度扩展
    reasoning: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] = ""  # 判断原因
    cluster_cohesion: Annotated[str, StringConstraints(strip_whitespace=True)] = ""  # 聚类凝聚度评估：high/medium/low
    has_clear_types: bool = False  # 是否有明确的文档类型特征

class ClassifySchema(BaseModel):
    doc_id: Annotated[int, Field(strict=True, gt=-1)]
    class_options: conset(int, min_length=1, max_length=100)
    class_label: int  # 改为单标签：每个文档只能属于一个类别

class EnrichSchema(BaseModel):
    node_to_enrich: Annotated[str, StringConstraints(strip_whitespace=True)]
    id: Annotated[str, StringConstraints(strip_whitespace=True)]
    commonsense_key_phrases: conset(str, min_length=20, max_length=50)
    commonsense_sentences: conset(str, min_length=10, max_length=50)

# 新增：企业文档Schema富化模式
class ElementSchemaEnrichment(BaseModel):
    """文档类型节点的要素Schema富化结果（单次生成模式）"""
    node_label: str
    node_id: str
    element_schema: dict  # 嵌套的Schema结构
    reasoning: str  # Schema设计思路
    schema_complexity: str  # low/medium/high
    should_distinct_from_siblings: bool  # 是否确实需要与兄弟节点区分

# 新增：种子Schema生成
class SeedSchemaGeneration(BaseModel):
    """种子Schema生成结果（迭代式方法的第一阶段）"""
    node_label: str
    node_id: str
    seed_schema: dict  # 初始的种子Schema
    reasoning: str  # 种子Schema的设计思路
    schema_complexity: str  # low/medium/high
    should_distinct_from_siblings: bool  # 是否确实需要与兄弟节点区分

# 新增：Schema修正结果
class RefinedSchemaResult(BaseModel):
    """基于文档修正Schema的结果（迭代式方法的修正阶段）"""
    node_label: str
    node_id: str
    refined_schema: dict  # 修正后的完整Schema
    changes: dict  # 本轮的改动记录：{"added_fields": [...], "replaced_fields": {...}, "adjusted_structure": [...]}
    reasoning: str  # 本轮修正的原因说明
    confidence: str  # 对当前Schema的信心程度：low/medium/high

# ============================================
# 企业文档维度定义(中文)
# ============================================

dimension_definitions = {
    'doc_type': """文档类型维度：定义文档本身是什么，关注文档的内在属性和本质特征，与使用场景无关。
    
    ⚠️ 核心原则：
    1. 关注文档的内容本质，而非呈现形式（表单/文本/图片/PDF等都只是载体）
    2. 同一内容本质的文档，无论以何种形式呈现，都应归为同一类型
    3. 例如：存档证明无论是表单形式还是文本形式，都是"存档证明"这一文档类型
    
    ⚠️ 多页文档和完整性原则：
    1. 不要根据文档的页数、完整程度来细分类别（如"第1页入党申请书"、"第2页入党申请书"）
    2. 多页文档应统一归为同一类型（如"入党申请书"），不论有多少页
    3. 不完整但有明显特征的文档仍归入对应类型（如只有部分页的入党申请书，仍是"入党申请书"）
    4. 判断标准：如果能从文档内容判断出它的原始类型，就不是文档组件
    
    ⚠️ 文档组件的定义标准：
    - 文档组件是指**无法从组件本身还原或判断出原始文档类型**的元素
    - 例如：单独的印章、签名、Logo、图表、表格片段等
    - 反例：不完整的入党申请书（即使只有部分内容，但有明显格式特征）不是文档组件，是"入党申请书"
    
    包括：证明凭证类（身份证、营业执照、发票、各类证明等）、合约协议类（合同、协议等）、规范标准类（规格书、标准、流程等）、报告分析类（财务报表、研究报告等）、方案计划类（项目方案、工作计划等）、制度政策类（公司制度、管理办法等）、申请记录类（各类申请书、登记记录等）、沟通文档类（邮件、会议纪要等）、说明指导类（操作手册、培训资料等）、营销宣传类（宣传册、演示文稿等）、参考资料类（行业资讯、外部资料等）、文档组件类（印章、签名、图片素材、表格数据、图表、Logo、二维码等非完整独立文档的元素和组件）。
    
    ⚠️ 注意：不要将"表单"、"文本文档"、"图片"等作为文档类型，这些是呈现形式而非内容本质。
    分类原则：同一份文档（如身份证）不论在何种业务场景使用，其文档类型保持不变。完整的独立文档归入对应类别，文档的组成部分或单独的元素归入文档组件类。""",
    
    'topic': """主题维度：定义文档用于什么业务场景、属于哪个业务领域。关注文档的使用场景和业务归属。
    包括：职能领域（财务、人力资源、法务、市场营销、研发、运营、客服等）、业务流程（采购、销售、招聘、培训、项目管理等）、业务主题（预算管理、合规审查、产品设计、客户服务等）。
    分类原则：同一份文档（如身份证）在不同业务场景使用时，可能属于不同的主题（财务场景 vs 人事场景）。""",
        
}

# 节点维度定义(用于生成子节点)
node_dimension_definitions = {
    'doc_type': """按照文档的本质属性和内在特征进行分类，而非使用场景或呈现形式。
    
    ⚠️ 核心原则：
    1. 关注"文档的内容本质是什么"，而不是"文档以什么形式呈现"
    2. 不要以呈现形式（表单、文本、图片、PDF等）作为分类依据
    3. 同一内容本质的文档，无论呈现形式如何，都归为同一类
    4. 不要根据页数、完整程度细分（多页文档统一归类，不完整但可识别的文档归入原类型）
    
    ⚠️ 文档组件识别标准：
    - 只有当**无法从内容判断原始文档类型**时，才归入文档组件类
    - 例如：单独的印章、签名无法判断来自哪种文档 → 文档组件
    - 反例：不完整的入党申请书有明显格式 → 仍是入党申请书，不是文档组件
    
    分类方向包括：证明凭证类、合约协议类、规范标准类、报告分析类、方案计划类、制度政策类、申请记录类、沟通文档类、说明指导类、营销宣传类、参考资料类、文档组件类等。
    
    核心原则：关注"文档本身是什么"，同一文档无论在何处使用其类型都不变（如身份证永远是身份证明类文档）。
    特别说明：文档组件类用于归类非完整独立文档的元素，如印章、签名、独立图片、表格、图表、Logo、二维码等文档片段或组成元素。""",
    
    'topic': """按照文档的使用场景和业务归属进行分类。
    分类方向包括：职能领域（财务、人力、法务、市场、研发、运营等）、业务流程（采购、销售、招聘、培训、项目管理等）、业务主题（预算、合规、产品、客户等）。
    核心原则：关注"文档用于什么场景"，同一文档可能在不同主题下使用（如身份证在财务审计时属于财务主题，在员工入职时属于人力主题）。""",
    
}

# ============================================
# 文档维度分类Prompt
# ============================================

type_cls_system_instruction = """你是一个专业的企业文档多维度分类助手,帮助识别文档所属的维度类型。文档可能属于一个或多个维度。

文档维度定义(维度:说明):

1. 文档类型(doc_type): 定义文档的形态和格式特征。所有文档都应该归属于某种文档类型,除非明确不属于任何已知类型。
2. 主题(topic): 定义文档所属的业务领域和主题内容。包括部门职能、业务流程和具体主题等。
"""

class TypeClsSchema(BaseModel):
    doc_type: bool
    topic: bool

def type_cls_main_prompt(document):
    """生成文档维度分类的主prompt"""
    return f"""根据以下文档的标题和内容,判断该文档属于哪些维度。请以JSON格式输出结果。

文档标题: {document.title}
文档内容: {document.get_summary(1000)}

你的输出应该是以下JSON格式:
{{
  "doc_type": true,
  "topic": <如果文档有明确的业务领域或主题内容则返回true,否则返回false>
}}
"""

# ============================================
# 多维度分类体系初始化Prompt
# ============================================

def multi_dim_prompt(node):
    """生成多维度分类体系的初始化prompt"""
    topic = node.label
    ancestors = ", ".join([ancestor.label for ancestor in node.get_ancestors()])
    
    system_instruction = f'''你是一个专业的企业文档分类体系构建助手,为主题"{topic}"{"" if ancestors == "" else "(隶属于: " + ancestors + ")"}构建{node.dimension}维度的分类体系。

我们定义{node.dimension}维度如下:
{dimension_definitions[node.dimension]}

请记住,企业文档将被映射到你构建的分类体系节点上。
'''
    
    # 根据维度调整生成数量
    max_categories = 12 if node.dimension == 'doc_type' else 5
    
    main_prompt = f'''你的根主题是: {topic}

子类别是指更精细地划分和组织相关文档的具体分类。请为"{topic}"生成{max_categories}个左右属于{node.dimension}维度的子类别,并为每个子类别生成一句话的描述。

确保每个子类别都是独特的,且适用于{node.dimension}维度下的"{topic}"{"和"+ancestors if ancestors else ""}。
'''
    
    if 'domain' in node.dimension or 'business' in node.dimension:
        main_prompt += f'\n记住,{node.dimension}维度是指文档可以应用的实际业务领域类别(例如,财务部门、人力资源部门等)。'
    
    json_output_format = f'''请仅以以下JSON格式输出你的分类体系,将每个标签名称替换为正确的子类别标签名称:

{{
  "root_topic": {{
    "<第一个子类别的中文名称>": {{
      "code": "<该子类别名称的英文翻译,使用小写字母和下划线连接,如:财务报告->financial_report, 人力资源->human_resources>",
      "description": "<生成该子类别的字符串描述>"
    }},
    ...,
    "<第k个子类别的中文名称>": {{
      "code": "<该子类别名称的英文翻译>",
      "description": "<生成子类别k的描述>"
    }}
  }}
}}

注意:
- 标签名称(key)使用中文
- code字段是该中文名称的英文翻译,使用小写字母、数字和下划线
- code应该是准确、专业的英文翻译,而不是拼音
- 确保每个code都是唯一的
- 示例: "财务报告" -> "financial_report", "合同管理" -> "contract_management"
'''
    
    return system_instruction, main_prompt, json_output_format

# ============================================
# 宽度扩展Prompt(发现同级新类别)
# ============================================

width_system_instruction = """你是一个执行分类体系宽度扩展的助手。宽度扩展是指在分类体系中增加更多的兄弟节点(共享同一父节点的类别),以捕获更广泛的概念范围。

你会收到一个父节点下的现有兄弟节点列表,以及一份文档的标题和内容。请识别该文档属于父节点下的哪个子主题,且该子主题与现有兄弟节点处于相同的具体程度。

"相同的具体程度"是指:这些主题在相同的抽象层次上,不存在包含关系,它们应该是分类体系中的兄弟节点。
"""

def width_main_prompt(document, node, ancestors, nl='\n'):
    """生成宽度扩展的主prompt"""
    return f"""
<输入>
<父节点>
{node.label}
</父节点>

<父节点描述>
{node.label}是一种{node.dimension}类型: {node.description}
</父节点描述>

<维度定义>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</维度定义>

<父节点路径>
{ancestors}
</父节点路径>

<文档标题>
{document.title}
</文档标题>

<文档内容>
{document.get_summary(1500)}
</文档内容>

<现有兄弟节点>
{nl.join([f"{c_label}:{nl}{nl}{c_label}的描述: {c.description}{nl}" for c_label, c in node.get_children().items()])}
</现有兄弟节点>

</输入>

根据输入的文档标题和内容,识别该文档属于父节点"{node.label}"下的哪个{node.dimension}类别标签,且该标签应该是现有兄弟节点的兄弟主题。换句话说,回答问题:该文档讨论的是哪种类型的{node.label} {node.dimension}?

你的输出应该是以下JSON格式:
{{
  "new_subtopic_label": <值类型为字符串; 字符串是一个新的主题标签(一种{node.dimension}类型),是该文档的真实主要主题,与现有兄弟节点中的其他类别标签处于相同的深度/具体程度>
}}
"""

width_cluster_system_instruction = """你是一个执行分类体系宽度扩展的聚类助手。宽度扩展是指在分类体系中增加更多的兄弟节点,以捕获更广泛的概念范围。

你需要根据讨论父节点的文档所涵盖的子主题,选择新的兄弟主题聚类。你的工作是从输入的文档主题集合中识别独特的聚类。对于你识别的每个聚类,必须提供聚类名称(格式类似于文档主题)作为其键,聚类名称的一句话描述,以及该聚类涵盖的所有输入文档主题列表。

你的新主题聚类应该是现有兄弟节点的兄弟主题,但要有明显区别。确保每个兄弟节点都具有相同的粒度/具体程度。还要确保你的每个新兄弟主题聚类都是独特的;它们不应该已经存在于现有节点集合中(existing_nodes)。
"""

def width_cluster_main_prompt(options, node, ancestors, all_node_labels, nl='\n'):
    """生成宽度扩展聚类的主prompt"""
    return f"""
<输入>
<父节点>
{node.label}
</父节点>

<父节点描述>
{node.label}是一种{node.dimension}类型: {node.description}
</父节点描述>

<维度定义>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</维度定义>

<父节点路径>
{ancestors}
</父节点路径>

你的新聚类主题不应该是以下任何一个:
<现有节点>
{all_node_labels}
</现有节点>

<现有兄弟节点>
{nl.join([f"{c_label}:{nl}{nl}{c_label}的描述: {c.description}{nl}" for c_label, c in node.get_children().items()])}
</现有兄弟节点>

<文档主题>
以下是文档主题字典,其中每个键是候选节点标签,值是映射到该候选节点的文档数量:
候选节点标签:\n{str(options)}
</文档主题>

</输入>

在父节点主题"{node.label}"下,哪些主要的{node.dimension}子主题聚类最能涵盖上述<文档主题>?

⚠️ 核心要求 - 以文档语义核心要素为锚点进行分类：

**核心原则：只有在核心要素有显著差异时才创建不同的兄弟类别**

1. **核心要素定义**：
   - ✅ 核心要素：文档必须包含的关键信息字段、核心功能、特定结构要求、涉及的特定主体或对象等本质性特征
   - 示例：介绍信需要"介绍人"、劳动合同需要"劳动关系条款"、发票需要"税务信息"
   - ❌ 非核心要素：具体格式、细微措辞、特定人群、具体场景、版面布局、页数等细节性差异
   
2. **兄弟类别创建标准**：
   - 必须判断：该兄弟类别是否有其他兄弟类别无法包含的独特核心要素？
   - 与现有兄弟节点对比，确保新增类别确实有独特的核心要素
   - 如果只是细节差异，核心要素相同，应合并或归入已有类别
   
3. **合并相似类别**：
   - 如果多个候选主题的核心要素相同或高度相似，必须合并为一个类别
   - 判断依据：去除非核心的修饰词（如具体组织名、具体人群、具体场景）后，是否为同一类型？
   
4. **最少文档数要求**：
   - 每个聚类应至少包含3个以上的文档主题（或总计文档数>=5）
   - 例外：如果某类文档有非常独特的核心要素，即使数量少也可以单独成类
   
5. **排除非实质性区分依据**：
   - ❌ 呈现形式（表单vs文本vs图片）
   - ❌ 页数和完整性（多页文档统一归类，不完整但可识别的归入原类型）
   - ❌ 版面布局、字体样式等视觉特征
   - ❌ 具体人名、日期、地点等实例化信息

这些应该是不重叠的主题聚类,最好地代表和划分所有文档主题(最大化映射到每个聚类的文档数量)。它们都应该是现有兄弟节点的兄弟(相同深度/具体程度)。你建议的每个新聚类主题都应该是父节点"{node.label}"下更具体的子主题,并且是{node.dimension}类型。然而,它们都应该是同等独特的(非重复的),任何单个文档都不应该轻易地同时属于两个聚类。

你的输出应该是以下JSON格式,最少一个子主题聚类,最多五个:
{{
  "new_cluster_topics": [
    {{
      "label": <字符串,{node.dimension}子标签中文名称,与现有兄弟节点中的其他主题处于相同的深度/具体程度>,
      "code": <字符串,label的英文翻译,使用小写字母和下划线连接>,
      "description": <字符串,{node.dimension}子主题的一句话描述>,
      "covered_doc_topics": <列表,该{node.dimension}子主题涵盖的所有输入文档主题>,
      "core_features": <字符串，详细说明该类别具有哪些其他兄弟类别无法包含的独特核心要素。必须列举具体的关键信息字段、核心功能、特定结构要求或涉及的特定主体。例如："介绍信需要'介绍人'、'介绍事由'、'被介绍人与介绍人的关系'等其他证明凭证不需要的要素">
    }},
    ...
  ]
}}

注意: 
- code字段是label的准确英文翻译,不是拼音。例如: "财务报告" -> "financial_report", "人力资源" -> "human_resources"
- core_features字段是关键，必须清晰说明该类别的独特核心要素是什么，以及为什么这些要素与其他类别有本质区别

---

你的输出JSON:

"""

# ============================================
# 深度扩展评估Prompt（可解释增强）
# ============================================

depth_expansion_evaluation_system_instruction = """你是一个分类体系深度扩展的评估助手。你的任务是判断一个节点是否应该进行深度扩展（生成子类别）。

⚠️ 核心判断原则：以文档语义核心要素为锚点

评估的根本依据是：**子类别是否具有父类别无法包含的独特核心要素**

核心要素是指：
- 文档必须包含的关键信息字段（如介绍信需要"介绍人"，但一般证明凭证不需要）
- 文档的核心功能和用途（如劳动合同强调劳动关系，与一般合同的侧重点不同）
- 文档的特定结构要求（如财务报表需要特定科目，而一般报告不需要）
- 文档涉及的特定主体或对象（如发票涉及买卖双方和税务信息，收据不涉及税务）

判断流程：
1. **识别潜在子类别**：从文档样本和聚类分布中，识别可能的子类别
2. **核心要素分析**：对每个潜在子类别，判断其是否有独特的核心要素（父类别无法包含的）
3. **做出决策**：如果存在多个具有独特核心要素的子类别，建议扩展；否则不扩展

你的判断应该基于实际观察到的文档内容，聚类分布仅作为辅助参考。
"""

def depth_expansion_evaluation_prompt(node, cluster_distribution, sample_docs, ancestors):
    """
    生成深度扩展评估的prompt
    
    Args:
        node: 当前节点
        cluster_distribution: 聚类分布字典 {cluster_id: count}
        sample_docs: 采样的文档列表 (list of EnterpriseDocument)
        ancestors: 祖先路径字符串
    
    Returns:
        tuple: (prompt_text, image_base64_list)
    """
    # 计算聚类统计
    total_clusters = len(cluster_distribution)
    total_docs = sum(cluster_distribution.values())
    small_clusters = sum(1 for count in cluster_distribution.values() if count < 3)
    medium_clusters = sum(1 for count in cluster_distribution.values() if 3 <= count < 10)
    large_clusters = sum(1 for count in cluster_distribution.values() if count >= 10)
    
    cluster_stats = f"""
聚类统计：
- 总文档数: {total_docs}
- 聚类数量: {total_clusters}
- 小簇数量 (<3份文档): {small_clusters} ({small_clusters/total_clusters*100:.1f}%)
- 中簇数量 (3-9份文档): {medium_clusters} ({medium_clusters/total_clusters*100:.1f}%)
- 大簇数量 (>=10份文档): {large_clusters} ({large_clusters/total_clusters*100:.1f}%)
- 平均簇大小: {total_docs/total_clusters:.1f}
"""
    
    # 显示聚类分布
    sorted_clusters = sorted(cluster_distribution.items(), key=lambda x: x[1], reverse=True)
    cluster_dist_str = ", ".join([f"cluster_{cid}: {count}份" for cid, count in sorted_clusters[:10]])
    if len(sorted_clusters) > 10:
        cluster_dist_str += f", ... (共{len(sorted_clusters)}个cluster)"
    
    # 构建文档样本展示
    doc_list = []
    image_base64_list = []
    
    for i, doc in enumerate(sample_docs[:10], 1):  # 最多展示10份文档
        doc_list.append(f"文档{i}:")
        doc_list.append(f"  标题: {doc.title}")
        doc_list.append(f"  内容摘要: {doc.get_summary(300)}")
        
        # 添加聚类信息
        cluster_label_col = 'cluster_label'  # 默认列名
        cluster_id = doc.metadata.get(cluster_label_col)
        if cluster_id is not None:
            doc_list.append(f"  所属cluster: {cluster_id} (该cluster共{cluster_distribution.get(cluster_id, 0)}份文档)")
        
        # 添加图片信息
        if doc.has_images():
            first_image_base64 = doc.get_image_base64(0)
            if first_image_base64:
                image_base64_list.append(first_image_base64)
                doc_list.append(f"  图片: [IMAGE_{len(image_base64_list)}] (可查看版面、格式、布局等)")
        
        doc_list.append("")
    
    doc_display = "\n".join(doc_list)
    
    # 添加图片说明
    image_note = ""
    if image_base64_list:
        image_note = f"\n\n注意：已提供{len(image_base64_list)}张图片样本，用[IMAGE_1]至[IMAGE_{len(image_base64_list)}]标记。请结合图片判断文档是否有明确的版面特征。"
    
    prompt_text = f"""
<输入>
<当前节点>
{node.label}
</当前节点>

<节点描述>
{node.label}是一种{node.dimension}类型: {node.description}
</节点描述>

<维度定义>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</维度定义>

<节点路径>
{ancestors}
</节点路径>

<聚类分析>
{cluster_stats}
聚类分布: {cluster_dist_str}
</聚类分析>

<文档样本（共展示{min(len(sample_docs), 10)}份）>
{doc_display}
</文档样本>
{image_note}
</输入>

请评估节点"{node.label}"是否应该进行深度扩展（生成更细分的子类别）。

⚠️ 评估标准 - 以文档语义核心要素为锚点：

**核心判断原则：子类别必须有父类别无法包含的独特核心要素**

**第一步：识别潜在子类别**
从提供的文档样本和聚类分布中，识别可能存在的子类别类型。

**第二步：核心要素分析**
对每个潜在子类别，逐一判断：
1. **识别独特要素**：该子类别有哪些必备的核心信息字段、关键功能、特定结构？
2. **对比父类别**：这些核心要素是父类别"{node.label}"无法包含或不必包含的吗？
3. **评估区分度**：这些核心要素是否足够显著，能够清晰地将子类别与父类别及其他子类别区分开？

**示例说明：**
- ✅ 应该扩展："证明凭证"下的"介绍信"
  - 独特要素：需要"介绍人"信息、介绍事由、被介绍人与介绍人的关系等
  - 父类别无法包含：一般"证明凭证"不需要这些要素
  - 结论：应该作为子类别细分

- ✅ 应该扩展："合同"下的"劳动合同"
  - 独特要素：劳动关系、工作内容、劳动报酬、工作时间、社保等
  - 父类别无法包含：一般合同不涉及这些劳动法相关要素
  - 结论：应该作为子类别细分

- ✗ 不应该扩展："行政文档"下的各种杂乱文档
  - 分析：虽然文档多样，但无法识别出具有独特核心要素的子类别
  - 文档过于分散，没有形成清晰的要素聚类
  - 结论：不扩展，保持粗粒度

**第三步：做出决策**
- **should_expand = true**：如果能识别出至少2个以上具有独特核心要素的子类别
- **should_expand = false**：如果无法识别出具有独特核心要素的子类别，或子类别之间缺乏清晰的要素边界

**注意事项：**
1. **聚类分布仅作参考**：聚类集中度不是决定性因素，核心要素差异才是
2. **数量不是关键**：即使某子类别文档少，但有独特核心要素也应扩展
3. **避免形式区分**：版面格式、呈现形式不算核心要素
4. **确保实质区分**：必须是语义层面的本质差异，而非细微的措辞或场景差异

你的输出应该是以下JSON格式：
{{
  "should_expand": <布尔值，true表示应该扩展，false表示不应该扩展>,
  "cluster_cohesion": <字符串，聚类凝聚度参考："high" / "medium" / "low"（仅作参考，不是决定因素）>,
  "has_clear_types": <布尔值，是否识别出具有独特核心要素的潜在子类别>,
  "reasoning": <字符串，详细说明你的判断依据，必须包括：
    1) 识别出哪些潜在子类别（如果有）
    2) 每个子类别的独特核心要素是什么
    3) 这些核心要素是否是父类别"{node.label}"无法包含的
    4) 基于核心要素分析，为什么建议扩展或不扩展
    注意：必须基于核心要素进行分析，而不是基于聚类分布或文档数量>
}}
"""
    
    return prompt_text, image_base64_list


# ============================================
# 组级别伪标签生成Prompt（用于基于聚类的扩展）
# ============================================

width_group_system_instruction = """你是一个执行分类体系宽度扩展的助手。你会收到一组相似的文档（这些文档已通过聚类算法归为同一组），需要综合判断这组文档应该归属于哪些类别。

核心要求：
1. **综合分析**：查看组内所有文档，综合判断它们的共同特征和差异
2. **一致性优先**：如果组内文档非常相似，应该只返回1个标签，确保标签一致性
3. **允许差异**：如果组内文档有显著的本质差异（不同的核心要素），可以返回多个标签（最多5个）
4. **避免过细**：不要因为细微差异（如格式、措辞、特定场景）就创建多个标签
"""

def width_group_main_prompt(documents, node, ancestors, nl='\n'):
    """
    生成组级别伪标签的主prompt
    
    Args:
        documents: 文档列表 (list of EnterpriseDocument)
        node: 父节点
        ancestors: 祖先路径字符串
    
    Returns:
        tuple: (prompt_text, image_base64_list) - prompt文本和对应的图片base64列表
    """
    # 构建文档列表展示
    doc_list = []
    image_base64_list = []  # 收集所有文档的第一张图片
    
    for i, doc in enumerate(documents, 1):
        doc_list.append(f"文档{i}:")
        doc_list.append(f"  标题: {doc.title}")
        doc_list.append(f"  内容摘要: {doc.get_summary(500)}")
        
        # 添加图片信息和占位符
        if doc.has_images():
            # 获取第一张图片的base64（为了节省token，每个文档只使用第一张）
            first_image_base64 = doc.get_image_base64(0)
            if first_image_base64:
                image_base64_list.append(first_image_base64)
                doc_list.append(f"  图片: [IMAGE_{len(image_base64_list)}] (文档的第一张图片，可能包含格式、布局、印章、图表等视觉信息)")
            else:
                doc_list.append(f"  图片: 无法读取")
        
        doc_list.append("")
    
    doc_display = nl.join(doc_list)
    
    # 添加图片说明（如果有图片）
    image_note = ""
    if image_base64_list:
        image_note = f"\n\n注意：文档组中包含{len(image_base64_list)}张图片，已用[IMAGE_1], [IMAGE_2]等标记。请结合图片的视觉信息（布局、格式、印章、图表等）进行判断。"
    
    prompt_text = f"""
<输入>
<父节点>
{node.label}
</父节点>

<父节点描述>
{node.label}是一种{node.dimension}类型: {node.description}
</父节点描述>

<维度定义>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</维度定义>

<父节点路径>
{ancestors}
</父节点路径>

<现有兄弟节点>
{nl.join([f"{c_label}: {c.description}" for c_label, c in node.get_children().items()])}
</现有兄弟节点>

<文档组（共{len(documents)}份文档）>
{doc_display}
</文档组>
{image_note}
</输入>

请综合分析以上{len(documents)}份文档（包括文本和图片信息），判断这组文档应该归属于父节点"{node.label}"下的哪些{node.dimension}类别标签。

⚠️ 核心判断标准 - 以文档语义核心要素为锚点：

1. **识别核心要素**：分析这组文档共同具有的核心要素
   - 核心要素：必须包含的关键信息字段、核心功能、特定结构要求、涉及的特定主体
   - 示例：介绍信需要"介绍人"、劳动合同需要"劳动关系条款"、发票需要"税务信息"

2. **优先一致性**：如果这些文档的核心要素相同或高度相似，应该只返回1个标签
   - 去除非核心修饰词（如具体组织名、人群、场景）后判断是否为同一类型
   
3. **允许差异**：只有在文档的核心要素有显著差异时，才返回多个标签（最多5个）
   - 必须判断：不同标签是否对应不同的核心要素集合？
   
4. **排除非实质性差异**：不要因为以下因素而区分标签
   - ❌ 呈现形式（表单vs文本vs图片）
   - ❌ 具体人名、日期、地点、场景等实例化信息
   - ❌ 细微的措辞差异
   - ❌ 页数或完整性
   - ❌ 版面布局、字体样式

5. **兄弟关系**：返回的标签应该是现有兄弟节点的兄弟（相同层级），而非子节点或父节点

6. **结合图片**：如果文档有图片，请结合图片的视觉信息辅助判断核心要素

你的输出应该是以下JSON格式:
{{
  "subtopic_labels": [<字符串列表，1-5个{node.dimension}类别标签，按重要性排序>],
  "reasoning": <字符串，简要说明为什么返回这些标签，特别是如果返回多个标签，需要说明它们之间的核心差异是什么>
}}

注意：标签名称应该使用中文，简洁明了，体现文档的本质特征。
"""
    
    return prompt_text, image_base64_list

depth_group_system_instruction = """你是一个执行分类体系深度扩展的助手。你会收到一组相似的文档（这些文档已通过聚类算法归为同一组），需要综合判断这组文档应该归属于哪些子类别。

核心要求：
1. **综合分析**：查看组内所有文档，综合判断它们的共同特征和差异
2. **一致性优先**：如果组内文档非常相似，应该只返回1个标签，确保标签一致性
3. **允许差异**：如果组内文档有显著的本质差异（不同的核心要素），可以返回多个标签（最多5个）
4. **避免过细**：不要因为细微差异（如格式、措辞、特定场景）就创建多个标签
5. **子类关系**：返回的标签应该是父节点的子类别（更具体），而非兄弟或父节点
"""

def depth_group_main_prompt(documents, node, ancestors, nl='\n'):
    """
    生成组级别伪标签的主prompt（深度扩展版本）
    
    Args:
        documents: 文档列表 (list of EnterpriseDocument)
        node: 父节点
        ancestors: 祖先路径字符串
    
    Returns:
        tuple: (prompt_text, image_base64_list) - prompt文本和对应的图片base64列表
    """
    # 构建文档列表展示
    doc_list = []
    image_base64_list = []  # 收集所有文档的第一张图片
    
    for i, doc in enumerate(documents, 1):
        doc_list.append(f"文档{i}:")
        doc_list.append(f"  标题: {doc.title}")
        doc_list.append(f"  内容摘要: {doc.get_summary(500)}")
        
        # 添加图片信息和占位符
        if doc.has_images():
            # 获取第一张图片的base64（为了节省token，每个文档只使用第一张）
            first_image_base64 = doc.get_image_base64(0)
            if first_image_base64:
                image_base64_list.append(first_image_base64)
                doc_list.append(f"  图片: [IMAGE_{len(image_base64_list)}] (文档的第一张图片，可能包含格式、布局、印章、图表等视觉信息)")
            else:
                doc_list.append(f"  图片: 无法读取")
        
        doc_list.append("")
    
    doc_display = nl.join(doc_list)
    
    # 添加图片说明（如果有图片）
    image_note = ""
    if image_base64_list:
        image_note = f"\n\n注意：文档组中包含{len(image_base64_list)}张图片，已用[IMAGE_1], [IMAGE_2]等标记。请结合图片的视觉信息（布局、格式、印章、图表等）进行判断。"
    
    prompt_text = f"""
<输入>
<父节点>
{node.label}
</父节点>

<父节点描述>
{node.label}是一种{node.dimension}类型: {node.description}
</父节点描述>

<维度定义>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</维度定义>

<父节点路径>
{ancestors}
</父节点路径>

<文档组（共{len(documents)}份文档）>
{doc_display}
</文档组>
{image_note}
</输入>

请综合分析以上{len(documents)}份文档（包括文本和图片信息），判断这组文档应该归属于父节点"{node.label}"下的哪些{node.dimension}子类别标签。

⚠️ 核心判断标准 - 以文档语义核心要素为锚点：

1. **识别核心要素**：分析这组文档共同具有的核心要素
   - 核心要素：必须包含的关键信息字段、核心功能、特定结构要求、涉及的特定主体
   - 示例：介绍信需要"介绍人"、劳动合同需要"劳动关系条款"、发票需要"税务信息"
   - 判断：这些核心要素是父类别"{node.label}"无法包含的吗？

2. **优先一致性**：如果这些文档的核心要素相同或高度相似，应该只返回1个标签
   - 去除非核心修饰词（如具体组织名、人群、场景）后判断是否为同一类型
   
3. **允许差异**：只有在文档的核心要素有显著差异时，才返回多个标签（最多5个）
   - 必须判断：不同标签是否对应不同的核心要素集合？
   
4. **排除非实质性差异**：不要因为以下因素而区分标签
   - ❌ 呈现形式（表单vs文本vs图片）
   - ❌ 具体人名、日期、地点、场景等实例化信息
   - ❌ 细微的措辞差异
   - ❌ 页数或完整性
   - ❌ 版面布局、字体样式

5. **子类关系**：返回的标签应该是父节点"{node.label}"的子类别（更具体、更细分），而不是兄弟或父节点
   - 确保子类别确实有父类别无法包含的独特核心要素

6. **结合图片**：如果文档有图片，请结合图片的视觉信息辅助判断核心要素

你的输出应该是以下JSON格式:
{{
  "subtopic_labels": [<字符串列表，1-5个{node.dimension}子类别标签，按重要性排序>],
  "reasoning": <字符串，简要说明为什么返回这些标签，特别是如果返回多个标签，需要说明它们之间的核心差异是什么>
}}

注意：标签名称应该使用中文，简洁明了，体现文档的本质特征。
"""
    
    return prompt_text, image_base64_list

# ============================================
# 深度扩展Prompt(生成子类别)
# ============================================

depth_system_instruction = """你是一个执行分类体系深度扩展的助手。深度扩展是指为给定的根主题节点添加更深层的子类别节点,这些子概念/主题完全属于指定的父节点,而不属于父节点的兄弟节点。

例如,给定一个文档类型分类体系,深度扩展"合同文档"(其兄弟节点是["报告文档","邮件文档","表单文档"])将创建子节点["采购合同","销售合同","劳动合同"](任何合适数量的子节点)。另一方面,"会议记录"不应该被添加,因为它属于兄弟节点"报告文档"。

你会收到一个父节点和一份文档的标题和内容。请识别该文档讨论的父节点的哪个子主题,该子主题应该比父节点更具体。换句话说,它们应该在分类体系中具有父子节点关系。
"""

def depth_main_prompt(document, node, ancestors, nl='\n'):
    """生成深度扩展的主prompt"""
    return f"""
<输入>
<父节点>
{node.label}
</父节点>

<父节点描述>
{node.label}是一种{node.dimension}类型: {node.description}
</父节点描述>

<维度定义>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</维度定义>

<父节点路径>
{ancestors}
</父节点路径>

<文档标题>
{document.title}
</文档标题>

<文档内容>
{document.get_summary(1500)}
</文档内容>

</输入>

根据输入的文档标题和内容,识别该文档属于父节点"{node.label}"下的哪个{node.dimension}类别标签。换句话说,回答问题:该文档提出的是哪种类型的{node.label} {node.dimension}?

你的输出应该是以下JSON格式:
{{
  "new_subtopic_label": <值类型为字符串; 字符串是一个新的主题标签(一种{node.dimension}类型),是该文档的真实主要主题,比{node.label}更深/更具体>
}}
"""

depth_cluster_system_instruction = """你是一个执行分类体系深度扩展的聚类助手。深度扩展是指为给定的根主题节点添加更深层的子类别节点,这些子概念/主题完全属于指定的父节点,而不属于父节点的兄弟节点。

例如,给定一个企业文档分类体系,深度扩展"合同文档"(其兄弟节点是["报告文档","邮件文档","表单文档"])将创建子节点["采购合同","销售合同","劳动合同"](任何合适数量的子节点)。另一方面,"会议记录"不应该被添加,因为它属于兄弟节点"报告文档"。

你需要根据讨论父主题的文档所涵盖的子主题,选择新的子主题聚类。你的工作是从输入的文档主题集合中识别独特的聚类。对于你识别的每个聚类,必须提供聚类名称(格式类似于文档主题)作为其键,聚类名称的一句话描述,以及该聚类涵盖的所有输入文档主题列表。

确保每个新子主题都是独特的并具有相同的粒度/具体程度。还要确保你的每个新主题聚类都是独特的;它们不应该已经存在于现有节点集合中(existing_nodes)。
"""

def depth_cluster_main_prompt(options, node, ancestors, all_node_labels):
    """生成深度扩展聚类的主prompt"""
    return f"""
<输入>
<父节点>
{node.label}
</父节点>

<父节点描述>
{node.label}是一种{node.dimension}类型: {node.description}
</父节点描述>

<维度定义>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</维度定义>

<父节点路径>
{ancestors}
</父节点路径>

你的新聚类主题不应该是以下任何一个:
<现有节点>
{all_node_labels}
</现有节点>

<文档主题>
以下是文档主题字典,其中每个键是候选节点标签,值是映射到该候选节点的文档数量:
候选节点标签:\n{str(options)}
</文档主题>

</输入>

在父节点主题"{node.label}"下,哪些主要的{node.dimension}子主题聚类最能涵盖上述<文档主题>?

⚠️ 核心要求 - 以文档语义核心要素为锚点进行分类：

**核心原则：只有在核心要素有显著差异时才创建不同的子类别**

1. **核心要素定义**：
   - ✅ 核心要素：文档必须包含的关键信息字段、核心功能、特定结构要求、涉及的特定主体或对象等本质性特征
   - 示例：介绍信需要"介绍人"、劳动合同需要"劳动关系条款"、发票需要"税务信息"
   - ❌ 非核心要素：具体格式、细微措辞、特定人群、具体场景、版面布局、页数等细节性差异
   
2. **子类别创建标准**：
   - 必须判断：该子类别是否有父类别"{node.label}"无法包含的独特核心要素？
   - 如果有独特核心要素，且这些要素足够显著，则创建子类别
   - 如果只是细节差异（如"研究生专项支部大会决议型预备党员接收志愿书"vs"支部大会决议型预备党员接收志愿书"），核心要素相同，应合并为一个类别
   
3. **合并相似类别**：
   - 如果多个候选主题的核心要素相同或高度相似，必须合并为一个类别
   - 判断依据：去除非核心的修饰词（如具体组织名、具体人群、具体场景）后，是否为同一类型？
   
4. **最少文档数要求**：
   - 每个聚类应至少包含3个以上的文档主题（或总计文档数>=5）
   - 例外：如果某类文档有非常独特的核心要素（如特殊的法律文书），即使数量少也可以单独成类
   
5. **排除非实质性区分依据**：
   - ❌ 呈现形式（表单vs文本vs图片）
   - ❌ 页数和完整性（多页文档统一归类，不完整但可识别的归入原类型）
   - ❌ 版面布局、字体样式等视觉特征
   - ❌ 具体人名、日期、地点等实例化信息

这些应该是不重叠的主题聚类,最好地代表和划分所有文档主题(最大化映射到每个聚类的文档数量)。它们都应该是父节点下的兄弟(相同深度/具体程度)。你建议的每个新聚类主题都应该是父节点"{node.label}"下更具体的子主题,并且是{node.dimension}类型。然而,它们都应该是同等独特的(非重复的),任何单个文档都不应该轻易地同时属于两个聚类。

你的输出应该是以下JSON格式,最少一个子主题聚类,最多五个:
{{
  "new_cluster_topics": [
    {{
      "label": <字符串,{node.dimension}子标签中文名称,比父节点{node.label}更深的深度/具体程度>,
      "code": <字符串,label的英文翻译,使用小写字母和下划线连接>,
      "description": <字符串,{node.dimension}子主题的一句话描述>,
      "covered_doc_topics": <列表,该{node.dimension}子主题涵盖的所有输入文档主题>,
      "core_features": <字符串，详细说明该子类别具有哪些父类别"{node.label}"无法包含的独特核心要素。必须列举具体的关键信息字段、核心功能、特定结构要求或涉及的特定主体。例如："介绍信需要'介绍人'、'介绍事由'、'被介绍人与介绍人的关系'等一般证明凭证不需要的要素">
    }},
    ...
  ]
}}

注意: 
- code字段是label的准确英文翻译,不是拼音。例如: "财务报告" -> "financial_report", "人力资源" -> "human_resources"
- core_features字段是关键，必须清晰说明该子类别的独特核心要素是什么，以及为什么这些要素是父类别无法包含的

---

你的输出JSON:

"""

# ============================================
# 文档分类Prompt
# ============================================

def classify_prompt(node, document):
    """
    生成文档分类的prompt
    如果文档有图片，会包含图片提示信息
    """
    has_images = hasattr(document, 'has_images') and document.has_images()
    image_count = document.get_image_count() if hasattr(document, 'get_image_count') else 0
    
    image_note = ""
    if has_images:
        image_note = f"\n注意: 此文档包含{image_count}张预览图片，请结合图片内容辅助进行分类判断。图片可能包含文档的格式、布局、图表等视觉信息。\n"
    
    return f"""根据以下文档的标题、内容{f' 和预览图片（共{image_count}张）' if has_images else ''},将文档分类到最合适的一个类别中。我们提供了类别选项(class_options)及其描述(class_descriptions)供你参考。这是一个单标签分类任务,每个文档只能属于一个类别。请选择最符合文档特征的类别。如果文档不应该被分配给任何类别,则输出-1。
{image_note}
⚠️ 分类原则：
1. 关注文档的内容本质，而非呈现形式（表单/文本等）
2. 多页文档或不完整文档只要能识别类型，就归入对应类别（如不完整的入党申请书仍归为"入党申请书"）
3. 只有完全无法判断原始类型的片段才考虑归入"文档组件"类

---
文档ID: {document.id}
文档标题: {document.title}
文档内容: {document.get_summary(2000)}
类别选项(类别ID: 类别标签名称): {"; ".join([f"{c.id}: {c.label}" for c in node.children.values()])}
类别描述: {"; ".join([f"{c.label}: {c.description}" for c in node.children.values()])}
---

你的输出格式应该是以下JSON格式:
---
{{
    "doc_id": {document.id},
    "class_options": {[c.id for c in node.children.values()]},
    "class_label": <整数,值是该文档应该被分配的唯一类别ID(选项在上面的'class_options'中提供),如果不属于任何类别则为-1>
}}
---
"""


# ============================================
# Region Schema 相关 Pydantic Schema
# ============================================

class RegionSchemaGeneration(BaseModel):
    """典型 Region Schema 生成结果（叶子节点专用），采用标准 JSON Schema 格式"""
    node_label: str
    node_id: str
    region_schema: dict  # JSON Schema object: {type, properties, ...}
    node_kv_schema: dict  # JSON Schema object: {type, properties, ...}
    reasoning: str


class RefinedRegionSchemaResult(BaseModel):
    """基于文档修正 Region Schema 的结果（迭代式方法的修正阶段）"""
    node_label: str
    node_id: str
    region_schema: dict  # JSON Schema object
    node_kv_schema: dict  # JSON Schema object
    changes: dict  # {"added_regions": [...], "removed_regions": [...], "added_keys": [...], "removed_keys": [...], "adjusted_structure": [...]}
    reasoning: str
    confidence: str  # low/medium/high


class RegionDeviationResult(BaseModel):
    """Cluster 偏离检测结果"""
    fits: bool
    structural_deviations: List[str] = []
    suggested_region_schema: Optional[dict] = None
    reasoning: str

class RegionExpansionCandidate(BaseModel):
    """扩展候选子类别"""
    label: str
    code: str
    description: str
    region_schema: dict  # JSON Schema object
    covered_clusters: List[str] = []

class RegionExpansionDecision(BaseModel):
    """Region 驱动的深度扩展决策"""
    should_expand: bool
    reasoning: str
    candidates: List[RegionExpansionCandidate] = []


# ============================================
# Region Schema 生成 Prompt
# ============================================

region_schema_system_instruction = """你是一个专业的文档结构分析专家，擅长通过分析真实文档来定义语义区域结构（Region Schema）。

Region Schema 采用 **标准 JSON Schema** 格式定义文档的层级化语义区域结构：顶层是一个 `type: "object"`，其 `properties` 的每个 key 是一个语义区域名称，区域内部的 `properties` 定义该区域可抽取的信息字段。

🚨 最重要的原则 —— 以文档为准，严禁臆造：
1. **只定义你在提供的文档示例中能够观察到的区域**。如果文档中没有出现某种信息，就不要凭名称猜测它"应该有"。
2. **先仔细阅读所有文档示例**，识别文档中实际出现了哪些信息区域，然后再组织成 Region Schema。
3. 宁可少定义区域，也不要臆造文档中不存在的区域。缺失的区域可以后续补充，但凭空捏造的区域会严重误导下游系统。

🚨 粒度匹配原则 —— Schema 必须匹配文档类型的抽象层级：
1. **Schema 的具体程度必须与文档类型的具体程度一致**。如果文档类型本身是一个泛化的类别（如"结构化数据组件"、"证明凭证"），那么 Schema 中的 key 也应该是泛化的（如"数据字段"、"记录条目"），**绝不能出现仅属于某个具体子类的业务字段**（如"党组织名称"、"入党时间"）。
2. 出现具体业务字段只能意味着两件事之一：(a) 该类型实际上应该有更细分的子类别来承载这些字段，或 (b) 文档一开始就被分类到了错误的类别。无论哪种情况，当前的泛类节点都不应该定义这些字段。
3. **判断标准**：每个 key 问自己——"这个 key 是否对该类型下的所有文档都普遍适用？"如果只适用于部分文档，说明粒度不匹配，不应该出现。

⚠️ 核心概念：
1. **语义区域 ≠ 版面区域**：region 是按信息的语义功能划分的逻辑区域，不是按物理位置划分的版面块。"签章区"不是指文档最后一页的签名位置，而是"签署相关信息的集合"。
2. **字段 key 是抽象的语义标识**：不关心具体表现形式。例如"甲方签章"可以是红色印章、手写签名或电子签——这些形式差异由下游抽取模型处理。
3. **同一概念可出现在不同区域**：如"甲方名称"在"协议方信息区"和"签章区"都可以出现，但语义角色不同（前者是主体声明，后者是签署确认），因此各区域独立定义字段。

⚠️ JSON Schema 结构规范：
- 整体是 `{"type": "object", "properties": {...}}`，顶层 properties 的每个 key 是一个语义区域
- 区域组（含子区域）：`"type": "object"` + `"properties": {子区域...}`
- 叶子区域（含抽取字段）：`"type": "object"` + `"properties": {字段...}`
- 字段类型：
  - 单值字段：`{"type": "string"}`
  - 简单多值：`{"type": "array", "items": {"type": "string"}}`
  - 结构化多条记录（表格明细）：`{"type": "array", "items": {"type": "object", "properties": {...}}}`
- 嵌套层级一般不超过 3 层
- 可选属性：`description`（按需）、`required`（高频必现字段列表）

📝 description 生成策略 —— 简洁优先，按需添加：
- **不需要 description**：字段名本身语义清晰的（如"姓名"、"发票号码"、"日期"、"地址"）
- **需要 description**：
  - 区域级：用一句话说明该区域的语义功能（如"购销双方基本信息"）
  - 领域专有名词或缩写（如字段 "USCI" → description: "统一社会信用代码"）
  - 存在歧义的字段（如"金额" → "不含税金额"）
  - array 类型字段说明每条记录代表什么
- **禁止**对每个字段都机械添加 "该字段表示XXX" 的冗余描述

🚨 禁止使用编号后缀 key：
- **严禁**出现 "介绍人1"、"介绍人2"、"签名1"、"签名2" 等编号命名。
- 多个同类实体用 `"type": "array"` 表达。
- 这条规则适用于 region_schema 和 node_kv_schema。"""


def region_schema_seed_prompt(node, ancestors, sibs, sample_docs=None):
    """生成典型 Region Schema 的 Prompt（JSON Schema 格式）"""
    if ancestors:
        ancestor_info = f"该节点在分类体系中的位置：{ancestors}"
    else:
        ancestor_info = "该节点是根节点"

    if sibs:
        sibling_info = f"与该节点平级的兄弟类别有：{', '.join(sibs)}"
    else:
        sibling_info = "该节点暂无兄弟类别"

    node_desc = f"\n节点描述：{node.description}" if node.description else ""

    doc_examples = ""
    has_docs = sample_docs and len(sample_docs) > 0
    if has_docs:
        doc_examples = "\n\n## 真实文档示例（你的 Region Schema 必须基于这些文档）\n\n**请先仔细通读以下所有文档**，识别文档中实际包含哪些信息区域，然后再组织成 Region Schema。\n"
        for idx, doc in enumerate(sample_docs[:5], 1):
            doc_examples += f"\n### 示例{idx}：\n"
            doc_examples += f"标题：{doc.title}\n"
            content_len = min(len(doc.content), 1500)
            doc_examples += f"内容：{doc.content[:content_len]}\n"
            if content_len < len(doc.content):
                doc_examples += f"...（内容已截断，原文共 {len(doc.content)} 字符）\n"

    grounding_rule = ""
    if has_docs:
        grounding_rule = """
🚨 **严禁臆造区域**：
- 你定义的每一个区域和字段 key 都必须能在上面的文档示例中找到对应的信息
- 如果文档中没有出现某类信息（如"研究生阶段表现"），即使你认为这类文档"应该有"，也**绝对不能**定义该区域
- 如果文档示例有图片，请结合图片分析文档的实际结构

🚨 **粒度匹配检查**：
- 生成前请先判断：该文档类型是一个**具体类别**（如"增值税专用发票"）还是一个**泛化类别**（如"结构化数据组件"、"表格类文档"）？
- 如果是泛化类别，key 必须是该类别下所有文档都通用的（如"表头"、"数据记录"），不能出现仅属于某个子类的具体业务字段
- 如果文档示例中的内容差异很大（有的是党员表、有的是财务表），说明该类别太宽泛，schema 应该只保留最通用的结构骨架"""
    else:
        grounding_rule = """
⚠️ 没有提供文档示例。请基于该文档类型的常识生成 Region Schema，但要保守——只定义该类型文档**几乎必然包含**的区域，不要猜测可能存在的区域。
⚠️ 注意粒度匹配：如果该类型是泛化类别（如"结构化数据组件"），只定义通用结构，不要出现任何具体业务领域的字段。"""

    prompt = f"""# 任务说明
请通过分析提供的真实文档，为文档类型 **"{node.label}"** 生成 Region Schema（语义区域结构定义）。

## 节点信息
- 节点名称：{node.label}
- 维度类型：{node.dimension}
{node_desc}
- {ancestor_info}
- {sibling_info}
{doc_examples}
{grounding_rule}

## JSON Schema 字段类型说明

所有字段使用标准 JSON Schema 类型定义：
- 单值字段：`{{"type": "string"}}`
- 简单多值：`{{"type": "array", "items": {{"type": "string"}}}}`
- 结构化多条记录（表格/明细）：`{{"type": "array", "items": {{"type": "object", "properties": {{...}}}}}}`
- 嵌套结构：`{{"type": "object", "properties": {{...}}}}`
- 可选添加 `"description"` 说明语义（仅在字段名不自明时添加）
- 可选添加 `"required"` 列出高频必现的字段名

🚨 **禁止编号 key**：不要写 "介绍人1"、"介绍人2"，而要用 `"type": "array"` 表达多实例。

## Region Schema 结构示例

**示例：增值税发票**
```json
{{
  "type": "object",
  "properties": {{
    "发票头信息区": {{
      "type": "object",
      "description": "发票基本标识信息",
      "properties": {{
        "发票类型": {{"type": "string"}},
        "发票代码": {{"type": "string"}},
        "发票号码": {{"type": "string"}},
        "开票日期": {{"type": "string"}}
      }},
      "required": ["发票类型", "发票代码", "发票号码", "开票日期"]
    }},
    "交易主体信息区": {{
      "type": "object",
      "description": "购销双方信息",
      "properties": {{
        "购买方信息": {{
          "type": "object",
          "properties": {{
            "名称": {{"type": "string"}},
            "纳税人识别号": {{"type": "string"}},
            "地址电话": {{"type": "string"}},
            "开户行及账号": {{"type": "string"}}
          }},
          "required": ["名称", "纳税人识别号"]
        }},
        "销售方信息": {{
          "type": "object",
          "properties": {{
            "名称": {{"type": "string"}},
            "纳税人识别号": {{"type": "string"}},
            "地址电话": {{"type": "string"}},
            "开户行及账号": {{"type": "string"}}
          }},
          "required": ["名称", "纳税人识别号"]
        }}
      }}
    }},
    "商品明细区": {{
      "type": "object",
      "description": "货物或服务的明细列表",
      "properties": {{
        "货物明细": {{
          "type": "array",
          "items": {{
            "type": "object",
            "properties": {{
              "货物名称": {{"type": "string"}},
              "规格型号": {{"type": "string"}},
              "单位": {{"type": "string"}},
              "数量": {{"type": "string"}},
              "单价": {{"type": "string"}},
              "金额": {{"type": "string"}}
            }}
          }}
        }}
      }}
    }},
    "合计与税额区": {{
      "type": "object",
      "description": "金额汇总信息",
      "properties": {{
        "合计金额": {{"type": "string"}},
        "合计税额": {{"type": "string"}},
        "价税合计(大写)": {{"type": "string"}},
        "价税合计(小写)": {{"type": "string"}}
      }}
    }}
  }}
}}
```

⚠️ 结构要点：
- 整体是 `{{"type": "object", "properties": {{...}}}}`，顶层 properties 的 key 即区域名
- 区域按语义功能划分，不按版面位置划分
- 字段名自明时不加 description；区域级和歧义字段才需要 description
- `required` 仅列出该类型文档中几乎必然出现的字段

## 输出格式

你需要同时输出两个 JSON Schema：

1. **region_schema**：层级化的语义区域结构（JSON Schema 格式，区域→子区域/字段）
2. **node_kv_schema**：扁平化的抽取要素 Schema（JSON Schema 格式），列出该文档类型的所有可抽取字段。相当于把 region_schema 里各区域的字段汇总扁平化，合并重复概念，用嵌套 object 保留层次感。

node_kv_schema 示例（对应上面增值税发票的 region_schema）：
```json
{{
  "type": "object",
  "properties": {{
    "发票类型": {{"type": "string"}},
    "发票代码": {{"type": "string"}},
    "发票号码": {{"type": "string"}},
    "开票日期": {{"type": "string"}},
    "购买方": {{
      "type": "object",
      "properties": {{
        "名称": {{"type": "string"}},
        "纳税人识别号": {{"type": "string"}},
        "地址电话": {{"type": "string"}},
        "开户行及账号": {{"type": "string"}}
      }}
    }},
    "销售方": {{
      "type": "object",
      "properties": {{
        "名称": {{"type": "string"}},
        "纳税人识别号": {{"type": "string"}},
        "地址电话": {{"type": "string"}},
        "开户行及账号": {{"type": "string"}}
      }}
    }},
    "货物明细": {{
      "type": "array",
      "items": {{
        "type": "object",
        "properties": {{
          "货物名称": {{"type": "string"}},
          "规格型号": {{"type": "string"}},
          "单位": {{"type": "string"}},
          "数量": {{"type": "string"}},
          "单价": {{"type": "string"}},
          "金额": {{"type": "string"}}
        }}
      }}
    }},
    "合计金额": {{"type": "string"}},
    "合计税额": {{"type": "string"}},
    "价税合计(大写)": {{"type": "string"}},
    "价税合计(小写)": {{"type": "string"}},
    "收款人": {{"type": "string"}},
    "复核人": {{"type": "string"}},
    "开票人": {{"type": "string"}}
  }}
}}
```

完整输出格式：
```json
{{
  "node_label": "{node.label}",
  "node_id": "{node.id}",
  "region_schema": {{
    "type": "object",
    "properties": {{
      // 各语义区域...
    }}
  }},
  "node_kv_schema": {{
    "type": "object",
    "properties": {{
      // 扁平化抽取要素...
    }}
  }},
  "reasoning": "说明：(1) 你在文档中观察到了哪些信息区域 (2) 如何组织成层级结构"
}}
```

现在请基于上面的文档示例，为"{node.label}"生成 Region Schema 和 node_kv_schema，只输出 JSON："""
    return prompt


# ============================================
# Region Schema 迭代修正 Prompt
# ============================================

refine_region_schema_system_instruction = """你是一个专业的文档结构分析专家，现在需要基于新的真实文档示例来修正和完善已有的 Region Schema（JSON Schema 格式）。

你的任务是：
1. 仔细分析本轮提供的文档示例
2. 对比当前 Region Schema 和 node_kv_schema，进行必要的调整：
   - **新增区域/字段**：文档中有但 Schema 中没有的重要区域或字段
   - **删除区域/字段**：之前基于有限样本加入的区域/字段，在更多文档中发现并非通用
   - **替换字段名**：发现更准确、更专业的字段名
   - **调整结构**：优化区域的层级组织，使其更合理
   - **保留合理部分**：Schema 中合理但本轮文档中未出现的区域/字段应保留（文档可能不完整）
   - **完善 description**：根据新文档补充必要的 description（仅限语义不自明的字段）
   - **更新 required**：根据多份文档观察，调整哪些字段是高频必现的

⚠️ 修正原则：
- 以文档中**实际出现的信息区域**为准
- 新增的区域/字段要确保是该类型文档的**通用要素**，而非个例
- 如果多份文档都包含某区域，说明它很重要，应该加入
- 如果某区域只在个别文档中出现，且不是该类型的通用特征，不要加入
- 保持嵌套层级合理（不超过 3 层）

⚠️ JSON Schema 结构规范：
- 整体 `{"type": "object", "properties": {...}}`
- 字段类型：`{"type": "string"}`、`{"type": "array", "items": {...}}`、`{"type": "object", "properties": {...}}`
- description 仅在字段名不自明时添加
- 🚨 禁止编号 key（如"签名1"、"签名2"），多实例用 `"type": "array"` 表达

输出格式要求：
- 返回**完整的**修正后的 region_schema 和 node_kv_schema（不是增量），均为 JSON Schema 格式
- Schema 只定义结构，绝对不要填充任何具体值"""


def refine_region_schema_prompt(node, current_region_schema, current_node_kv_schema, doc_batch, batch_idx, total_batches):
    """
    基于文档示例修正 Region Schema 的 Prompt

    Args:
        node: 当前节点
        current_region_schema: 当前的 region_schema（list）
        current_node_kv_schema: 当前的 node_kv_schema（dict）
        doc_batch: 当前批次的文档列表
        batch_idx: 当前批次索引（从 1 开始）
        total_batches: 总批次数
    """
    import json as _json
    rs_str = _json.dumps(current_region_schema, ensure_ascii=False, indent=2)
    kv_str = _json.dumps(current_node_kv_schema, ensure_ascii=False, indent=2)

    doc_examples = ""
    image_indices = []
    for idx, doc in enumerate(doc_batch, 1):
        doc_examples += f"\n### 文档示例 {idx}：\n"
        doc_examples += f"标题：{doc.title}\n"
        content_len = min(len(doc.content), 1500)
        doc_examples += f"内容：{doc.content[:content_len]}\n"
        if content_len < len(doc.content):
            doc_examples += f"...（内容已截断，原文共 {len(doc.content)} 字符）\n"
        if hasattr(doc, 'has_images') and doc.has_images():
            image_indices.append(idx)
            doc_examples += f"图片: [IMAGE_{len(image_indices)}]\n"

    image_note = ""
    if image_indices:
        image_note = f"\n注意：本轮有 {len(image_indices)} 张图片，请结合图片分析文档实际区域结构。"

    prompt = f"""# 任务说明
你正在对文档类型 **"{node.label}"** 的 Region Schema（JSON Schema 格式）进行第 {batch_idx}/{total_batches} 轮修正。

## 当前 Region Schema（待修正）
```json
{rs_str}
```

## 当前 node_kv_schema（待修正）
```json
{kv_str}
```

## 本轮文档示例
以下是 {len(doc_batch)} 份属于"{node.label}"类型的真实文档：
{doc_examples}
{image_note}

## 修正任务

请仔细分析上述文档，对 Region Schema 和 node_kv_schema 进行必要的修正：

### 1. 区域级修正
- **新增区域**：文档中存在但当前 Schema 没有覆盖的重要语义区域
- **删除区域**：之前加入的区域在更多文档中发现并非该类型通用特征
- **调整区域结构**：优化区域的层级组织（拆分、合并、调整父子关系）

### 2. 字段级修正
- **新增字段**：文档中出现的核心字段，当前 Schema 中缺失
- **删除字段**：只在个例中出现的字段，不具有通用性
- **替换字段名**：文档中使用了更准确、更专业的字段名
- **调整类型**：根据业务含义修正字段的 JSON Schema type（string / array / object）
- **完善 description**：为语义不自明的字段补充简短 description
- **更新 required**：根据多份文档观察调整必现字段列表

### 3. 保留原则
- Schema 中合理但本轮文档中未出现的区域/字段 → **保留**
- 只有在确认该区域/字段不具通用性时才删除

## 输出格式
请按照以下 JSON 格式输出**完整的**修正后结果（JSON Schema 格式）：

```json
{{
  "node_label": "{node.label}",
  "node_id": "{node.id}",
  "region_schema": {{
    "type": "object",
    "properties": {{
      // 完整的修正后 region_schema
    }}
  }},
  "node_kv_schema": {{
    "type": "object",
    "properties": {{
      // 完整的修正后 node_kv_schema
    }}
  }},
  "changes": {{
    "added_regions": ["新增的区域名称列表"],
    "removed_regions": ["删除的区域名称列表"],
    "added_keys": ["新增的字段列表"],
    "removed_keys": ["删除的字段列表"],
    "adjusted_structure": ["结构调整说明"]
  }},
  "reasoning": "本轮修正的主要改动和原因",
  "confidence": "low/medium/high"
}}
```

**重要提醒**：所有字段值必须是 JSON Schema 类型定义（如 `{{"type": "string"}}`），绝对不要填充具体值！

只输出 JSON："""
    return prompt


# ============================================
# Cluster 偏离检测 Prompt
# ============================================

region_deviation_system_instruction = """你是一个文档结构差异分析专家。你的任务是判断一组文档的实际区域结构是否与给定的典型 Region Schema（JSON Schema 格式）一致。

⚠️ 核心判断标准：
1. **只关注区域结构层面的差异**：是否有典型 schema 中不存在的全新区域？是否缺少典型 schema 中的重要区域？是否有区域的层级结构发生了根本变化？
2. **忽略字段层面的细节差异**：不同文档在同一区域内的 key 措辞不同（如"联系电话" vs "联系方式"）不算结构偏离。
3. **忽略内容差异**：不同文档填写的具体内容不同不算偏离，schema 只关注"应该有哪些区域和要素"。
4. **关注本质差异**：只有当文档群体在区域组成上有根本性不同（如某类合同多出"试用期条款区"而另一类没有），才算结构偏离。"""


def region_deviation_prompt(node, canonical_schema, cluster_docs):
    """
    生成 cluster 偏离检测的 Prompt

    Args:
        node: 当前节点
        canonical_schema: 典型 region schema (dict)
        cluster_docs: 该 cluster 的文档列表
    Returns:
        tuple: (prompt_text, image_base64_list)
    """
    import json as _json
    schema_str = _json.dumps(canonical_schema, ensure_ascii=False, indent=2)

    doc_list = []
    image_base64_list = []
    for i, doc in enumerate(cluster_docs[:8], 1):
        doc_list.append(f"文档{i}:")
        doc_list.append(f"  标题: {doc.title}")
        doc_list.append(f"  内容摘要: {doc.get_summary(500)}")
        if doc.has_images():
            img = doc.get_image_base64(0)
            if img:
                image_base64_list.append(img)
                doc_list.append(f"  图片: [IMAGE_{len(image_base64_list)}]")
        doc_list.append("")
    doc_display = "\n".join(doc_list)

    image_note = ""
    if image_base64_list:
        image_note = f"\n注意：已提供 {len(image_base64_list)} 张图片样本，请结合图片判断文档的实际区域结构。"

    prompt_text = f"""# 任务说明
请判断以下一组文档的实际区域结构是否符合给定的典型 Region Schema。

## 当前文档类型
{node.label}（{node.description or ''}）

## 典型 Region Schema
```json
{schema_str}
```

## 文档组（共 {len(cluster_docs)} 份，展示前 {min(len(cluster_docs), 8)} 份）
{doc_display}
{image_note}

## 判断要求
1. 分析这组文档的实际区域结构
2. 与典型 Region Schema 对比，判断是否存在**区域结构层面**的差异
3. 如果偏离显著，生成这组文档实际适用的 region schema

⚠️ 什么是"结构偏离"：
- ✅ 结构偏离：该组文档有典型 schema 中**不存在**的重要区域（如多出"试用期条款区"）
- ✅ 结构偏离：该组文档**缺少**典型 schema 中的重要区域
- ✅ 结构偏离：区域的层级组织方式有根本不同
- ✗ 不算偏离：同一区域内 key 的措辞不同（"联系电话" vs "联系方式"）
- ✗ 不算偏离：某些文档缺少部分 kv 值（文档不完整导致）
- ✗ 不算偏离：kv 的具体数量不同（如有 2 个签署方 vs 3 个签署方）

## 输出格式
```json
{{
  "fits": <布尔值，true 表示符合典型 schema，false 表示存在显著结构偏离>,
  "structural_deviations": [<字符串列表，描述具体的结构差异>],
  "suggested_region_schema": <如果 fits=false，给出这组文档实际适用的 region schema（JSON Schema 格式）；如果 fits=true 则为 null>,
  "reasoning": "判断依据的简要说明"
}}
```

只输出 JSON："""
    return prompt_text, image_base64_list


# ============================================
# Region 扩展决策汇总 Prompt
# ============================================

region_expansion_decision_system_instruction = """你是一个分类体系深度扩展的决策专家。你的任务是根据多个 cluster 组的 Region Schema（JSON Schema 格式）偏离分析结果，决定是否应该对当前节点进行深度扩展（细分为子类别）。

⚠️ 决策原则：
1. 只有当不同 cluster 组在**区域结构**上存在**本质性差异**时才建议扩展
2. 如果所有 cluster 都符合同一个典型 schema（仅有字段细节差异），不应扩展
3. 扩展产生的子类别必须有明显不同的 region schema（而不仅是部分字段的差异）
4. 子类别的数量要合理（通常 2-5 个），不要过度细分"""


def region_expansion_decision_prompt(node, canonical_schema, deviation_summaries, ancestors):
    """
    汇总偏离分析结果，生成扩展决策 Prompt

    Args:
        node: 当前节点
        canonical_schema: 典型 region schema
        deviation_summaries: 各 cluster 的偏离检测结果列表
        ancestors: 祖先路径字符串
    """
    import json as _json
    schema_str = _json.dumps(canonical_schema, ensure_ascii=False, indent=2)

    fits_count = sum(1 for d in deviation_summaries if d.get('fits', True))
    deviates_count = len(deviation_summaries) - fits_count

    summary_lines = []
    for i, d in enumerate(deviation_summaries, 1):
        status = "符合" if d.get('fits', True) else "偏离"
        cluster_id = d.get('cluster_id', f'cluster_{i}')
        doc_count = d.get('doc_count', '?')
        summary_lines.append(f"  - {cluster_id}（{doc_count} 份文档）: {status}")
        if d.get('structural_deviations'):
            for dev in d['structural_deviations']:
                summary_lines.append(f"    差异: {dev}")
    summary_display = "\n".join(summary_lines)

    return f"""# 任务说明
请根据以下偏离分析结果，决定是否对文档类型节点 **"{node.label}"** 进行深度扩展。

## 节点信息
- 节点名称：{node.label}
- 节点路径：{ancestors}
- 维度：{node.dimension}

## 典型 Region Schema
```json
{schema_str}
```

## 各 Cluster 偏离检测结果汇总
共分析 {len(deviation_summaries)} 个 cluster 组：
- 符合典型 schema: {fits_count} 个
- 存在结构偏离: {deviates_count} 个

详细结果：
{summary_display}

## 决策要求
1. 如果大多数 cluster 都符合典型 schema → 不扩展
2. 如果存在多个 cluster 有**本质性结构偏离**（不同的区域组成）→ 扩展
3. 如果只有 1-2 个 cluster 有轻微偏离 → 通常不扩展
4. 如果决定扩展，需要给出候选子类别及其各自的 region schema

## 输出格式
```json
{{
  "should_expand": <布尔值>,
  "reasoning": "详细说明决策依据",
  "candidates": [
    {{
      "label": "<子类别中文名称>",
      "code": "<英文编码，小写下划线>",
      "description": "<一句话描述>",
      "region_schema": {{"type": "object", "properties": {{...}}}},
      "covered_clusters": ["<覆盖的 cluster ID 列表>"]
    }}
  ]
}}
```

如果 should_expand=false，candidates 为空列表。
只输出 JSON："""


# ============================================
# 兄弟节点 Region Schema 冗余对比 Prompt
# ============================================

class SiblingMergeGroup(BaseModel):
    """一组建议合并的兄弟节点"""
    node_labels: List[str]
    merged_label: str
    merged_code: str
    merged_description: str
    merged_region_schema: dict  # JSON Schema object
    reasoning: str

class SiblingMergeResult(BaseModel):
    """兄弟节点 Region Schema 冗余对比结果"""
    has_redundancy: bool
    merge_groups: List[SiblingMergeGroup] = []
    reasoning: str


# ============================================================
# 非叶子节点 node_kv_schema 抽象生成
# ============================================================

class AbstractNodeKvSchema(BaseModel):
    """非叶子节点 node_kv_schema 抽象结果（JSON Schema 格式）"""
    node_label: str
    node_kv_schema: dict  # JSON Schema object
    reasoning: str


abstract_node_kv_schema_system_instruction = """你是一个专业的文档信息抽取 Schema 设计专家。你的任务是根据子类别各自的 node_kv_schema（JSON Schema 格式），为它们的父类别生成一个**更高层抽象**的 node_kv_schema。

node_kv_schema 是一个 JSON Schema，定义了从该类型文档中可以抽取哪些信息字段。

⚠️ 核心原则：
1. **抽象而非合并**：父节点的 schema 不是所有子节点 schema 的并集，而是一个提取了共性、忽略了细节的**概括性抽取模板**。
2. **保留共性字段**：所有（或大多数）子节点都有的字段应该保留，但名称可以泛化。例如子节点分别有"基本工资"和"商品单价"，父节点可抽象为"金额"。
3. **忽略子类特有的细节字段**：只有个别子类才有的细节字段不需要出现在父节点中。
4. **保持简洁**：父节点的 schema 应该比子节点的更短、更概括。字段数量通常 5-15 个。
5. **JSON Schema 类型**：`{"type": "string"}` 单值，`{"type": "array", "items": {...}}` 多值/记录列表，`{"type": "object", "properties": {...}}` 嵌套对象。
6. **description 按需**：只为语义不自明的字段添加 description。
7. **禁止编号 key**：不要出现 "字段1"、"字段2" 等编号命名，多实例用 `"type": "array"` 表达。"""


def abstract_node_kv_schema_prompt(parent_node, children_kv_schemas):
    """
    生成非叶子节点抽象 node_kv_schema 的 Prompt。

    Args:
        parent_node: 父节点
        children_kv_schemas: list of dict, 每项 {"label", "code", "description", "node_kv_schema"}
    """
    import json as _json

    children_display = []
    for i, cs in enumerate(children_kv_schemas, 1):
        schema_str = _json.dumps(cs['node_kv_schema'], ensure_ascii=False, indent=2) if cs.get('node_kv_schema') else "(无)"
        children_display.append(f"""### 子类别 {i}: {cs['label']}
- 描述: {cs.get('description', '')}
- node_kv_schema:
```json
{schema_str}
```""")

    children_text = "\n\n".join(children_display)

    return f"""# 任务说明
请根据以下 {len(children_kv_schemas)} 个子类别的 node_kv_schema（JSON Schema 格式），为它们的父类别 **"{parent_node.label}"** 生成一个**抽象概括性**的 node_kv_schema。

## 父节点信息
- 名称: {parent_node.label}
- 描述: {parent_node.description or ''}

## 子类别的 node_kv_schema

{children_text}

## 抽象要求
1. 找出大多数子类别共有的字段 → 保留，名称可以泛化
2. 只有个别子类才有的细节字段 → 不出现在父节点中
3. 结果应该简洁概括，让人一看就知道这类文档可以抽取哪些**核心信息**
4. 字段数量控制在 5-15 个
5. 输出必须是标准 JSON Schema 格式

## 输出格式
```json
{{
  "node_label": "{parent_node.label}",
  "node_kv_schema": {{
    "type": "object",
    "properties": {{
      // 抽象概括的抽取要素（JSON Schema 格式）
    }}
  }},
  "reasoning": "抽象思路：哪些是共性字段，哪些被泛化，哪些被忽略"
}}
```

只输出 JSON："""


# 保留旧接口向后兼容（已废弃，请使用 AbstractNodeKvSchema）
class AbstractRegionSchema(BaseModel):
    """[已废弃] 父节点 Region Schema 抽象结果"""
    node_label: str
    region_schema: dict  # JSON Schema object
    reasoning: str


# ============================================================
# 兄弟节点冗余合并
# ============================================================

sibling_merge_system_instruction = """你是一个分类体系质量审查专家。你的任务是对比同一父节点下各叶子类别的 Region Schema（JSON Schema 格式），判断是否存在本质上相同的类别（应该合并）。

⚠️ 判断标准——什么算"本质上相同"：
1. **区域结构高度重合**：两个类别的区域 properties 层级结构几乎一样（相同的区域名称和层级组织），仅在个别字段 key 上有措辞差异
2. **无区分性区域**：没有任何一个区域是 A 有而 B 没有的（或反之）
3. **功能等价**：两个类别描述的其实是同一种文档，只是名称不同

⚠️ 什么**不算**冗余：
1. 区域结构有本质不同：A 有"试用期条款区"而 B 没有
2. 同名区域但字段组成差异大：A 的"主体信息区"有法人代表等字段，B 的没有
3. 文档用途不同：虽然结构相似但一个是内部文档、一个是外部提交文档"""


def sibling_merge_prompt(parent_node, sibling_schemas):
    """
    生成兄弟节点 Region Schema 冗余对比 Prompt。

    Args:
        parent_node: 父节点
        sibling_schemas: list of dict, 每项包含 {"label", "code", "description", "region_schema"}
    """
    import json as _json

    siblings_display = []
    for i, s in enumerate(sibling_schemas, 1):
        schema_str = _json.dumps(s['region_schema'], ensure_ascii=False, indent=2) if s.get('region_schema') else "(未生成)"
        siblings_display.append(f"""### 类别 {i}: {s['label']}
- code: {s['code']}
- 描述: {s.get('description', '')}
- Region Schema:
```json
{schema_str}
```""")

    siblings_text = "\n\n".join(siblings_display)

    return f"""# 任务说明
请对比以下同父节点 **"{parent_node.label}"** 下的 {len(sibling_schemas)} 个叶子类别的 Region Schema，判断是否存在应该合并的冗余类别。

## 父节点信息
- 名称: {parent_node.label}
- 描述: {parent_node.description or ''}

## 各叶子类别的 Region Schema

{siblings_text}

## 对比要求
1. 逐对比较各类别的 Region Schema 结构
2. 如果两个或多个类别的区域结构**本质上相同**（参考系统指令中的判断标准），建议合并
3. 合并建议需要给出合并后的名称、描述和 region schema
4. 不存在冗余则 has_redundancy=false

## 输出格式
```json
{{
  "has_redundancy": <布尔值>,
  "merge_groups": [
    {{
      "node_labels": ["<应合并的类别名称列表>"],
      "merged_label": "<合并后的类别名称>",
      "merged_code": "<合并后的英文编码>",
      "merged_description": "<合并后的描述>",
      "merged_region_schema": {{"type": "object", "properties": {{...}}}},
      "reasoning": "<为什么这些类别应该合并>"
    }}
  ],
  "reasoning": "整体判断说明"
}}
```

只输出 JSON："""
