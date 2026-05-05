"""
企业文档对象定义
"""

import os
import base64
from typing import Optional, List


class EnterpriseDocument:
    """企业文档类"""
    
    def __init__(self, doc_id, title, content, metadata=None, label_opts=None, image_url_list=None):
        """
        初始化企业文档对象
        
        Args:
            doc_id: 文档唯一标识
            title: 文档标题
            content: 文档内容(前N个字符或摘要)
            metadata: 文档元数据，应包含以下字段：
                - _index: 文档索引
                - hash_value: 文档哈希值
                - domain_code: 领域代码
                - domain_name: 领域名称
                - scene_code: 场景代码
                - scene_name: 场景名称
                - file_type: 文件类型代码
                - file_type_name: 文件类型名称
                - file_path: 文件路径
                - file_name: 文件名
            label_opts: 可选的标签维度列表
            image_url_list: 文档图片URL列表（可以是文件路径或base64编码）
        """
        self.id = doc_id
        self.title = title
        self.content = content
        self.metadata = metadata or {}
        self.labels = {l: [] for l in (label_opts or [])}
        
        # 企业文档规范化metadata字段
        self._index = self.metadata.get('_index', None)
        self.hash_value = self.metadata.get('hash_value', None)
        self.domain_code = self.metadata.get('domain_code', None)
        self.domain_name = self.metadata.get('domain_name', 'unknown')
        self.scene_code = self.metadata.get('scene_code', None)
        self.scene_name = self.metadata.get('scene_name', 'unknown')
        self.file_type = self.metadata.get('file_type', None)
        self.file_type_name = self.metadata.get('file_type_name', 'unknown')
        self.file_path = self.metadata.get('file_path', None)
        self.file_name = self.metadata.get('file_name', 'unknown')
        
        # 文档图片列表
        self.image_url_list = image_url_list or []  # 可以是文件路径列表或base64编码列表
    
    def add_label(self, label, dimension):
        """为文档添加标签（单标签分类：替换当前标签）"""
        if dimension in self.labels:
            self.labels[dimension] = [label]
    
    def get_summary(self, max_length=500):
        """获取文档摘要"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def get_images_base64(self) -> List[str]:
        """
        获取所有图片的base64编码列表
        如果图片是文件路径，则读取并转换为base64
        如果已经是base64编码，直接返回
        
        Returns:
            图片base64编码列表
        """
        if not self.image_url_list:
            return []
        
        base64_list = []
        for image_url in self.image_url_list:
            if not image_url:
                continue
                
            # 如果是文件路径，读取并转换为base64
            if os.path.exists(image_url):
                try:
                    with open(image_url, 'rb') as image_file:
                        base64_str = base64.b64encode(image_file.read()).decode('utf-8')
                        base64_list.append(base64_str)
                except Exception as e:
                    print(f"警告: 无法读取图片 {image_url}: {str(e)}")
            else:
                # 假设已经是base64编码
                base64_list.append(image_url)
        
        return base64_list
    
    def get_image_base64(self, index: int = 0) -> Optional[str]:
        """
        获取指定索引的图片base64编码
        
        Args:
            index: 图片索引，默认为0（第一张图片）
            
        Returns:
            指定图片的base64编码，如果不存在则返回None
        """
        images = self.get_images_base64()
        if 0 <= index < len(images):
            return images[index]
        return None
    
    def has_images(self) -> bool:
        """检查文档是否有图片"""
        return len(self.image_url_list) > 0
    
    def get_image_count(self) -> int:
        """获取文档图片数量"""
        return len(self.image_url_list)
    
    def __str__(self):
        return (f"EnterpriseDocument(id: {self.id}, title: '{self.title}', "
                f"domain: '{self.domain_name}', scene: '{self.scene_name}', "
                f"file_type: '{self.file_type_name}', images: {self.get_image_count()}, "
                f"labels: {self.labels})")
    
    def __repr__(self):
        return self.__str__()
