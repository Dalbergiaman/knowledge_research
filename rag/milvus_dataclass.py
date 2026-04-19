from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# 定义milvus api数据模型

# 定义查询请求模型
class QueryRequest(BaseModel):
    """查询请求模型，包含查询文本、返回结果数量、搜索类型、向量搜索权重和是否重新排序字段"""
    query: str = Field(..., description="查询文本")
    top_k: int = Field(10, description="返回结果数量")
    search_type: str = Field("hybrid", description="搜索类型, 可选值: 'hybrid', 'vector', 'keyword'")
    vector_weight: float = Field(0.5, description="向量搜索权重, 仅在hybrid搜索中有效")
    rerank: bool = Field(False, description="是否对结果进行重新排序")

# 定义插入请求模型
class InsertRequest(BaseModel):
    """插入请求模型，包含文本内容和来源信息"""
    text: str = Field(..., description="文本内容")
    source: Optional[str] = Field(None, description="文本来源信息")
    section: Optional[str] = Field(None, description="文本所属章节")

# 定义删除请求的模型
class DeleteRequest(BaseModel):
    """删除请求模型"""
    delete_type: str = Field(..., description="删除类型, 可选值: 'id', 'source', 'section'")
    value: str = Field(..., description="删除条件值")

# 集合的信息模型
class CollectionInfo(BaseModel):
    """集合信息模型，包含集合名称、总文档数和索引类型"""
    name: str = Field(..., description="集合名称")
    num_entities: int = Field(..., description="集合中的文档数量")
    index: List[Dict[str, Any]] = Field(..., description="集合中使用的索引类型列表")