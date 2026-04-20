from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class Domain(str, Enum):
    "枚举可选的领域"
    LAW = "law"
    FINANCE = "finance"
    AI = "ai"
    ARCHITECTURE = "architecture"
    GENERAL = "general"


class ChatRequest(BaseModel):
    "聊天请求数据模型"
    question: str = Field(..., description="用户输入的问题")
    use_rag: bool = Field(True, description="是否使用RAG进行回答")
    top_k: int = Field(5, description="检索的top k片段数量")
    search_type: str = Field("hybrid", description="搜索类型, 可选值: 'hybrid', 'vector', 'keyword'")
    stream: bool = Field(False, description="是否使用流式响应")
    enable_query_expansion: bool = Field(True, description="是否启用查询扩展")
    enable_intent_recognition: bool = Field(True, description="是否启用意图识别")


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str = Field(..., description="生成的回答文本")
    source: List[Dict] = Field([], description="参考来源")
    search_results: Optional[List[Dict]] = Field(None, description="检索到的相关片段列表")
    domain: Optional[str] = Field(None, description="识别出的领域")
    expanded_query: Optional[List[str]] = Field(None, description="扩展后的查询文本")
    referenced_fragments: Optional[List[Dict]] = Field(None, description="引用的片段列表")


