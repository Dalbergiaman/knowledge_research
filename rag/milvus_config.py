import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='rag/.env')

# milvus configuration
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = int(os.getenv('MILVUS_PORT', 19530))
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'ai_knowledge_base')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 1024))

# 阿里云配置
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-V4')
EMBEDDING_BASE_URL = os.getenv('EMBEDDING_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')

# 配置索引
INDEX_TYPE = os.getenv('INDEX_TYPE', 'IVF_FLAT')

## 计算相似度的方式（IP(INNER_PRODUCT) L2 COSINE HAMMING JACCARD）
METRIC_TYPE = os.getenv('METRIC_TYPE', 'L2')

N_LIST = int(os.getenv('N_LIST', 128))  
NPROBE = int(os.getenv('NPROBE', 10))  


TOP_K = int(os.getenv('TOP_K', 10))
SEARCH_PARAMS = {
    "metric_type": METRIC_TYPE,
    "params": {"nprobe": NPROBE}
}

# 定义字段
FIELDS = [
    {"name": "id", "type": "INT64", "is_primary": True, "auto_id": True},
    {"name": "text", "type": "VARCHAR", "max_length": 65535},
    {"name": "embedding", "type": "FLOAT_VECTOR", "dimension": EMBEDDING_DIMENSION},
    {"name":"source", "type": "VARCHAR", "max_length": 500},
    {"name":"section", "type": "VARCHAR", "max_length": 500},
    {"name":"keywords", "type": "VARCHAR", "max_length": 1000},
]