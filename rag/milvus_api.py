import os
import re
import json
import jieba
import uvicorn
from openai import OpenAI
from dotenv import load_dotenv  
import milvus_config as config
from contextlib import asynccontextmanager
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from typing import List, Optional, Dict, Any

# 导入辅助函数
from milvus_helper import generate_embedding, extract_keywords, split_markdown_content

# 导入数据模型
from milvus_dataclass import QueryRequest, InsertRequest, DeleteRequest

# =================================================================================

# 加载环境变量
load_dotenv(dotenv_path='rag/.env')

# 生命周期管理，asynccontextmanager简化异步资源管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的初始化代码
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT
    )
    print(f"👌已连接Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    yield
    # 停止时的清理代码
    connections.disconnect("default")
    print("👋已断开Milvus连接")

# 建立app
app = FastAPI(
    title="MilvusDB API",
    description="一个用于管理Milvus向量数据库的API，支持文档上传、查询和管理。",
    version="1.0.0",
    lifespan=lifespan
)

# 初始化OpenAI客户端
embedding_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.EMBEDDING_BASE_URL
)

### --------------------------------- ###
###               API实现             ###
### --------------------------------- ###

@app.get("/")
async def root():
    return {
        "message": "欢迎使用MilvusDB API！请访问 /docs 查看API文档。",
        "version": "1.0.0",
        "endpoints": {
            "/query": "查询相似文档",
            "/insert": "插入文档",
            "delete": "删除文档",
            "collection/info": "获取集合信息",
            "/health": "检查API健康状态"
        }
    }

@app.get("/health")
async def health_check():
    """
    健康检查接口，返回API状态
    """
    try:
        collections = utility.list_collections()
        return {"status": "health", "connection": "milvus保持连接", "collections_count": len(collections)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
    

@app.post("/query")
async def query(request: QueryRequest):
    """
    查询相似文档
    """
    if not utility.has_collection(config.COLLECTION_NAME):
        raise HTTPException(status_code=404, detail=f"集合 {config.COLLECTION_NAME} 不存在")
    
    collection = Collection(config.COLLECTION_NAME)
    collection.load()

    results = []
    if request.search_type == 'vector':
        # 仅向量搜索
        # query embedding
        query_embedding = generate_embedding(request.query)
        # 向量搜索
        search_results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=config.SEARCH_PARAMS,
            limit=request.top_k,
            output_fields=["id", "text", "source", "section", "keywords"]
        )

        for hits in search_results:
            for hit in hits:
                results.append({
                    "id": hit.id,
                    "score": float(1/(1+hit.distance)),  ## 反比函数做归一化，距离越大分数越小
                    "text": hit.entity.get("text", ""),
                    "source": hit.entity.get("source", ""),
                    "section": hit.entity.get("section", ""),
                    "keywords": hit.entity.get("keywords", ""),
                })

@app.post("/insert")
async def insert(request: InsertRequest):
    """
    插入文档，支持Markdown文本切分和关键词提取
    """
    try:
        if not utility.has_collection(config.COLLECTION_NAME):
            fields = []
            for f in config.FIELDS:
                if f["name"] == "id":
                    fields.append(FieldSchema(name=f["name"], dtype=DataType.INT64, is_primary=True, auto_id=True))
                elif f["name"] == "embedding":
                    fields.append(FieldSchema(name=f["name"], dtype=DataType.FLOAT_VECTOR, dimension=config.EMBEDDING_DIMENSION))
                elif f["type"] == "VARCHAR":
                    fields.append(FieldSchema(name=f["name"], dtype=DataType.VARCHAR, max_length=f["max_length"]))

            schema = CollectionSchema(fields=fields, description="AI知识库")
            collection = Collection(name=config.COLLECTION_NAME, schema=schema)

            index_params = {
                "index_type": config.INDEX_TYPE,
                "metric_type": config.METRIC_TYPE,
                "params": {"nlist": config.N_LIST}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            print(f"✅集合 {config.COLLECTION_NAME} 创建成功，索引类型: {config.INDEX_TYPE}, 距离度量: {config.METRIC_TYPE}")
        else:
            collection = Collection(config.COLLECTION_NAME)

        # 生成text的embedding和关键词
        embedding = generate_embedding(request.text)
        keywords = extract_keywords(request.text)

        # 插入数据
        data = [
            [request.text[:65000]],
            [embedding],
            [request.source or ""],
            [request.section or ""],
            [keywords]
        ]
        collection.insert(data)
        collection.flush()
        collection.load()

        return {"message": "文档插入成功", "source": request.source, "section": request.section, "keywords": keywords}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档插入失败: {str(e)}")


@app.post("/insert/file")
async def insert_file(file: UploadFile = File(...)):
    """
    上传并插入文件，支持Markdown文本切分和关键词提取
    """
    try:
        if not file.filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="仅支持Markdown文件上传")
        
        content = await file.read()
        text = content.decode('utf-8')
        
        # 查看有没有目标collection，没有就创建
        if not utility.has_collection(config.COLLECTION_NAME):
                fields = []
                for f in config.FIELDS:
                    if f["name"] == "id":
                        fields.append(FieldSchema(name=f["name"], dtype=DataType.INT64, is_primary=True, auto_id=True))
                    elif f["name"] == "embedding":
                        fields.append(FieldSchema(name=f["name"], dtype=DataType.FLOAT_VECTOR, dimension=config.EMBEDDING_DIMENSION))
                    elif f["type"] == "VARCHAR":
                        fields.append(FieldSchema(name=f["name"], dtype=DataType.VARCHAR, max_length=f["max_length"]))

                schema = CollectionSchema(fields=fields, description="AI知识库")
                collection = Collection(name=config.COLLECTION_NAME, schema=schema)

                index_params = {
                    "index_type": config.INDEX_TYPE,
                    "metric_type": config.METRIC_TYPE,
                    "params": {"nlist": config.N_LIST}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                print(f"✅集合 {config.COLLECTION_NAME} 创建成功，索引类型: {config.INDEX_TYPE}, 距离度量: {config.METRIC_TYPE}")
        else:
            collection = Collection(config.COLLECTION_NAME)

        texts = []
        embeddings = []
        sources = []
        sections = []
        keywords_list = []

        # 切分文本成多个chunk，每个chunk注入共同的标题信息，保证上下文完整，用的helper函数split_markdown_content
        chunks = split_markdown_content(text, source=file.filename)

        for chunk in chunks:
            text = chunk['text'][:65000]  # 截断文本以适应VARCHAR字段限制
            embedding = generate_embedding(text)
            keywords = extract_keywords(text)

            texts.append(text)
            embeddings.append(embedding)
            sources.append(chunk['source'])
            sections.append(chunk['section'])
            keywords_list.append(keywords)

        # 批量插入数据
        if texts:
            data = [
                texts,
                embeddings,
                sources,
                sections,
                keywords_list
            ]
            collection.insert(data)
            collection.flush()
            collection.load()
        
        return {"message": f"文件 '{file.filename}' 插入成功，共 {len(texts)} 个文本块", "source": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件插入失败: {str(e)}")

@app.delete("/delete")
async def delete(request: DeleteRequest):
    """
    删除文档，支持根据id、source或section删除
    """
    try:
        if not utility.has_collection(config.COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"集合 {config.COLLECTION_NAME} 不存在")
        
        collection = Collection(config.COLLECTION_NAME)
        if request.delete_type == 'id':
            if isinstance(request.value, list):
                expr = f"id in {request.value}"
            else:
                expr = f"id == {request.value}"

        elif request.delete_type == 'source':
            result = collection.query(expr=f"source == '{request.value}'", output_fields=["id"])

            if not result:
                return {"message": f"没有找到来源为 '{request.value}' 的文档"}
            ids_to_delete = [item['id'] for item in result]
            expr = f"id in {ids_to_delete}"

        elif request.delete_type == 'section':
            result = collection.query(expr=f"section == '{request.value}'", output_fields=["id"])

            if not result:
                return {"message": f"没有找到章节为 '{request.value}' 的文档"}
            ids_to_delete = [item['id'] for item in result]
            expr = f"id in {ids_to_delete}"

        else:
            raise HTTPException(status_code=400, detail="无效的删除类型, 可选值: 'id', 'source', 'section'")
        
        collection.delete(expr)
        collection.flush()

        return {
            "message": "删除成功",
            "delete_type": request.delete_type,
            "value": request.value
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
    
@app.get("/collection/info")
async def get_collection_info():
    """获取集合信息"""
    try:
        if not utility.has_collection(config.COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"集合 {config.COLLECTION_NAME} 不存在")

        collection = Collection(config.COLLECTION_NAME)
        collection.load()

        # 获取字段信息
        fields = []
        for field in collection.schema.fields:
            field_info = {
                "name": field.name,
                "type": field.dtype.name,
                "is_primary": field.is_primary,
                "auto_id": field.auto_id
            }
            if field.dtype.name == "FLOAT_VECTOR":
                field_info["dim"] = field.dim
            elif field.dtype.name == "VARCHAR":
                field_info["max_length"] = field.params.get("max_length")
            fields.append(field_info)

        # 获取索引信息
        indexes = []
        for index in collection.indexes:
            indexes.append({
                "field_name": index.field_name,
                "index_type": index.params.get("index_type"),
                "metric_type": index.params.get("metric_type"),
                "params": index.params.get("params", {})
            })

        return {
            "name": config.COLLECTION_NAME,
            "num_entities": collection.num_entities,
            "fields": fields,
            "indexes": indexes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取信息失败: {str(e)}")

@app.get("/collection/stats")
async def get_collection_stats():
    """获取集合统计信息"""
    try:
        if not utility.has_collection(config.COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"集合 {config.COLLECTION_NAME} 不存在")

        collection = Collection(config.COLLECTION_NAME)
        collection.load()

        # 获取来源统计
        sources_result = collection.query(
            expr="id > 0",
            output_fields=["source"],
            limit=999999
        )

        source_stats = {}
        for r in sources_result:
            source = r["source"]
            source_stats[source] = source_stats.get(source, 0) + 1

        return {
            "total_documents": collection.num_entities,
            "source_distribution": source_stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

def main():
    print("🚀正在启动MilvusDB API...")
    print(f"Milvus Root连接信息: {config.MILVUS_HOST}:{config.MILVUS_PORT}, 集合名称: {config.COLLECTION_NAME}")
    print(f"API文档地址: http://localhost:8000/docs")
    print(f"openapi地址: http://localhost:8000/openapi.json")

    uvicorn.run(
        "milvus_api:app", 
        host="0.0.0.0", 
        port=8000, reload=True, 
        log_level="info"
        )
    
if __name__ == "__main__":
    main()