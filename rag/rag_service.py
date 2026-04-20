"""
定义rag服务
支持向量检索和llm生成, 提供完整的rag服务接口
支持查询拓展，意图识别，领域过滤，参考片段
"""

import os
import re
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from enum import Enum

# 导入dataclass
from rag_dataclass import ChatRequest, ChatResponse, Domain

load_dotenv("rag/.env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在这里进行任何需要在应用启动时执行的初始化操作
    print("RAG服务启动")
    yield
    # 在这里进行任何需要在应用关闭时执行的清理操作
    print("RAG服务关闭")

app = FastAPI(
    title="RAG服务",
    description="一个基于milvus向量检索和LLM生成的RAG服务",
    version="1.0.0",
    lifespan=lifespan
    )

# 配置
MILVUS_API_URL = "http://localhost:8000"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化llm客户端
llm_client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 开始构建RAG服务的核心逻辑，包括意图识别，查询拓展，领域过滤，参考片段提取等功能
class RAGPipeline:
    """构建rag pipelinr"""

    @staticmethod
    def detect_intent_with_llm(question: str) -> Tuple[Domain, float, str]:
        """使用llm进行意图识别，返回领域"""
        try:
            # 这里调用llm接口进行意图识别
            prompt = f"""请你分析以下问题属于哪个领域，并给出你的判断依据和置信度。
            问题: {question}

            可选领域如下：
            1. law (法律)：涉及法律法规、司法案例、合同协议等相关问题。
            2. finance (金融)：涉及投资理财、金融市场、银行业务等相关问题。
            3. ai (人工智能)：涉及机器学习、深度学习、自然语言处理等相关问题。
            4. architecture (建筑)：涉及建筑设计、施工、建筑材料等相关问题。
            5. general (综合)：不属于以上领域的一般性问题。

            输出要求，你必须按照以下json格式输出，只能返回json, 严禁markdown格式或者其他格式:
            {{
            "domain": "law",  // 识别出的领域，必须是上述五个领域之一
            "confidence": 0.95,  // 置信度，范围0-1
            "reason": "你的判断依据，简要说明为什么这个问题属于该领域"
            }}

            再次注意：只返回json，不要输出其他内容。
            """

            messages = [
                {"role": "system", "content": "你是一个专业的意图识别专家，能够准确识别用户问题所属的领域。"},
                {"role": "user", "content": prompt}
            ]


            completion = llm_client.chat.completions.create(
                model="qwen3.6-plus",  # 您可以按需更换为其它深度思考模型
                messages=messages,
                temperature=0.2,
                max_tokens=200,
                extra_body={"enable_thinking": False},
                stream=False
            )
            response_content = completion.choices[0].message.content.strip()

            import json
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                domain_str = result.get("domain", "general")
                confidence = float(result.get("confidence", 0.0))
                reason = result.get("reason", "")

                domain_map = {
                    "law": Domain.LAW,
                    "finance": Domain.FINANCE,
                    "ai": Domain.AI,
                    "architecture": Domain.ARCHITECTURE,
                    "general": Domain.GENERAL
                }

                domain = domain_map.get(domain_str, Domain.GENERAL)
                return domain, confidence, reason
        except Exception as e:
            print(f"意图识别失败: {e}")
            
        return Domain.GENERAL, 0.5, "未识别出领域，默认归为general"
    
    @staticmethod
    def expand_query_with_llm(question: str, domain: Domain) -> List[str]:
        """
        使用LLM进行查询拓展
        """
        domain_context = {
            Domain.LAW: "法律领域相关的背景知识和术语",
            Domain.FINANCE: "金融领域相关的背景知识和术语",
            Domain.AI: "人工智能领域相关的背景知识和术语",
            Domain.ARCHITECTURE: "建筑领域相关的背景知识和术语",
            Domain.GENERAL: "一般性背景知识和术语"
        }

        prompt = f"""请你根据以下问题和领域信息，生成一些相关的查询扩展词或者短语，以帮助更好地检索相关信息。
        原始问题： {question}
        领域信息： {domain_context.get(domain, "通用领域")}

        要求：
        1. 从不同角度扩展原始问题
        2. 保持与原问题的相关性
        3. 每个拓展查询独立一行
        4. 直接输出查询内容，不需要编号

        请输出拓展查询：
        """

        messages = [
            {"role": "system", "content": "你是一个专业的查询扩展专家，能够根据用户的问题和领域信息生成相关的查询扩展词。"},
            {"role": "user", "content": prompt}
        ]

        completion = llm_client.chat.completions.create(
                model="qwen3.6-plus",  # 您可以按需更换为其它深度思考模型
                messages=messages,
                temperature=0.2,
                max_tokens=200,
                extra_body={"enable_thinking": False},
                stream=False
            )
        response_content = completion.choices[0].message.content.strip()
        expanded_queries = [line.strip() for line in response_content.split("\n") if line.strip()][:3]
        return expanded_queries

    @staticmethod
    def search_knowledge_with_filter(quries: List[str], domain: Domain, top_k: int=5, search_type: str="hybrid") -> List[Dict[str, Any]]:
        """
        根据领域过滤进行知识检索
        """
        # 这里调用milvus的搜索接口，传入查询和领域过滤条件
        all_results = []
        seen_ids = set()

        for query in quries:
            try:
                request_data = {
                    "query": query,
                    "top_k": top_k,
                    "search_type": search_type,
                    "vector_weight": 0.7,
                    "rerank": True,
                }
                response = requests.post(f"{MILVUS_API_URL}/query", json=request_data)

                if response.status_code == 200:
                    results = response.json()

                    for item in results.get("results", []):
                        if item["id"] not in seen_ids:
                            seen_ids.add(item["id"])
                            all_results.append(item)

            except Exception as e:
                print(f"检索{query}失败: {e}")

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k]
    
    @staticmethod
    def search_knowledge(query: str, top_k: int=5, search_type: str="hybrid") -> List[Dict[str, Any]]:
        
        try:
            request_data = {
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "vector_weight": 0.7,
                "rerank": True,
            }
            response = requests.post(f"{MILVUS_API_URL}/query", json=request_data)

            if response.status_code == 200:
                results = response.json()
                return results
            else:
                print(f"检索失败，状态码: {response.status_code}, 响应内容: {response.text}")
            
        except Exception as e:
            print(f"检索失败: {e}")
        
        return None

    @staticmethod
    def build_context_with_fragments(search_result):
        """构建上下文并提取引用片段"""
        if not search_result:
            return "", []
        
        # 处理不同格式的输入
        if isinstance(search_result, dict) and "results" in search_result:
            results = search_result["results"]
        elif isinstance(search_result, list):
            results = search_result
        else:
            return "", []
        
        if not results:
            return "", []
        
        context_parts = []
        fragments = []

        for i, result in enumerate(results[:5]):
            text = result.get("text", "")
            fragment_length = 300
            if len(text) > fragment_length:
                fragment = text[:fragment_length//2] + "..." + text[-fragment_length//2:]
            else:
                fragment = text

            context_parts.append(f"参考片段{i+1}:\n{fragment}\n")

            fragments.append(
                {
                    "index": i+1,
                    "source": result.get("source", "unknown"),
                    "section": result.get("section", "unknown"),
                    "score": result.get("score", 0),
                    "fragment": fragment[:200] + "..." if len(fragment) > 200 else fragment
            }
            )

        return "\n\n".join(context_parts), fragments
    
    @staticmethod
    def build_context(search_result):
        """构建上下文"""
        context, _ = RAGPipeline.build_context_with_fragments(search_result)
        return context
    
    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        """构建给LLM的prompt"""
        prompt = f"""你是一个专业的AI助手, 请你基于下面的参考内容回答用户的问题，如果参考内容不足以回答问题，请你诚实的说明，并提供你知道的相关信息。
        
        参考片段:
        {context}

        问题: {question}

        要求：
        1. 回答要详细且信息丰富
        2. 直接回答问题，不要输出其他无关内容

        请生成回答：
        """
        return prompt

    @staticmethod
    def generate_answer(prompt: str, stream: bool = False) -> str:
        """调用LLM接口生成回答"""
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的AI助手，能够基于提供的参考内容生成详细且信息丰富的回答。"},
                {"role": "user", "content": prompt}
            ]

            if stream:
                # 这里可以实现流式响应的逻辑，逐步返回生成的内容
                completion = llm_client.chat.completions.create(
                    model="qwen3.6-plus",  # 您可以按需更换为其它深度思考模型
                    messages=messages,
                    extra_body={"enable_thinking": False},
                    stream=True
                )

                return completion  # 直接返回流式生成的对象，由调用方处理流式响应
            else:

                completion = llm_client.chat.completions.create(
                        model="qwen3.6-plus",  # 您可以按需更换为其它深度思考模型
                        messages=messages,
                        extra_body={"enable_thinking": False},
                        stream=False
                    )
                response_content = completion.choices[0].message.content
                return response_content
        except Exception as e:
            print(f"生成回答失败: {str(e)}")
            return "抱歉，生成回答时发生错误: {str(e)}"

# ========定义路由部分========

@app.get("/")
async def root():
    return {
        "service": "RAG服务",
        "version": "1.0.0",
        "description": "一个基于milvus向量检索和LLM生成的RAG服务",
        "endpoints": {
            "/chat": "智能对话",
            "/health": "健康检查",
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """rag对话接口"""
    # RAG流水线实例化
    pipeline = RAGPipeline()

    # 1. 意图识别
    domain = Domain.GENERAL
    intent_reason = ""

    if request.enable_intent_recognition:
        domain, confidence, intent_reason = pipeline.detect_intent_with_llm(request.question)
        print(f"识别领域: {domain}, 置信度: {confidence:.2f}")

        if domain == Domain.GENERAL and confidence < 0.5:  #如果通用领域且置信度较低，可以考虑直接回答
            prompt = pipeline.build_prompt(request.question)
            answer = pipeline.generate_answer(prompt, stream=request.stream)
            return ChatResponse(
                answer=answer, 
                source=[], 
                search_results=None, 
                domain=domain.value,
                expanded_query=None,
                referenced_fragments=None
            )
        
    # 2. RAG检索
    if request.use_rag:
        # 2.1 查询拓展
        if request.enable_query_expansion:
            expanded_queries = pipeline.expand_query_with_llm(request.question, domain)
            print(f"拓展查询: {expanded_queries}")
        else:
            expanded_queries = [request.question]

        # 2.2 知识检索
        if len(expanded_queries) > 1:
            search_result = pipeline.search_knowledge_with_filter(
                expanded_queries, 
                domain, 
                top_k=request.top_k, 
                search_type=request.search_type
                )
            # 格式化标准结果
            search_result_dict = {
                "results": search_result,
            }
        else:
            search_result = pipeline.search_knowledge(
                query=request.question, 
                top_k=request.top_k, 
                search_type=request.search_type
                )
            search_result_dict = {
                "results": search_result,
            }

        # 2.3 构建上下文和提取引用片段
        context, referenced_fragments = pipeline.build_context_with_fragments(search_result_dict)

        # 3. LLM生成回答
        if context:
            enhanced_prompt = f"""你是一个专业的{domain.value}领域的助手, 请你基于下面的参考内容回答用户的问题，如果参考内容不足以回答问题，请你诚实的说明，并提供你知道的相关信息。
            用户问题： {request.question}

            意图识别的结果：{intent_reason}

            参考片段:
            {context}

            要求：
            1. 回答要详细且信息丰富
            2. 直接回答问题，不要输出其他无关内容

            请生成回答：
            """
        else:
            enhanced_prompt = pipeline.build_prompt(request.question, context)

        # 4. 生成回答
        answer = pipeline.generate_answer(enhanced_prompt, stream=request.stream)

        source = []
        for frag in referenced_fragments[:3]:
            source.append(
                {
                    "source": frag["source"],
                    "section": frag["section"],
                    "score": frag["score"],
                }
            )
        
        return ChatResponse(
            answer=answer, 
            source=source, 
            search_results=search_result_dict.get("results", []), 
            domain=domain.value,
            expanded_query=expanded_queries if len(expanded_queries) > 1 else None,
            referenced_fragments=referenced_fragments
        )
    else:
        # 不使用RAG，直接生成回答
        prompt = pipeline.build_prompt(request.question, context="")

        answer = pipeline.generate_answer(prompt, stream=request.stream)
        return ChatResponse(
            answer=answer, 
            source=[], 
            search_results=None, 
            domain=domain.value if request.enable_intent_recognition else None,
            expanded_query=None,
            referenced_fragments=None
            )
    
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """调用流式响应接口"""

    from fastapi.responses import StreamingResponse
    import asyncio

    async def generate():
        pipeline = RAGPipeline()

        if request.use_rag:
            search_results = pipeline.search_knowledge(
                query=request.question,
                top_k=request.top_k,
                search_type=request.search_type
            )
            context = pipeline.build_context(search_results)
            prompt = pipeline.build_prompt(request.question, context)

            # 先发送检索结果
            yield f"data: {json.dumps({'type': 'sources', 'data': search_results}, ensure_ascii=False)}\n\n"
        else:
            prompt = pipeline.build_prompt(request.question)

        # 流式生成回答
        stream = pipeline.generate_answer(prompt, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'type': 'content', 'data': content}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


## 对话管理
class ConversationManager:
    """对话管理器"""

    def __init__(self):
        self.conversations = {}

    def create_session(self, session_id: str):
        """创建会话"""
        self.conversations[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息"""
        if session_id not in self.conversations:
            self.create_session(session_id)

        self.conversations[session_id].append({
            "role": role,
            "content": content
        })

    def get_history(self, session_id: str, limit: int = 10):
        """获取历史"""
        if session_id in self.conversations:
            return self.conversations[session_id][-limit:]
        return []

# 创建全局对话管理器
conversation_manager = ConversationManager()


@app.post("/chat/with-history")
async def chat_with_history(
    session_id: str,
    request: ChatRequest
):
    """带历史记录的对话"""

    # 获取历史
    history = conversation_manager.get_history(session_id)

    # RAG 检索
    pipeline = RAGPipeline()

    if request.use_rag:
        search_results = pipeline.search_knowledge(
            query=request.question,
            top_k=request.top_k,
            search_type=request.search_type
        )
        context = pipeline.build_context(search_results)
    else:
        search_results = None
        context = ""

    # 构建带历史的消息
    messages = [
        {"role": "system", "content": "你是一个专业、严谨的助手。"}
    ]

    # 添加历史
    messages.extend(history)

    # 添加当前问题
    if context:
        user_prompt = f"基于以下参考内容回答问题：\n\n{context}\n\n问题：{request.question}"
    else:
        user_prompt = request.question

    messages.append({"role": "user", "content": user_prompt})

    # 生成回答
    try:
        completion = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            stream=False
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"生成回答时出错：{str(e)}"

    # 保存到历史
    conversation_manager.add_message(session_id, "user", request.question)
    conversation_manager.add_message(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "answer": answer,
        "sources": search_results["results"][:3] if search_results else [],
        "history_length": len(conversation_manager.get_history(session_id))
    }


if __name__ == "__main__":
    # 这里可以进行一些测试，验证RAGPipeline的功能
    question = "人工智能创新趋势有哪些"
    intent_domain, confidence, reason = RAGPipeline.detect_intent_with_llm(question)
    print(f"识别领域: {intent_domain}, 置信度: {confidence}, 判断依据: {reason}")