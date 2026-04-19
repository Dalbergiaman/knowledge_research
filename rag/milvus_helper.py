import re
import jieba
import jieba.analyse
from openai import OpenAI
import milvus_config as config
from fastapi import HTTPException
from typing import List

import warnings
# 忽略 jieba 库产生的 SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning, module="jieba")

# 创建embedding client
embedding_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.EMBEDDING_BASE_URL
)


def generate_embedding(text: str) -> List[float]:
    """生成文本的向量表示"""
    try:
        response = embedding_client.embeddings.create(
            input=text,
            model=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSION,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成embedding失败: {str(e)}")
    
# 使用jieba提取关键词
def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """使用jieba提取文本关键词"""
    keywords = jieba.analyse.extract_tags(text, topK=top_k)
    return ' '.join(keywords)

# 切割markdown文本
def split_markdown_content(content: str, source: str):
    """切分 Markdown 内容"""
    chunks = []

    # 按 ## 切分
    h2_sections = re.split(r'\n## ', content)

    for i, h2_section in enumerate(h2_sections):

        # 第一段是前言，可能没有 ## 标题, 比较标准的是# 开头，但也有可能直接是正文内容
        if i == 0 and not h2_section.startswith('## '):
            if h2_section.strip():
                chunks.append({
                    'text': h2_section.strip(),
                    'source': source,
                    'section': '前言'
                })
            continue
        

        lines = h2_section.split('\n')
        h2_title = lines[0].strip()  # 用于记录chunk的共同标题
        h2_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""

        if '### ' in h2_content:
            h3_sections = re.split(r'\n### ', h2_content)

            for j, h3_section in enumerate(h3_sections):
                if j == 0 and h3_section.strip():
                    chunks.append({
                        'text': f"## {h2_title}\n{h3_section.strip()}",  # 每次注入标题，保证上下文完整
                        'source': source,
                        'section': h2_title
                    })
                elif j > 0:
                    h3_lines = h3_section.split('\n')
                    h3_title = h3_lines[0].strip()
                    h3_content = '\n'.join(h3_lines[1:]) if len(h3_lines) > 1 else ""

                    chunks.append({
                        'text': f"## {h2_title}\n### {h3_title}\n{h3_content.strip()}",
                        'source': source,
                        'section': f"{h2_title} - {h3_title}"
                    })
        else:
            chunks.append({
                'text': f"## {h2_title}\n{h2_content.strip()}",
                'source': source,
                'section': h2_title
            })

    return chunks
    
    

if __name__ == "__main__":
    # sample_text = "人工智能正在改变世界，尤其是在自然语言处理领域。"
    # # embedding = generate_embedding(sample_text)
    # keywords = extract_keywords(sample_text, top_k=5)
    # # print("Embedding:", embedding)
    # print("Keywords:", keywords)

    with open('rag/test_doc/fin.md', 'r', encoding='utf-8') as f:
        content = f.read()
        chunks = split_markdown_content(content, source='fin.md')
        for chunk in chunks:
            print("Section:", chunk['section'])
            print("Text:", chunk['text'][:100])  # 只打印前100个字符
            print("-" * 50)