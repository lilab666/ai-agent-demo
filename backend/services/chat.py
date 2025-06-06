import json
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from .knowledge_base import knowledge_base
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

# 加载环境变量
load_dotenv()

# 初始化 DeepSeek 客户端
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"))

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def chat_handler(request):  # 使用 Pydantic 模型 ChatRequest
    full_response = ""
    docs = []
    metadata = []
    try:
        # 知识库模式预处理
        if request.mode == "knowledge":
            if not knowledge_base.kb_ready:
                yield json.dumps({"type": "error", "data": "知识库未就绪，请先上传文档"}) + "\n"
                return  # 显式结束生成器

            # 生成查询的嵌入向量，确保是 1024 维
            query_embeddings = knowledge_base.embeddings.embed_query(request.prompt, dimensions=1024)

            # 在 Chroma 中执行相似度搜索
            results = knowledge_base.collection.query(
                query_embeddings=query_embeddings,  # 传递正确的嵌入向量
                n_results=request.k  # 返回前 k 个相似的结果
            )

            # 获取匹配的文档和元数据
            docs = results['documents']  # 提取匹配的文档
            metadata = results['metadatas'][0]  # 提取文档的元数据
            context = "\n\n".join(
                [f"【信息片段 {i + 1}】（来源：{meta['source']}）\n{doc}" for i, (doc, meta) in enumerate(zip(docs, metadata))]
            )
            messages = [{"role": "user", "content": f"请严格根据以下信息回答问题：\n{context}\n\n问题：{request.prompt}"}]
        else:
            messages = [{"role": "user", "content": request.prompt}]

        # 添加系统提示
        messages.insert(0, {"role": "system", "content": "你是一个严谨的智能助手，回答需基于提供的信息"})

        # 调用 DeepSeek 生成响应
        response = deepseek_client.chat.completions.create(model="deepseek-chat", messages=messages, stream=True)

        # 流式返回内容
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            yield json.dumps({"type": "content", "data": content}) + "\n"

        # 返回来源信息
        yield json.dumps({"type": "sources", "data": [meta['source'] for meta in metadata]}) + "\n"

    except Exception as e:
        yield json.dumps({"type": "error", "data": f"生成回答失败：{str(e)}"}) + "\n"
