import json
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from tavily import TavilyClient

from .knowledge_base import knowledge_base
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

# 加载环境变量
load_dotenv()

# 初始化通义千问客户端（OpenAI 接口格式）
qwen_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# 添加系统提示
messages = [{"role": "system", "content": "你是一个严谨的智能助手，回答需基于提供的信息和历史消息"}]

def trim_messages(messages, max_length=12):
    while len(messages) > max_length:
        if len(messages) >= 3:
            del messages[1:3]
        else:
            break

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def chat_handler(request):
    full_response = ""
    docs = []
    metadata = []

    # 联网搜索调用 Tavily
    response = tavily_client.search(
        query=request.prompt,
        search_depth="advanced",
        max_results=10,
        include_answer="advanced"
    )

    search_answer = response.get("answer", "")

    try:
        if request.mode == "knowledge":
            if not knowledge_base.kb_ready:
                yield json.dumps({"type": "error", "data": "知识库未就绪，请先上传文档"}) + "\n"
                return

            query_embeddings = knowledge_base.embeddings.embed_query(request.prompt, dimensions=1024)

            results = knowledge_base.collection.query(
                query_embeddings=query_embeddings,
                n_results=request.k
            )

            docs = results['documents']
            metadata = results['metadatas'][0]

            context_kb = "\n\n".join(
                [f"【信息片段 {i + 1}】（来源：{meta['source']}）\n{doc}" for i, (doc, meta) in enumerate(zip(docs, metadata))]
            )

            context = f"【联网搜索摘要】\n{search_answer}\n\n【知识库内容】\n{context_kb}"
            final_prompt = f"请根据以下知识库提供片段(不一定符合问题需求，信息片段权重大于网络搜索片段)和之前的对话(当前问题很可能跟之前的内容关联)回答问题：\n{context}\n\n问题：{request.prompt}"
        else:
            final_prompt = f"【联网搜索摘要】(对问题不一定有用)\n{search_answer}\n\n问题：{request.prompt}"

        messages.append({"role": "user", "content": final_prompt})
        trim_messages(messages)

        # 使用通义千问（DashScope）调用
        response = qwen_client.chat.completions.create(
            model="qwen-turbo-2025-04-28",
            messages=messages,
            stream=True,
            # 如果你用的是 Qwen3 开源模型，可以取消下行注释
            # extra_body={"enable_thinking": False}
        )

        full_response = ""

        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                yield json.dumps({"type": "content", "data": content}) + "\n"

        messages.append({"role": "assistant", "content": full_response})

        yield json.dumps({"type": "sources", "data": [meta['source'] for meta in metadata]}) + "\n"

    except Exception as e:
        yield json.dumps({"type": "error", "data": f"生成回答失败：{str(e)}"}) + "\n"
