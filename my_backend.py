# my_backend.py
# pip install fastapi uvicorn python-multipart langchain_community openai python-docx pdfminer.six nltk
import shutil
from pathlib import Path
import json
import os
import tempfile
import time
import hashlib
from typing import List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tenacity import retry, stop_after_attempt, wait_fixed

# LangChain 组件
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import nltk

# 初始化NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI(title="智能知识库后端API")

load_dotenv()


class QwenEmbeddings(Embeddings):
    """通义千问专用嵌入模型"""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v3", batch_size: int = 10):
        if not api_key:
            raise ValueError("API key is required")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.batch_size = min(batch_size, 10)

    def embed_documents(self, texts: List[str], dimensions: int = 1024) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=dimensions,
                encoding_format="float"
            )
            embeddings.extend([data.embedding for data in response.data])
        return embeddings

    def embed_query(self, text: str, dimensions: int = 1024) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding


class KnowledgeBase:
    def __init__(self):
        self.vector_store = None
        self.kb_ready = False
        self.kb_version = 0

    def update_knowledge_base(self, chunks, embeddings, file_names):
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.kb_ready = True
        self.kb_version = self._generate_version_hash(file_names)

    def _generate_version_hash(self, file_names):
        return hashlib.sha256(",".join(sorted(file_names)).encode()).hexdigest()


knowledge_base = KnowledgeBase()

# 在 process_files 函数前定义资源目录
RESOURCES_DIR = Path(__file__).parent / "resources"
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在


def process_files(
        files: List[UploadFile],
        qwen_api_key: str,
        qwen_base_url: str
) -> Tuple[List[Document], QwenEmbeddings, List[str]]:
    """处理上传文件并生成知识库所需数据（单体式强化版）"""
    documents = []
    file_names = []
    temp_files = []  # 跟踪创建的临时文件

    try:
        # ================== 文件加载阶段 ==================
        for file in files:

            # ============ 新增文件保存逻辑 ============
            # 生成安全文件名
            safe_name = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
            save_path = RESOURCES_DIR / f"{int(time.time())}_{safe_name}"

            # 永久保存文件到 resources 目录
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 重置文件指针以便后续处理
            file.file.seek(0)
            # ============ 结束新增逻辑 ============

            file_content = file.file.read()

            # 创建安全临时文件
            with tempfile.NamedTemporaryFile(
                    mode="wb",
                    delete=False,
                    suffix=f"_{file.filename}"
            ) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
                temp_files.append(tmp_path)

            print(f"[DEBUG] Filename: {file.filename}, Content-Type: {file.content_type}")

            # 根据文件类型选择加载器
            try:

                # 修改原有的类型判断部分
                file_extension = os.path.splitext(file.filename)[1].lower()

                # 强化的类型判断逻辑
                if file_extension == ".pdf" or file.content_type == "application/pdf":
                    loader = PyPDFLoader(tmp_path)
                elif file_extension == ".txt" or file.content_type == "text/plain":
                    loader = TextLoader(tmp_path)
                elif file_extension == ".docx":
                    # 显式指定DOCX类型（即使content_type为空）
                    loader = UnstructuredFileLoader(tmp_path, mode="elements")
                else:
                    print(f"跳过不支持的类型: 文件名[{file.filename}] 扩展名[{file_extension}] 服务端类型[{file.content_type}]")
                    continue

                # 加载文档并增强元数据
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({
                        "source": file.filename,
                        "file_type": file.content_type,
                        "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "file_size": f"{len(file_content) / 1024:.1f}KB"
                    })

                documents.extend(loaded_docs)
                file_names.append(file.filename)

            except Exception as load_error:
                raise RuntimeError(
                    f"文件 [{file.filename}] 加载失败: {str(load_error)}"
                ) from load_error

        # ================== 内容验证阶段 ==================
        if not documents:
            raise ValueError("所有文件均无可提取内容")

        # ================== 文本分割阶段 ==================
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("文本分割后无有效内容")

        # ================== 嵌入初始化阶段 ==================
        embeddings = QwenEmbeddings(
            api_key=qwen_api_key,
            base_url=qwen_base_url,
            batch_size=10
        )

        # 预验证嵌入服务可用性
        if not embeddings.client.models.list().data:
            raise ConnectionError("嵌入服务连接失败")

        return chunks, embeddings, file_names

    finally:
        # ================== 资源清理阶段 ==================
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as clean_error:
                print(f"清理临时文件失败: {tmp_path} - {str(clean_error)}")


class ChatRequest(BaseModel):
    prompt: str
    mode: str = "knowledge"
    k: int = 3


class FileUploadResponse(BaseModel):
    status: str
    chunk_count: int
    version: str


class KnowledgeBase:
    def __init__(self):
        self.vector_store = None
        self.kb_version = 0

    def update(self, chunks, embeddings, filenames):
        if self.vector_store:
            self.vector_store.merge_from(FAISS.from_documents(chunks, embeddings))
        else:
            self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.kb_version = hash(tuple(filenames))


@app.post("/upload")
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def upload_endpoint(
        files: List[UploadFile] = File(...),
        api_key: str = os.getenv("DASHSCOPE_API_KEY"),
        base_url: str = os.getenv("DASHSCOPE_BASE_URL")
):
    try:
        chunks, embeddings, filenames = process_files(files, api_key, base_url)
        knowledge_base.update_knowledge_base(chunks, embeddings, filenames)
        return JSONResponse(
            content={
                "status": "success",
                "chunk_count": len(chunks),
                "version": knowledge_base.kb_version
            }
        )
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(ve)}
        )
    except ConnectionError as ce:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "message": str(ce)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"系统错误: {str(e)}"}
        )


# DeepSeek客户端初始化
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


@app.post("/chat")
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def chat_handler(request: ChatRequest):
    def generate():
        full_response = ""
        docs = []
        try:
            # 知识库模式预处理
            if request.mode == "knowledge":
                if not knowledge_base.kb_ready:
                    yield json.dumps({
                        "type": "error",
                        "data": "知识库未就绪，请先上传文档"
                    }) + "\n"
                    return

                docs = knowledge_base.vector_store.similarity_search(
                    request.prompt,
                    k=request.k
                )
                context = "\n\n".join(
                    [f"【信息片段 {i + 1}】（来源：{doc.metadata['source']}）\n{doc.page_content}"
                     for i, doc in enumerate(docs)]
                )
                messages = [{
                    "role": "user",
                    "content": f"请严格根据以下信息回答问题：\n{context}\n\n问题：{request.prompt}"
                }]
            else:
                messages = [{"role": "user", "content": request.prompt}]

            # 添加系统提示
            messages.insert(0, {
                "role": "system",
                "content": "你是一个严谨的智能助手，回答需基于提供的信息"
            })

            # 流式调用DeepSeek
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True
            )

            # 流式返回内容
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                yield json.dumps({
                    "type": "content",
                    "data": content
                }) + "\n"

            # 返回来源信息
            yield json.dumps({
                "type": "sources",
                "data": [doc.metadata["source"] for doc in docs]
            }) + "\n"

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "data": f"生成回答失败：{str(e)}"
            }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/status")
async def get_status():
    return {
        "ready": knowledge_base.kb_ready,
        "version": knowledge_base.kb_version,
        "chunk_count": knowledge_base.vector_store.index.ntotal if knowledge_base.kb_ready else 0
    }


@app.get("/files")
async def list_files():
    """列出已保存文件"""
    return {
        "files": [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            }
            for f in RESOURCES_DIR.glob("*")
            if f.is_file()
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
