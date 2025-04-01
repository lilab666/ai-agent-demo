import shutil
import tempfile
import time
import os
from pathlib import Path
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import FAISS
from fastapi import APIRouter, File, UploadFile, HTTPException
from .embeddings import QwenEmbeddings
from .knowledge_base import knowledge_base  # 导入持久化的知识库实例
import json

# 设置资源文件夹路径
RESOURCES_DIR = Path(__file__).parent.parent.parent / "resources"
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# 设置元数据文件存储路径
METADATA_PATH = RESOURCES_DIR / "metadata.json"


def load_metadata():
    """加载现有的文档元数据"""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_metadata(metadata):
    """将文档元数据保存到 JSON 文件"""
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def process_files(
        files: List[UploadFile],
        qwen_api_key: str,
        qwen_base_url: str
) -> Tuple[List[Document], QwenEmbeddings, List[str]]:
    documents = []
    file_names = []
    temp_files = []
    metadata = load_metadata()  # 获取现有的文档元数据

    try:
        for file in files:
            safe_name = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
            save_path = RESOURCES_DIR / f"{int(time.time())}_{safe_name}"
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 重置文件指针以便后续处理
            file.file.seek(0)

            file_content = file.file.read()
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=f"_{file.filename}") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
                temp_files.append(tmp_path)

            # 根据文件类型选择不同的加载器
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension == ".pdf" or file.content_type == "application/pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_extension == ".txt" or file.content_type == "text/plain":
                loader = TextLoader(tmp_path)
            elif file_extension == ".docx":
                loader = UnstructuredLoader(tmp_path, mode="elements")
            else:
                print(f"跳过不支持的类型: 文件名[{file.filename}] 扩展名[{file_extension}] 服务端类型[{file.content_type}]")
                continue

            # 加载文档并更新元数据
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata.update({
                    "source": file.filename,
                    "file_type": file.content_type,
                    "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "file_size": f"{len(file_content) / 1024:.1f}KB"
                })

                # 存储文件元数据到列表
                file_metadata = {
                    "file_name": file.filename,
                    "file_type": file.content_type,
                    "upload_time": doc.metadata["upload_time"],
                    "file_size": doc.metadata["file_size"]
                }
                metadata.append(file_metadata)

            documents.extend(loaded_docs)
            file_names.append(file.filename)

        if not documents:
            raise ValueError("所有文件均无可提取内容")

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("文本分割后无有效内容")

        # 初始化嵌入模型
        embeddings = QwenEmbeddings(api_key=qwen_api_key, base_url=qwen_base_url)

        # 为每个文档生成嵌入并添加到文档的 metadata 字段中
        document_texts = [doc.page_content for doc in chunks]
        embeddings_list = embeddings.embed_documents(document_texts)

        # 将嵌入存储在 metadata 字段中
        for doc, embedding in zip(chunks, embeddings_list):
            doc.metadata["embedding"] = embedding  # 将嵌入存储在 metadata 字段中

        # 更新知识库并保存
        knowledge_base.update_knowledge_base(chunks, embeddings, file_names)

        # 保存更新后的元数据到 JSON 文件
        save_metadata(metadata)

        return chunks, embeddings, file_names

    finally:
        # 清理临时文件
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
