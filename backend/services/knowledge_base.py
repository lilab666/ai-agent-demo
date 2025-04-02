from pathlib import Path
import chromadb
import hashlib
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from .embeddings import QwenEmbeddings  # 导入嵌入模型
import os

load_dotenv()


class KnowledgeBase:
    def __init__(self, storage_path=None, embeddings=None):
        # 设置默认的向量存储路径（存储在 resources 文件夹中）
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent / "resources" / "knowledge_base"
        self.storage_path = storage_path

        self.embeddings = embeddings if embeddings else QwenEmbeddings(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL")
        )

        # 创建 Chroma 客户端，支持持久化路径
        self.client = chromadb.PersistentClient(path=str(self.storage_path))  # 使用指定的存储路径

        # 检查集合是否已存在，如果已存在则不再创建
        collection_name = "my_collection"
        # 使用 list_collections() 返回的对象的 name 属性进行比较
        # 获取集合名称列表并检查是否包含目标集合名称
        existing_collections = self.client.list_collections()
        if collection_name not in existing_collections:
            self.collection = self.client.create_collection(collection_name)
        else:
            self.collection = self.client.get_collection(collection_name)

        # 设置是否准备好知识库
        self.kb_ready = False
        self.kb_version = 0

    def update_knowledge_base(self, chunks: List[Document], embeddings, file_names: List[str], ids: List[str]):
        """更新知识库并将其保存到文件"""
        if not chunks:
            raise ValueError("chunks is empty")

        # 为每个文档生成嵌入并添加到文档的 metadata 字段中
        document_texts = [doc.page_content for doc in chunks]
        embeddings_list = embeddings.embed_documents(document_texts)

        # 将向量和元数据存储到 Chroma 集合中
        self.collection.add(
            ids=ids,  # 添加 IDs 参数
            embeddings=embeddings_list,  # 存储嵌入向量
            documents=[doc.page_content for doc in chunks],
            metadatas=[doc.metadata for doc in chunks]
        )

        self.collection.update(
            ids=ids,  # 添加 IDs 参数
            embeddings=embeddings_list,  # 存储嵌入向量
        )

        # 生成版本哈希值
        self.kb_version = self._generate_version_hash(file_names)
        self.save()

    def _generate_version_hash(self, file_names: List[str]) -> str:
        """生成版本哈希值"""
        return hashlib.sha256(",".join(sorted(file_names)).encode()).hexdigest()

    def save(self):
        """保存向量存储到磁盘"""
        print(f"知识库已保存到 {self.storage_path}")

    def load(self):
        """加载磁盘上的向量存储"""
        # 当集合已加载时标记 kb_ready 为 True
        self.kb_ready = True
        print(f"知识库已从 {self.storage_path} 加载")

    def get_version(self) -> int:
        """获取当前知识库版本"""
        return self.kb_version

    def list_files(self):
        """查看当前知识库中的所有文件和元数据"""
        if not self.kb_ready:
            print("知识库尚未加载！")
            return []
        return self.collection.get()  # 获取集合中的所有文档

    def delete_file(self, file_name: str, upload_time: str):
        """从知识库中删除指定的文件"""
        print(f"删除文件：{file_name}, 上传时间: {upload_time}")

        # 先根据 source 查找文件
        results = self.collection.get(where={"source": {"$eq": file_name}})

        # 获取文档 IDs 和元数据
        ids = results['ids']
        metadatas = results['metadatas']

        # 筛选出符合 upload_time 的文档
        to_delete_ids = []
        for i, doc_metadata in enumerate(metadatas):
            if doc_metadata.get('upload_time') == upload_time:  # 确保有 'upload_time' 字段
                to_delete_ids.append(ids[i])  # 添加符合条件的 ID 到删除列表

        if to_delete_ids:
            # 删除匹配的文件
            self.collection.delete(ids=to_delete_ids)
            print(f"文件已删除: {to_delete_ids}")
            return True
        else:
            print("没有找到符合条件的文件进行删除")
            return False


    def get_status(self):
        """获取当前知识库状态"""
        # 使用 Chroma 中的 count 方法来获取文档数量
        chunk_count = self.collection.count() if self.kb_ready else 0

        return {
            "ready": self.kb_ready,
            "version": self.kb_version,
            "chunk_count": chunk_count
        }


# 实例化一个全局知识库对象，供其他模块引用
knowledge_base = KnowledgeBase()
knowledge_base.load()  # 可选：启动时自动加载
