# services/knowledge_base.py
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import hashlib
from typing import List
from langchain.schema import Document
import os
from pathlib import Path
from .embeddings import QwenEmbeddings  # 导入嵌入模型

load_dotenv()


class KnowledgeBase:
    def __init__(self, storage_path=None, embeddings=None):
        # 设置默认的存储路径为项目根目录下的 resources 文件夹
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent / "resources" / "knowledge_base.index"

        self.vector_store = None
        self.kb_ready = False
        self.kb_version = 0
        self.storage_path = storage_path  # 存储文件的路径
        self.embeddings = embeddings if embeddings else QwenEmbeddings(api_key=os.getenv("QWEN_API_KEY"),
                                                                       base_url=os.getenv("QWEN_BASE_URL"))  # 确保 embeddings 存在
        self.load()  # 尝试加载现有的知识库

    def update_knowledge_base(self, chunks: List[Document], embeddings, file_names: List[str]):
        """更新知识库并将其保存到文件"""
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.kb_ready = True
        self.kb_version = self._generate_version_hash(file_names)
        self.save()  # 保存到文件

    def _generate_version_hash(self, file_names: List[str]) -> str:
        """生成知识库版本的哈希值"""
        return hashlib.sha256(",".join(sorted(file_names)).encode()).hexdigest()

    def save(self):
        """保存向量存储到磁盘"""
        if self.vector_store:
            # 确保保存的路径正确
            self.vector_store.save_local(self.storage_path)
            print(f"知识库已保存到 {self.storage_path}")
        else:
            print("没有知识库可以保存！")

    def load(self):
        """加载磁盘上的向量存储"""
        if os.path.exists(self.storage_path):
            try:
                # 显式传递 allow_dangerous_deserialization=True
                self.vector_store = FAISS.load_local(self.storage_path, embeddings=self.embeddings, allow_dangerous_deserialization=True)
                self.kb_ready = True
                print(f"知识库已从 {self.storage_path} 加载")
            except Exception as e:
                print(f"加载知识库时出错: {e}")
        else:
            print(f"未找到现有知识库文件：{self.storage_path}")

    def get_version(self) -> int:
        """获取当前知识库版本"""
        return self.kb_version


# 实例化一个知识库对象
knowledge_base = KnowledgeBase(
    embeddings=QwenEmbeddings(api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_BASE_URL")))
