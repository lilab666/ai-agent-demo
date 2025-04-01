# services/knowledge_base.py
from langchain_community.vectorstores import FAISS
import hashlib
from typing import List
from langchain.schema import Document


class KnowledgeBase:
    def __init__(self):
        self.vector_store = None
        self.kb_ready = False
        self.kb_version = 0

    def update_knowledge_base(self, chunks: List[Document], embeddings, file_names: List[str]):
        """更新知识库"""
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.kb_ready = True
        self.kb_version = self._generate_version_hash(file_names)

    def _generate_version_hash(self, file_names: List[str]) -> str:
        """生成知识库版本的哈希值"""
        return hashlib.sha256(",".join(sorted(file_names)).encode()).hexdigest()

    def get_version(self) -> int:
        """获取当前知识库版本"""
        return self.kb_version


# 实例化一个知识库对象
knowledge_base = KnowledgeBase()
