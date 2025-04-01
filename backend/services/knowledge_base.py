from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
from pathlib import Path
import faiss
import hashlib
import sqlite3  # 导入SQLite库
from typing import List
from .embeddings import QwenEmbeddings  # 导入嵌入模型
import numpy as np

load_dotenv()


class KnowledgeBase:
    def __init__(self, storage_path=None, db_path=None, embeddings=None):
        # 设置默认的向量存储路径（存储在 resources 文件夹中）
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent / "resources" / "knowledge_base.index"
        self.storage_path = storage_path

        # 设置默认的 SQLite 数据库路径（存储在 resources_db 文件夹中）
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "resources_db" / "metadata.db"
        self.db_path = db_path

        self.vector_store = None
        self.kb_ready = False
        self.kb_version = 0
        self.embeddings = embeddings if embeddings else QwenEmbeddings(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL")
        )

        self.create_db()  # 创建SQLite数据库
        self.load()  # 尝试加载向量数据库

    def create_db(self):
        """创建SQLite数据库表"""
        if not os.path.exists(self.db_path):
            # 确保数据库目录存在
            db_dir = Path(self.db_path).parent
            if not db_dir.exists():
                db_dir.mkdir(parents=True)  # 创建数据库文件夹
                print(f"创建数据库文件夹：{db_dir}")

            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(''' 
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT,
                    file_type TEXT,
                    upload_time TEXT,
                    file_size TEXT,
                    vector_id INTEGER
                ) 
            ''')
            conn.commit()
            conn.close()
            print(f"创建SQLite数据库：{self.db_path}")
        else:
            print(f"SQLite数据库已存在：{self.db_path}")

    def insert_metadata(self, file_metadata):
        """插入文档元数据到SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(''' 
            INSERT INTO documents (file_name, file_type, upload_time, file_size, vector_id) 
            VALUES (?, ?, ?, ?, ?) 
        ''', (file_metadata['file_name'], file_metadata['file_type'], file_metadata['upload_time'],
              file_metadata['file_size'], file_metadata['vector_id']))
        conn.commit()
        conn.close()

    def update_knowledge_base(self, chunks: List[Document], embeddings, file_names: List[str]):
        """更新知识库并将其保存到文件"""
        # 获取嵌入维度（从 metadata 字段中获取嵌入）
        if not chunks:
            raise ValueError("chunks is empty")

        embedding_dim = len(chunks[0].metadata["embedding"])  # 从 metadata 中获取嵌入的维度

        # 使用 IndexIDMap 包装 IndexFlat 索引，支持删除操作
        index = faiss.IndexFlatL2(embedding_dim)  # 获取嵌入维度
        self.vector_store = faiss.IndexIDMap(index)  # 将其包装成支持删除的索引

        # 添加向量到索引
        vectors = np.array([doc.metadata["embedding"] for doc in chunks], dtype=np.float32)  # 从 metadata 中获取嵌入
        ids = np.arange(len(chunks))  # 使用文档的索引作为ID
        self.vector_store.add_with_ids(vectors, ids)

        self.kb_ready = True
        self.kb_version = self._generate_version_hash(file_names)
        self.save()  # 保存向量存储

        # 更新SQLite数据库
        for doc_id, doc in enumerate(chunks):
            file_metadata = {
                "file_name": doc.metadata.get("source"),
                "file_type": doc.metadata.get("file_type"),
                "upload_time": doc.metadata.get("upload_time"),
                "file_size": doc.metadata.get("file_size"),
                "vector_id": doc_id  # 向量ID
            }
            self.insert_metadata(file_metadata)  # 插入到数据库

    def _generate_version_hash(self, file_names: List[str]) -> str:
        """生成版本哈希值"""
        return hashlib.sha256(",".join(sorted(file_names)).encode()).hexdigest()

    def save(self):
        """保存向量存储到磁盘"""
        if self.vector_store:
            # 确保保存路径存在
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path)
                print(f"创建目录 {self.storage_path}")

            faiss.write_index(self.vector_store, str(self.storage_path))
            print(f"知识库已保存到 {self.storage_path}")
        else:
            print("没有知识库可以保存！")

    def load(self):
        """加载磁盘上的向量存储"""
        if os.path.exists(self.storage_path):
            try:
                index = faiss.read_index(str(self.storage_path))
                self.vector_store = faiss.IndexIDMap(index)
                self.kb_ready = True
                print(f"知识库已从 {self.storage_path} 加载")
            except Exception as e:
                print(f"加载知识库时出错: {e}")
        else:
            print(f"未找到现有知识库文件：{self.storage_path}")

    def get_version(self) -> int:
        """获取当前知识库版本"""
        return self.kb_version

    def list_files(self):
        """查看当前知识库中的所有文件和元数据"""
        if not self.kb_ready:
            print("知识库尚未加载！")
            return []

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM documents')
        files = c.fetchall()
        conn.close()

        return files  # 返回SQLite查询的文件元数据

    def delete_file(self, file_name: str, upload_time: str):
        """从知识库中删除指定的文件"""
        if not self.kb_ready:
            print("知识库尚未加载！")
            return False

        # 从 SQLite 获取文件的 vector_id
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(''' 
            SELECT vector_id FROM documents 
            WHERE file_name = ? AND upload_time = ? 
        ''', (file_name, upload_time))

        vector_ids = c.fetchall()

        if not vector_ids:
            print(f"未找到匹配的文件元数据：{file_name}, 上传时间: {upload_time}")
            return False

        # 从 FAISS 删除文档
        doc_ids_to_delete = [vector_id[0] for vector_id in vector_ids]

        # 删除指定 ID 的向量
        doc_ids_to_delete = np.array(doc_ids_to_delete, dtype=np.int64)
        self.vector_store.remove_ids(doc_ids_to_delete)

        self.save()  # 保存 FAISS 更新后的状态
        print(f"已从 FAISS 删除文件：{file_name}, 上传时间: {upload_time}")

        # 从 SQLite 删除文档元数据
        c.execute(''' 
            DELETE FROM documents 
            WHERE file_name = ? AND upload_time = ? 
        ''', (file_name, upload_time))
        conn.commit()

        if c.rowcount == 0:
            print(f"未找到匹配的文件元数据：{file_name}, 上传时间: {upload_time}")
            return False

        conn.close()
        print(f"已从 SQLite 删除文件元数据：{file_name}, 上传时间: {upload_time}")
        return True


# 实例化一个知识库对象
knowledge_base = KnowledgeBase(
    embeddings=QwenEmbeddings(api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_BASE_URL"))
)
