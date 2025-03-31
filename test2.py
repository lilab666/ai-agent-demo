# 请先安装依赖：pip install streamlit openai langchain-community faiss-cpu pypdf python-docx pdfminer.six docx2txt tiktoken

import streamlit as st
from openai import OpenAI
import os
import tempfile
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import List, Union, Optional
from langchain.embeddings.base import Embeddings
from langchain.document_loaders import UnstructuredFileLoader

# LangChain 组件
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# 初始化配置
st.set_page_config(
    page_title="智能知识库助手",
    page_icon="🤖",
    layout="centered"
)


# 自定义嵌入模型（已修复批量限制问题）
class QwenEmbeddings(Embeddings):
    """通义千问专用嵌入模型（支持分批处理）"""

    def __init__(self, api_key: str, model: str = "text-embedding-v3", batch_size: int = 10):
        if not api_key:
            raise ValueError("API key is required for QwenEmbeddings")
        self.client = OpenAI(
            api_key=api_key,
            base_url=st.secrets["QWEN_BASE_URL"]
        )
        self.model = model
        self.batch_size = min(batch_size, 10)  # 强制遵守API限制

    def embed_documents(self, texts: List[str], dimensions: int = 1024) -> List[List[float]]:
        """分批处理文档嵌入"""
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
        """嵌入单个查询"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding


# 初始化客户端
@st.cache_resource
def get_openai_client():
    return OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url=st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/beta")
    )


client = get_openai_client()


# ==================== 状态管理 ====================
def init_session_state():
    """初始化会话状态"""
    defaults = {
        "messages": [{"role": "system", "content": "你是一个智能助手"}],
        "vector_store": None,
        "kb_ready": False,
        "kb_version": 0,
        "last_search_time": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ==================== 文件处理 ====================
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def load_document(file_path: str, file_type: str) -> List[Document]:
    """加载单个文档"""
    loader_map = {
        "application/pdf": PyPDFLoader,
        "text/plain": TextLoader,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader
    }
    loader_class = loader_map.get(file_type)
    if not loader_class:
        raise ValueError(f"不支持的文件类型: {file_type}")
    return loader_class(file_path).load()


def process_files(uploaded_files):
    """处理上传文件（已修复元数据冲突）"""
    try:
        # 预检查NLTK资源
        try:
            from nltk import data
            data.find('tokenizers/punkt')
        except LookupError:
            import nltk
            nltk.download('punkt')

        documents = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            try:
                file_type = file.type
                if file_type == "application/pdf":
                    loader = PyPDFLoader(tmp_path)
                elif file_type == "text/plain":
                    loader = TextLoader(tmp_path)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    loader = UnstructuredFileLoader(tmp_path, mode="elements")
                else:
                    st.error(f"不支持的文件类型: {file_type}")
                    continue

                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({
                        "source": file.name,
                        "file_type": file_type,
                        "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                documents.extend(loaded_docs)
            except Exception as e:
                st.error(f"处理文件 {file.name} 失败: {str(e)}")
                if "punkt" in str(e):
                    st.error("请运行: python -m nltk.downloader punkt")
            finally:
                os.unlink(tmp_path)

        if not documents:
            st.error("没有可处理的文档内容")
            return

        # 文本分块处理
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            st.error("文档分割后无有效内容")
            return

        # 生成向量存储（已修复批量限制）
        with st.spinner("正在生成知识库..."):
            api_key = st.secrets["QWEN_API_KEY"]
            embeddings = QwenEmbeddings(
                api_key=api_key,
                batch_size=10  # 显式设置批处理大小
            )

            st.session_state.vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.kb_ready = True
            st.session_state.kb_version = hash(tuple(f.name for f in uploaded_files))
            st.success("知识库创建成功！")

    except Exception as e:
        st.error(f"创建知识库失败: {str(e)}")
        st.session_state.vector_store = None
        st.session_state.kb_ready = False


# ==================== 检索功能 ====================
def is_kb_valid() -> bool:
    """验证知识库是否有效"""
    return bool(
        st.session_state.kb_ready and
        st.session_state.vector_store and
        hasattr(st.session_state.vector_store, "similarity_search")
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def safe_retrieval(query: str, k: int = 3) -> List[Document]:
    """带错误处理的检索方法"""
    if not is_kb_valid():
        raise ValueError("知识库未就绪")

    start_time = time.time()
    docs = st.session_state.vector_store.similarity_search(query, k=k)
    st.session_state.last_search_time = time.time() - start_time
    return docs


# ==================== 界面组件 ====================
def show_sidebar():
    """侧边栏组件"""
    with st.sidebar:
        st.header("📚 知识库管理")

        # 文件上传
        uploaded_files = st.file_uploader(
            "上传文档（PDF/TXT/DOCX）",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True
        )

        if uploaded_files:
            process_files(uploaded_files)

        # 知识库状态
        if st.session_state.kb_ready:
            st.success("✅ 知识库已就绪")
            try:
                st.caption(f"知识片段数量: {st.session_state.vector_store.index.ntotal}")
            except:
                pass

            if st.button("🔄 清除知识库"):
                st.session_state.vector_store = None
                st.session_state.kb_ready = False
                st.session_state.kb_version = 0
                st.rerun()

        # 性能监控
        if st.session_state.last_search_time:
            st.divider()
            st.metric("上次检索耗时", f"{st.session_state.last_search_time:.2f}s")


def chat_interface():
    """主聊天界面"""
    st.title("💬 智能知识库助手")

    # 模式切换
    mode = st.radio(
        "模式选择",
        ["💬 普通聊天", "📚 知识库问答"],
        horizontal=True,
        index=1 if st.session_state.kb_ready else 0
    )

    # 显示历史消息
    for msg in st.session_state.messages[1:]:  # 跳过系统消息
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入处理
    if prompt := st.chat_input("请输入您的问题："):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 准备响应
        with st.chat_message("assistant"):
            try:
                # 知识库模式处理
                if mode == "📚 知识库问答":
                    if not is_kb_valid():
                        st.error("知识库未就绪，请先上传文档")
                        st.stop()

                    # 检索上下文
                    try:
                        docs = safe_retrieval(prompt, k=3)
                        context = "\n\n".join(
                            [f"**[来源：{doc.metadata['source']}]**\n{doc.page_content}"
                             for doc in docs]
                        )
                        enhanced_prompt = f"请根据以下信息回答问题：\n{context}\n\n问题：{prompt}"
                    except Exception as e:
                        st.error(f"检索失败: {str(e)}")
                        st.stop()
                else:
                    enhanced_prompt = prompt

                # 流式输出
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.messages[:-1]] +
                             [{"role": "user", "content": enhanced_prompt}],
                    stream=True
                )

                full_response = ""
                placeholder = st.empty()
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                # 更新消息历史
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")


# ==================== 主程序 ====================
show_sidebar()
chat_interface()
