# app.py
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from backend.routers import upload, chat, status, files
from backend.services.knowledge_base import knowledge_base
import nltk

# 自动添加项目根目录到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

# 初始化NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI(title="智能知识库后端API")

# ⭐ 添加跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以换成 ["http://localhost:3000"] 指定前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化知识库
# knowledge_base = KnowledgeBase()

# 添加路由
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(status.router)
app.include_router(files.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
