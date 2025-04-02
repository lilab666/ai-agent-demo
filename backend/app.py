# app.py
from fastapi import FastAPI
from dotenv import load_dotenv
from backend.routers import upload, chat, status, files
from backend.services.knowledge_base import knowledge_base
import nltk

load_dotenv()

# 初始化NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI(title="智能知识库后端API")

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
