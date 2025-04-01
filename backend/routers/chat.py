# routers/chat.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse  # 导入 StreamingResponse
from ..models import ChatRequest
from ..services.chat import chat_handler

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    # 使用 StreamingResponse 包装 chat_handler，返回流式数据
    return StreamingResponse(chat_handler(request), media_type="application/x-ndjson")
