# models.py
from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str  # 用户输入的问题
    mode: str = "knowledge"  # 默认为 "knowledge"，可以是 "knowledge" 或其他模式
    k: int = 3  # 默认为 3，表示返回的结果数量
