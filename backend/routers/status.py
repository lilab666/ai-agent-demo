from fastapi import APIRouter
from ..services.knowledge_base import knowledge_base

router = APIRouter()


@router.get("/status")
async def get_status():
    # 获取知识库状态
    status = knowledge_base.get_status()  # 调用 get_status 方法
    return status
