# routers/status.py
from fastapi import APIRouter
from ..services.knowledge_base import knowledge_base

router = APIRouter()


@router.get("/status")
async def get_status():
    return {
        "ready": knowledge_base.kb_ready,
        "version": knowledge_base.kb_version,
        "chunk_count": knowledge_base.vector_store.index.ntotal if knowledge_base.kb_ready else 0
    }
