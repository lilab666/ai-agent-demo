# routers/upload.py
from dotenv import load_dotenv
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Tuple
from ..services.file_processing import process_files
from ..services.knowledge_base import knowledge_base
from tenacity import retry, stop_after_attempt, wait_fixed
import os

load_dotenv()

router = APIRouter()


@router.post("/upload")
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def upload_endpoint(
        files: List[UploadFile] = File(...),
        api_key: str = os.getenv("QWEN_API_KEY"),
        base_url: str = os.getenv("QWEN_BASE_URL")
):
    try:
        chunks, embeddings, filenames = process_files(files, api_key, base_url)
        knowledge_base.update_knowledge_base(chunks, embeddings, filenames)
        return JSONResponse(
            content={
                "status": "success",
                "chunk_count": len(chunks),
                "version": knowledge_base.kb_version
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"系统错误: {str(e)}"}
        )
