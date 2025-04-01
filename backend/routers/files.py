# routers/files.py
from fastapi import APIRouter
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent / "resources"

router = APIRouter()


@router.get("/files")
async def list_files():
    return {
        "files": [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            }
            for f in RESOURCES_DIR.glob("*")
            if f.is_file()
        ]
    }
