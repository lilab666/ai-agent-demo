from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from ..services.knowledge_base import knowledge_base  # 导入知识库实例

router = APIRouter()


# Pydantic模型用于输入验证
class FileMetadata(BaseModel):
    file_name: str
    upload_time: str


@router.get("/files", response_model=List[dict])
async def list_files():
    """
    获取当前知识库中所有文件的详细信息。
    """
    try:
        files = knowledge_base.list_files()  # 调用服务中的list_files方法
        if not files:
            raise HTTPException(status_code=404, detail="知识库为空，未找到文件")

        metadatas = files['metadatas']

        # 需要返回每个文件的元数据（文件名、文件类型、上传时间等）
        file_details = []
        seen_files = set()  # 用来追踪已添加的文件

        for file in metadatas:
            file_metadata = (
                file['source'], file['file_type'], file['upload_time'], file['file_size']
            )

            # 使用元数据中的几个字段作为唯一标识符
            if file_metadata not in seen_files:  # 如果该文件元数据未出现过
                seen_files.add(file_metadata)  # 将其加入到已处理的集合中
                file_details.append({
                    "file_name": file['source'],  # file_name
                    "file_type": file['file_type'],  # file_type
                    "upload_time": file['upload_time'],  # upload_time
                    "file_size": file['file_size'],  # file_size
                })

        return file_details

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件失败: {str(e)}")


@router.delete("/files", status_code=200)
async def delete_file(file_metadata: FileMetadata):
    """
    从知识库中删除指定的文件。
    根据文件名、文件类型和上传时间删除文件。
    """
    try:

        success = knowledge_base.delete_file(file_metadata.file_name, file_metadata.upload_time)
        if not success:
            raise HTTPException(status_code=404, detail="未找到匹配的文件")
        return {"status": "success", "message": "文件已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")
