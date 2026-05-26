"""教学大纲智能体 API。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.outline import OutlineGenerateRequest, OutlineListResponse, OutlineResponse
from backend.services.outline_service import PPTX_MIME, XLSX_MIME, OutlineService
from backend.services.runtime_service import list_runtime_logs


router = APIRouter(prefix="/outline", tags=["outline"])
service = OutlineService()


@router.post("/generate", response_model=OutlineResponse, summary="生成教学大纲")
async def generate_outline(
    payload: OutlineGenerateRequest,
    db: Session = Depends(get_db),
) -> OutlineResponse:
    try:
        return service.generate_outline(db, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"教学大纲生成失败: {exc}") from exc


@router.get("", response_model=OutlineListResponse, summary="查询历史大纲列表")
async def list_outlines(
    conversation_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> OutlineListResponse:
    return OutlineListResponse(items=service.list_outlines(db, conversation_id=conversation_id, limit=limit))


@router.get("/{outline_id}", response_model=OutlineResponse, summary="查询单个大纲")
async def get_outline(outline_id: int, db: Session = Depends(get_db)) -> OutlineResponse:
    outline = service.get_outline(db, outline_id)
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    return outline


@router.get("/{outline_id}/logs", summary="查询大纲执行日志")
async def get_outline_logs(outline_id: int, db: Session = Depends(get_db)):
    outline = service.get_outline(db, outline_id)
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    return list_runtime_logs(db, outline_id=outline_id)


@router.get("/{outline_id}/download/xlsx", summary="下载 XLSX 文件")
async def download_xlsx(outline_id: int, db: Session = Depends(get_db)) -> FileResponse:
    outline = service.get_outline(db, outline_id)
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")

    path = Path(outline.xlsx_file_path) if outline.xlsx_file_path else service.generate_xlsx(db, outline)
    if not path.exists():
        path = service.generate_xlsx(db, outline)
    return FileResponse(
        path,
        media_type=XLSX_MIME,
        filename=path.name,
    )


@router.post("/{outline_id}/generate-pptx", response_model=OutlineResponse, summary="根据大纲生成 PPTX")
async def generate_pptx(outline_id: int, db: Session = Depends(get_db)) -> OutlineResponse:
    outline = service.get_outline(db, outline_id)
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    try:
        service.generate_pptx(db, outline)
        return outline
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PPTX 生成失败: {exc}") from exc


@router.get("/{outline_id}/download/pptx", summary="下载 PPTX 文件")
async def download_pptx(outline_id: int, db: Session = Depends(get_db)) -> FileResponse:
    outline = service.get_outline(db, outline_id)
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    if not outline.pptx_file_path:
        raise HTTPException(status_code=404, detail="PPTX 尚未生成")

    path = Path(outline.pptx_file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="PPTX 文件不存在，请重新生成")
    return FileResponse(
        path,
        media_type=PPTX_MIME,
        filename=path.name,
    )
