"""智能体卡片 API。"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.apps import AppCreate, AppUpdate
from backend.services import app_service


router = APIRouter(tags=["apps"])


@router.get("/app-stages", summary="获取智能体分类")
async def get_app_stages(db: Session = Depends(get_db)) -> list[str]:
    categories = app_service.list_app_categories(db)
    db.commit()
    return categories


@router.get("/apps", summary="获取智能体卡片")
async def get_apps(
    category: Optional[str] = Query(None, description="分类筛选"),
    stage_filter: Optional[str] = Query(None, description="兼容旧字段：教学阶段筛选"),
    enabled_only: bool = Query(True, description="是否只返回启用卡片"),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    apps = app_service.list_apps(
        db,
        category=category or stage_filter,
        enabled_only=enabled_only,
    )
    db.commit()
    return apps


@router.get("/apps/{app_id}", summary="获取单个智能体卡片")
async def get_app(app_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    app = app_service.get_app(db, app_id)
    if not app:
        raise HTTPException(status_code=404, detail="智能体不存在")
    return app_service.serialize_app(app)


@router.post("/apps", summary="创建智能体卡片；空请求体兼容旧版列表接口")
async def create_app(
    payload: Optional[AppCreate] = Body(None),
    stage_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    if payload is None:
        apps = app_service.list_apps(db, category=stage_filter, enabled_only=True)
        db.commit()
        return apps
    app = app_service.create_app(db, payload)
    db.commit()
    db.refresh(app)
    return app_service.serialize_app(app)


@router.put("/apps/{app_id}", summary="更新智能体卡片")
async def update_app(
    app_id: int,
    payload: AppUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    app = app_service.get_app(db, app_id)
    if not app:
        raise HTTPException(status_code=404, detail="智能体不存在")
    app = app_service.update_app(db, app, payload)
    db.commit()
    db.refresh(app)
    return app_service.serialize_app(app)


@router.delete("/apps/{app_id}", summary="删除智能体卡片")
async def delete_app(app_id: int, db: Session = Depends(get_db)) -> dict[str, bool]:
    app = app_service.get_app(db, app_id)
    if not app:
        raise HTTPException(status_code=404, detail="智能体不存在")
    app_service.delete_app(db, app)
    db.commit()
    return {"ok": True}

