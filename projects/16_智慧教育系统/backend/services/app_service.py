"""智能体应用卡片服务。"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from backend.business.apps import RAW_APPS_DATA
from backend.models.app import AppModel
from backend.schemas.apps import AppCreate, AppUpdate


APP_CODE_MAP = {
    1: "outline_agent",
    2: "preview_content_agent",
    3: "courseware_agent",
    4: "class_interaction_agent",
    5: "knowledge_explanation_agent",
    6: "focus_monitor_agent",
    7: "qa_agent",
    8: "homework_grading_agent",
    9: "mistake_notebook_agent",
    10: "lab_report_comment_agent",
    11: "subject_tutor_agent",
    12: "learning_planner_agent",
    13: "exam_paper_agent",
    14: "question_quality_agent",
}


def seed_default_apps(db: Session) -> None:
    """把现有卡片数据写入 MySQL，保留为数据库初始化种子。"""
    exists = db.query(AppModel.id).first()
    if exists:
        return

    for item in RAW_APPS_DATA:
        app_id = item["id"]
        app = AppModel(
            app_code=APP_CODE_MAP.get(app_id, f"agent_{app_id}"),
            title=item["title"],
            description=item["description"],
            icon=item.get("icon", "FileText"),
            icon_bg=item.get("iconBg", "bg-blue-600"),
            icon_tone=item.get("iconTone", "blue"),
            tag=item.get("tag", "其他"),
            tag_color=item.get("tagColor", "bg-slate-100 text-slate-600"),
            tag_tone=item.get("tagTone", "slate"),
            category=item.get("stage", "其他"),
            resource=item.get("resource", "无"),
            route_path="/outline.view" if app_id == 1 else "/workflow.view",
            enabled=True,
            show_action=bool(item.get("showAction", False)),
            sort_order=app_id,
        )
        db.add(app)
    db.flush()


def serialize_app(app: AppModel) -> dict:
    """返回兼容当前前端字段的卡片数据。"""
    return {
        "id": app.id,
        "app_code": app.app_code,
        "appCode": app.app_code,
        "title": app.title,
        "description": app.description,
        "icon": app.icon,
        "icon_bg": app.icon_bg,
        "iconBg": app.icon_bg,
        "iconTone": app.icon_tone,
        "icon_tone": app.icon_tone,
        "tag": app.tag,
        "tag_color": app.tag_color,
        "tagColor": app.tag_color,
        "tagTone": app.tag_tone,
        "tag_tone": app.tag_tone,
        "category": app.category,
        "stage": app.category,
        "resource": app.resource,
        "route_path": app.route_path,
        "routePath": app.route_path,
        "enabled": app.enabled,
        "show_action": app.show_action,
        "showAction": app.show_action,
        "sort_order": app.sort_order,
        "sortOrder": app.sort_order,
        "created_at": app.created_at,
        "updated_at": app.updated_at,
    }


def list_apps(
    db: Session,
    *,
    category: Optional[str] = None,
    enabled_only: bool = True,
) -> list[dict]:
    seed_default_apps(db)
    query = db.query(AppModel)
    if enabled_only:
        query = query.filter(AppModel.enabled.is_(True))
    if category and category != "所有":
        query = query.filter(AppModel.category == category)
    apps = query.order_by(AppModel.sort_order.asc(), AppModel.id.asc()).all()
    return [serialize_app(app) for app in apps]


def list_app_categories(db: Session) -> list[str]:
    seed_default_apps(db)
    categories = [
        row[0]
        for row in db.query(AppModel.category)
        .filter(AppModel.enabled.is_(True))
        .distinct()
        .order_by(AppModel.category.asc())
        .all()
    ]
    order = ["教学前", "教学中", "教学后", "其他"]
    sorted_categories = [item for item in order if item in categories]
    sorted_categories.extend([item for item in categories if item not in sorted_categories])
    return ["所有", *sorted_categories]


def get_app(db: Session, app_id: int) -> Optional[AppModel]:
    seed_default_apps(db)
    return db.query(AppModel).filter(AppModel.id == app_id).one_or_none()


def create_app(db: Session, payload: AppCreate) -> AppModel:
    app = AppModel(**payload.model_dump())
    db.add(app)
    db.flush()
    return app


def update_app(db: Session, app: AppModel, payload: AppUpdate) -> AppModel:
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(app, key, value)
    db.flush()
    return app


def delete_app(db: Session, app: AppModel) -> None:
    db.delete(app)
    db.flush()
