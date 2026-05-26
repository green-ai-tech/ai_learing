"""智能体应用卡片 API Schema。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AppBase(BaseModel):
    app_code: str = Field(..., min_length=2, max_length=80)
    title: str
    description: str
    icon: str = "FileText"
    icon_bg: str = "bg-blue-600"
    icon_tone: str = "blue"
    tag: str = "其他"
    tag_color: str = "bg-slate-100 text-slate-600"
    tag_tone: str = "slate"
    category: str = "其他"
    resource: str = "无"
    route_path: Optional[str] = None
    enabled: bool = True
    show_action: bool = False
    sort_order: int = 100


class AppCreate(AppBase):
    pass


class AppUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    icon_bg: Optional[str] = None
    icon_tone: Optional[str] = None
    tag: Optional[str] = None
    tag_color: Optional[str] = None
    tag_tone: Optional[str] = None
    category: Optional[str] = None
    resource: Optional[str] = None
    route_path: Optional[str] = None
    enabled: Optional[bool] = None
    show_action: Optional[bool] = None
    sort_order: Optional[int] = None


class AppResponse(AppBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def stage(self) -> str:
        return self.category

    @property
    def iconBg(self) -> str:
        return self.icon_bg

    @property
    def iconTone(self) -> str:
        return self.icon_tone

    @property
    def tagColor(self) -> str:
        return self.tag_color

    @property
    def tagTone(self) -> str:
        return self.tag_tone

    @property
    def showAction(self) -> bool:
        return self.show_action

