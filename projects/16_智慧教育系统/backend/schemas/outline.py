"""教学大纲 API Schema。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def normalize_list(value: Any) -> list[str]:
    """将文本、多行文本或列表统一为字符串列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).replace("，", "\n").replace(",", "\n").replace("、", "\n")
    return [item.strip("- 0123456789.、\t ") for item in text.splitlines() if item.strip()]


class OutlineGenerateRequest(BaseModel):
    conversation_id: Optional[str] = None
    course_title: str = Field(..., min_length=2, description="课程名称")
    course_description: str = Field(default="", description="课程介绍")
    target_students: str = Field(default="", description="目标学生")
    total_hours: int = Field(default=32, ge=1, le=300, description="总课时")
    teaching_goals: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    difficult_points: list[str] = Field(default_factory=list)
    difficulty: str = Field(default="中等")
    stage: str = Field(default="通用")
    teaching_methods: list[str] = Field(default_factory=list)
    assessment_methods: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)

    @field_validator(
        "teaching_goals",
        "key_points",
        "difficult_points",
        "teaching_methods",
        "assessment_methods",
        "references",
        mode="before",
    )
    @classmethod
    def coerce_list(cls, value: Any) -> list[str]:
        return normalize_list(value)

    def ensure_conversation_id(self) -> str:
        return self.conversation_id or f"conv_{uuid4().hex}"


class FileInfo(BaseModel):
    file_type: str
    file_path: str
    file_name: str
    mime_type: str
    created_at: Optional[datetime] = None


class RuntimeLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: Optional[str] = None
    outline_id: Optional[int] = None
    runtime_name: str
    level: str
    event: str
    message: str
    payload: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None


class OutlineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: str
    user_input: dict[str, Any]
    outline_json: Optional[dict[str, Any]] = None
    xlsx_file_path: Optional[str] = None
    pptx_file_path: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class OutlineListResponse(BaseModel):
    items: list[OutlineResponse]
