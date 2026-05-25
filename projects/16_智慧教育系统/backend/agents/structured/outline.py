"""教学大纲结构化输出模型（旧版）

ChatAgent 模式下使用的结构化输出模型，保留作为参考。
新版实现请参见 outline_models.py。

Author: LogicYe
Date: 2026-05-19
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class Chapter(BaseModel):
    """章节模型（ChatAgent 模式）

    用于约束 ChatAgent 返回的章节 JSON 格式。
    """
    name: str = Field(description="章名称")
    description: str = Field(description="章的描述")
    learning_objectives: List[str] = Field(default_factory=list, description="学习目标列表")


class Section(BaseModel):
    """节模型（ChatAgent 模式）

    用于约束 ChatAgent 返回的节 JSON 格式。
    """
    name: str = Field(description="节名称")
    description: str = Field(description="节的描述")
    learning_objectives: List[str] = Field(default_factory=list, description="节学习目标")
    chapter_name: str = Field(description="所属章名称")


class KnowledgePoint(BaseModel):
    """知识点模型（ChatAgent 模式）

    用于约束 ChatAgent 返回的知识点 JSON 格式。
    """
    name: str = Field(description="知识点名称")
    description: str = Field(description="知识点描述")
    key_points: List[str] = Field(default_factory=list, description="核心要点")
    section_name: str = Field(description="所属节名称")
    difficulty: str = Field(description="知识点难度")


class Outline(BaseModel):
    """章大纲模型（ChatAgent 模式）

    表示 LLM 返回的完整章大纲。
    """
    chapters: List[Chapter] = Field(description="章列表")


class QualityCheckResult(BaseModel):
    """质量检查结果模型

    预留的质量评估数据模型，尚未在现有工作流中启用。
    """
    is_qualified: bool = Field(description="是否合格")
    score: int = Field(description="质量评分 0-100")
    issues: List[str] = Field(default_factory=list, description="问题列表")
    suggestions: List[str] = Field(default_factory=list, description="优化建议")