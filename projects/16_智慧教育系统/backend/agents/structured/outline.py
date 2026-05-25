from pydantic import BaseModel, Field
from typing import List, Optional


class Chapter(BaseModel):
    name: str = Field(description="章名称")
    description: str = Field(description="章的描述")
    learning_objectives: List[str] = Field(default_factory=list, description="学习目标列表")


class Section(BaseModel):
    name: str = Field(description="节名称")
    description: str = Field(description="节的描述")
    learning_objectives: List[str] = Field(default_factory=list, description="节学习目标")
    chapter_name: str = Field(description="所属章名称")


class KnowledgePoint(BaseModel):
    name: str = Field(description="知识点名称")
    description: str = Field(description="知识点描述")
    key_points: List[str] = Field(default_factory=list, description="核心要点")
    section_name: str = Field(description="所属节名称")
    difficulty: str = Field(description="知识点难度")


class Outline(BaseModel):
    chapters: List[Chapter] = Field(description="章列表")


class QualityCheckResult(BaseModel):
    is_qualified: bool = Field(description="是否合格")
    score: int = Field(description="质量评分 0-100")
    issues: List[str] = Field(default_factory=list, description="问题列表")
    suggestions: List[str] = Field(default_factory=list, description="优化建议")