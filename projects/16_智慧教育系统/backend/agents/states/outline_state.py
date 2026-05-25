"""教学大纲状态模型定义

定义 LangGraph 工作流中使用的状态数据结构，包括课程大纲、章、节和知识点模型。

Author: LogicYe
Date: 2026-05-19
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class Chapter(BaseModel):
    """章节模型

    表示课程大纲中的一章，包含名称、描述和学习目标。
    """
    name: str = Field(description="章名称")
    description: str = Field(description="章的描述")
    learning_objectives: List[str] = Field(default_factory=list, description="学习目标列表")


class Section(BaseModel):
    """节模型

    表示章下的一节，通过 chapter_name 关联所属章。
    """
    chapter_name: str = Field(description="所属章名称")
    name: str = Field(description="节名称")
    description: str = Field(description="节的描述")
    learning_objectives: List[str] = Field(default_factory=list, description="学习目标列表")


class KnowledgePoint(BaseModel):
    """知识点模型

    表示节下的一个知识点，通过 chapter_name / section_name 关联所属章和节。
    """
    chapter_name: str = Field(description="所属章名称")
    section_name: str = Field(description="所属节名称")
    name: str = Field(description="知识点名称")
    description: str = Field(description="知识点描述")
    key_points: List[str] = Field(default_factory=list, description="核心要点")
    difficulty: str = Field(default="中等", description="知识点难度: 简单/中等/困难")


class CourseOutline(BaseModel):
    """课程大纲状态

    LangGraph 工作流的全局状态，包含输入参数和工作流各节点的输出结果。

    输入字段:
        course_name: 课程名称
        course_description: 课程描述
        difficulty_level: 难度级别
        target_audience: 目标人群

    输出字段:
        chapters: 章列表（由 chapter 节点生成）
        sections: 节列表（由 section 节点生成）
        knowledge_points: 知识点列表（由 knowledge_point 节点生成）
    """
    course_name: str = Field(description="课程名称")
    course_description: str = Field(description="课程描述（50-100字）")
    difficulty_level: str = Field(description="难度级别")
    target_audience: str = Field(description="目标人群")

    chapters: Optional[List[Chapter]] = Field(default=None, description="章列表")
    sections: Optional[List[Section]] = Field(default=None, description="节列表")
    knowledge_points: Optional[List[KnowledgePoint]] = Field(default=None, description="知识点列表")
