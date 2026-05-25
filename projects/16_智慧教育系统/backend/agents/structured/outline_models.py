"""教学大纲结构化输出模型

定义 LLM 结构化输出所需的 Pydantic 模型，用于约束智能体返回的 JSON 格式。

Author: LogicYe
Date: 2026-05-19
"""
from pydantic import BaseModel, Field
from typing import List


class Chapter(BaseModel):
    """章节结构化输出模型

    LLM 返回的章节数据格式，用于约束智能体的结构化输出。
    """
    name: str = Field(description="章名称")
    description: str = Field(description="章的描述")
    learning_objectives: List[str] = Field(default_factory=list, description="学习目标列表")


class Outline(BaseModel):
    """章大纲结构化输出模型

    LLM 返回的完整章大纲格式，包含若干章的列表。
    """
    chapters: List[Chapter] = Field(description="章列表")


class SectionItem(BaseModel):
    """节结构化输出模型

    LLM 返回的单节数据格式，通过 chapter_name 关联所属章。
    """
    chapter_name: str = Field(description="所属章名称")
    name: str = Field(description="节名称")
    description: str = Field(description="节的描述")
    learning_objectives: List[str] = Field(default_factory=list, description="学习目标列表")


class SectionOutline(BaseModel):
    """节大纲结构化输出模型

    LLM 返回的完整节大纲格式，包含所有章对应的所有节。
    """
    sections: List[SectionItem] = Field(description="节列表")


class KnowledgePointItem(BaseModel):
    """知识点结构化输出模型

    LLM 返回的单知识点格式，通过 chapter_name / section_name 关联所属章和节。
    """
    chapter_name: str = Field(description="所属章名称")
    section_name: str = Field(description="所属节名称")
    name: str = Field(description="知识点名称")
    description: str = Field(description="知识点描述")
    key_points: List[str] = Field(default_factory=list, description="核心要点")
    difficulty: str = Field(default="中等", description="知识点难度: 简单/中等/困难")


class KnowledgePointOutline(BaseModel):
    """知识点大纲结构化输出模型

    LLM 返回的完整知识点大纲格式，包含所有节对应的所有知识点。
    """
    knowledge_points: List[KnowledgePointItem] = Field(description="知识点列表")
