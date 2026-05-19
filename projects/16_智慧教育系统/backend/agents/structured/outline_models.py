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
    name: str                           = Field(description="章名称")
    description: str                     = Field(description="章的描述")
    learning_objectives : List[str]     = Field(default_factory=list, description="学习目标列表")
    
class Outline(BaseModel):
    """大纲结构化输出模型

    LLM 返回的完整大纲格式，包含若干章的列表。
    """
    chapters : List[Chapter]            = Field(description="章列表")