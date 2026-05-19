"""教学大纲状态模型定义

定义 LangGraph 工作流中使用的状态数据结构，包括课程大纲和章节模型。

Author: LogicYe
Date: 2026-05-19
"""
from pydantic import BaseModel, Field
from typing  import List, Dict, Optional, Any

class Chapter(BaseModel):
    """章节模型

    表示课程大纲中的一章，包含名称、描述和学习目标。
    """
    name: str                           = Field(description="章名称")
    description: str                     = Field(description="章的描述")
    learning_objectives : List[str]     = Field(default_factory=list, description="学习目标列表")
    

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
    """
    # 输入
    course_name: str                    = Field(description="课程名称")  
    course_description: str             = Field(description="课程描述（50-100字）")
    difficulty_level: str                = Field(description="难度级别")
    target_audience: str                = Field(description="目标人群")
    # 输出(章)
    chapters : Optional[List[Chapter]]  = Field(description="章列表", default=None)
    # 输出(节)
    # 输出(知识点)
    # 其他(记录tokens数)