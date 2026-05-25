"""教学大纲子智能体定义

定义教学大纲生成流水线中的结构化输出链：
    - 章生成
    - 节生成
    - 知识点生成

Author: LogicYe
Date: 2026-05-19
"""
from .model_factory import create_chat_model
from .structured.outline_models import (
    Outline,
    SectionOutline,
    KnowledgePointOutline,
)


def create_chapter_chain():
    """创建章生成结构化输出链"""
    model = create_chat_model()
    return model.with_structured_output(Outline, method="json_mode")


def create_section_chain():
    """创建节生成结构化输出链"""
    model = create_chat_model()
    return model.with_structured_output(SectionOutline, method="json_mode")


def create_knowledge_point_chain():
    """创建知识点生成结构化输出链"""
    model = create_chat_model()
    return model.with_structured_output(KnowledgePointOutline, method="json_mode")


chapter_chain = create_chapter_chain()
section_chain = create_section_chain()
knowledge_point_chain = create_knowledge_point_chain()
