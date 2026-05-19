"""教学大纲子智能体定义

定义教学大纲生成流水线中的各子智能体：
    - 章生成智能体
    - 节生成智能体（待实现）
    - 知识点生成智能体（待实现）
    - 质量检测智能体（待实现）

Author: LogicYe
Date: 2026-05-19
"""
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from .prompts.outline_prompts import (
    CHAPTER_SYSTEM_PROMPT,
)
from .structured.outline_models import (
    Outline
)
from .model_factory import create_chat_model


def create_chapter_agent():
    """创建章生成智能体

    使用模型工厂创建的 ChatModel 和 ToolStrategy 结构化输出策略构建智能体，
    根据课程基本信息生成完整的章节目录。

    Returns:
        配置完成的章节生成智能体实例
    """
    _agent = create_agent(
        model=create_chat_model(),
        system_prompt=CHAPTER_SYSTEM_PROMPT,
        response_format=ToolStrategy(Outline),
        tools=[],
        middleware=[],
        state_schema=None,
        checkpointer=None,
        context_schema=None,
        store=None,
    )
    return _agent

######################################
"""章生成智能体

模块级单例，由 create_chapter_agent() 初始化。
用于工作流中 chapter 节点调用，根据课程信息生成章节目录。
"""
chapter_agent = create_chapter_agent()
