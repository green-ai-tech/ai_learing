"""教学大纲生成工作流（简化版）

基于 LangGraph + LangChain ChatAgent 的老版实现，保留作为参考。
新版实现请参见 outline_workflow.py。

Author: LogicYe
Date: 2026-05-19
"""

from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from .states.outline_state import CourseOutline
from .prompts.outline_prompts import CHAPTER_GENERATION_PROMPT
from .outline_agents import chapter_chain as chapter_agent


def chapter(state: CourseOutline) -> dict | None:
    """章生成节点

    使用 ChatAgent 调用 LLM 生成课程章目录，解析结构化响应后回填到状态中。
    """
    print("=== 生成章 ===")

    chapter_template = ChatPromptTemplate.from_template(CHAPTER_GENERATION_PROMPT)
    prompts = chapter_template.invoke({
        "course_name": state.course_name,
        "course_description": state.course_description,
        "difficulty_level": state.difficulty_level,
        "target_audience": state.target_audience,
    })

    _response = chapter_agent.invoke({
        "messages": [
            HumanMessage(content=prompts.to_string())
        ]
    })

    results = _response["structured_response"]
    updates = {"chapters": []}
    for ch in results.chapters:
        updates["chapters"].append({
            "name": ch.name,
            "description": ch.description,
            "learning_objectives": ch.learning_objectives,
        })
    return updates


def section(state: CourseOutline) -> dict | None:
    """节生成节点（未实现，预留扩展）"""
    print("=== 生成节 ===")
    return


def knowledge_point(state: CourseOutline) -> dict | None:
    """知识点生成节点（未实现，预留扩展）"""
    print("=== 生成知识点 ===")
    return


def create_outline_agent():
    """构建教学大纲生成工作流

    START → 章生成 → 节生成 → 知识点生成 → END
    """
    _graph = StateGraph(CourseOutline)

    _graph.add_node("chapter", chapter)
    _graph.add_node("section", section)
    _graph.add_node("knowledge_point", knowledge_point)

    _graph.add_edge(START, "chapter")
    _graph.add_edge("chapter", "section")
    _graph.add_edge("section", "knowledge_point")
    _graph.add_edge("knowledge_point", END)

    return _graph.compile()


if __name__ == "__main__":
    agent = create_outline_agent()
    response = agent.invoke(CourseOutline(
        course_name="机器学习实战",
        course_description="本课程系统讲解机器学习核心算法，包括监督学习、无监督学习、特征工程、模型评估与优化。通过Scikit-learn和Pytorch实战项目，培养数据分析和模型构建能力。",
        difficulty_level="中等难度",
        target_audience="大学本科生",
    ))
    print(response)