"""教学大纲生成工作流

基于 LangGraph 构建的教学大纲自动生成流水线：
章生成 → 节生成 → 知识点生成

Author: LogicYe
Date: 2026-05-19
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

from .states.outline_state import CourseOutline
from .prompts.outline_prompts import (
    CHAPTER_GENERATION_PROMPT,
    SECTION_GENERATION_PROMPT,
    KNOWLEDGE_POINT_GENERATION_PROMPT,
)
from .outline_agents import (
    chapter_chain,
    section_chain,
    knowledge_point_chain,
)


def _format_chapters_text(chapters) -> str:
    lines = []
    for i, ch in enumerate(chapters, 1):
        lines.append(f"{i}. 【{ch.name}】{ch.description}")
        if ch.learning_objectives:
            objectives = "、".join(ch.learning_objectives)
            lines.append(f"   学习目标: {objectives}")
    return "\n".join(lines)


def _format_sections_text(sections) -> str:
    lines = []
    for i, sec in enumerate(sections, 1):
        lines.append(f"{i}. [{sec.chapter_name}] {sec.name} - {sec.description}")
        if sec.learning_objectives:
            objectives = "、".join(sec.learning_objectives)
            lines.append(f"   学习目标: {objectives}")
    return "\n".join(lines)


def chapter(state: CourseOutline) -> dict | None:
    """章生成节点"""
    print("=== 生成章 ===")

    chapter_template = ChatPromptTemplate.from_template(CHAPTER_GENERATION_PROMPT)
    prompt_text = chapter_template.invoke({
        "course_name": state.course_name,
        "course_description": state.course_description,
        "difficulty_level": state.difficulty_level,
        "target_audience": state.target_audience,
    }).to_string()

    result = chapter_chain.invoke(prompt_text)
    return {"chapters": [ch.model_dump() for ch in result.chapters]}


def section(state: CourseOutline) -> dict | None:
    """节生成节点"""
    print("=== 生成节 ===")

    if not state.chapters:
        print("警告: 没有章数据，跳过节生成")
        return {"sections": []}

    chapters_text = _format_chapters_text(state.chapters)

    section_template = ChatPromptTemplate.from_template(SECTION_GENERATION_PROMPT)
    prompt_text = section_template.invoke({
        "course_name": state.course_name,
        "difficulty_level": state.difficulty_level,
        "chapters_text": chapters_text,
    }).to_string()

    result = section_chain.invoke(prompt_text)
    return {"sections": [sec.model_dump() for sec in result.sections]}


def knowledge_point(state: CourseOutline) -> dict | None:
    """知识点生成节点"""
    print("=== 生成知识点 ===")

    if not state.sections:
        print("警告: 没有节数据，跳过知识点生成")
        return {"knowledge_points": []}

    sections_text = _format_sections_text(state.sections)

    kp_template = ChatPromptTemplate.from_template(KNOWLEDGE_POINT_GENERATION_PROMPT)
    prompt_text = kp_template.invoke({
        "course_name": state.course_name,
        "sections_text": sections_text,
    }).to_string()

    result = knowledge_point_chain.invoke(prompt_text)
    return {"knowledge_points": [kp.model_dump() for kp in result.knowledge_points]}


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
    print("\n=== 最终结果 ===")
    for key, value in response.items():
        if isinstance(value, list):
            print(f"\n{key} ({len(value)}项):")
            for item in value:
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                print(f"  {item}")
        else:
            print(f"{key}: {value}")
