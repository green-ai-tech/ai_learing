"""
教学大纲生成Agent（自动化流程）
"""

from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from .states.outline import (CourseOutline)
from .prompts.outline import (CHAPTER_GENERATION_PROMPT)
from .outline_agents import(chapter_agent)

def chapter(state: CourseOutline)-> dict | None:
    """
    """
    print("生成章")
    # 加载提示词模板
    chapter_template = ChatPromptTemplate.from_template(CHAPTER_GENERATION_PROMPT)
    # 格式化提示词
    prompts = chapter_template.invoke({
        "course_name": state.course_name,
        "course_description": state.course_description,
        "difficulty_level": state.difficulty_level,
        "target_audience": state.target_audience    
    })  # PromptValue
    
    _response = chapter_agent.invoke({
        "messages": [
            HumanMessage(content=prompts.to_string())
        ]
    })
    # print(_response)
    updates = {}
    results = _response["structured_response"]
    updates = {
        "chapters": []
    }
    for chapter in results.chapters:
        dict_chapter = {}
        dict_chapter["name"] = chapter.name
        dict_chapter["description"] = chapter.description
        dict_chapter["learning_objectives"] = chapter.learning_objectives
        updates["chapters"].append(dict_chapter)
    print(updates)
    return updates


def section(state: CourseOutline) -> dict | None:
    print("生成节")
    return 

def knowledge_point(state: CourseOutline) -> dict| None:
    print("生成知识点")
    return 


# 构建自动化智能体
def create_outline_agent():
    _graph = StateGraph(CourseOutline)  # 自动化流程图
    
    _graph.add_node("chapter", chapter)
    _graph.add_node("section", section)
    _graph.add_node("knowledge_point", knowledge_point)
    
    _graph.add_edge(START, "chapter")
    _graph.add_edge("chapter", "section")
    _graph.add_edge("section", "knowledge_point")
    _graph.add_edge("knowledge_point", END)
    
    _agent = _graph.compile()
    return _agent

# 测试代码
if __name__ == "__main__":
    agent = create_outline_agent()
    response = agent.invoke(CourseOutline(
        course_name="机器学习实战",
        course_description="本课程系统讲解机器学习核心算法，包括监督学习、无监督学习、特征工程、模型评估与优化。通过Scikit-learn和Pytorch实战项目，培养数据分析和模型构建能力。",
        difficulty_level="中等难度",
        target_audience="大学本科生",
    ))
    print(response)