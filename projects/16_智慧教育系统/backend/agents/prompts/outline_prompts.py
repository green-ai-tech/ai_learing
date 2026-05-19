"""教学大纲生成提示词模板

定义课程大纲生成所需的 System Prompt 和 Generation Prompt。

Author: LogicYe
Date: 2026-05-19
"""
"""章节生成智能体的 System Prompt"""
CHAPTER_SYSTEM_PROMPT = "你是一位资深的课程设计专家，请详细仔细回答。"

"""章节生成 User Prompt 模板

占位符:
    {course_name}: 课程名称
    {course_description}: 课程描述
    {difficulty_level}: 难度级别
    {target_audience}: 目标受众
"""
CHAPTER_GENERATION_PROMPT = """你是一位资深的课程设计专家，拥有10年以上教学大纲设计经验。

## 课程信息
- 课程名称: {course_name}
- 课程描述: {course_description}
- 难度级别: {difficulty_level}
- 目标受众: {target_audience}

## 任务要求
请为上述课程设计完整的章的目录结构。

## 设计原则
1. 章节数量: 5-8章（根据课程深度合理分配）
2. 逻辑递进: 从基础概念→核心知识→高级应用→实战项目
3. 每章应有明确的学习目标（3-5个）
4. 章节名称需简洁明了，体现核心内容
"""