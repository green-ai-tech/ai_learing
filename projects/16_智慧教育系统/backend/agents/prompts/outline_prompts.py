"""教学大纲生成提示词模板

定义课程大纲生成所需的 System Prompt 和 Generation Prompt。

Author: LogicYe
Date: 2026-05-19
"""

CHAPTER_SYSTEM_PROMPT = "你是专业课程架构师，根据课程信息生成结构化的章大纲。严格按照指定JSON格式输出。"
SECTION_SYSTEM_PROMPT = "你是专业课程设计师，根据已生成的章内容为每章设计合理的节结构。严格按照指定JSON格式输出。"
KNOWLEDGE_POINT_SYSTEM_PROMPT = "你是专业课程讲师，根据已生成的节内容设计细粒度的知识点。严格按照指定JSON格式输出。"

CHAPTER_GENERATION_PROMPT = """请根据以下课程信息，生成合理的课程章节目录：

课程名称：{course_name}
课程描述：{course_description}
难度等级：{difficulty_level}
目标受众：{target_audience}

要求：
1. 生成5-8章
2. 每章有明确的学习目标（3-5个）
3. 章节逻辑递进：基础概念 → 核心知识 → 高级应用

请严格按照以下JSON格式输出，字段名必须完全一致：
{{
  "chapters": [
    {{
      "name": "第一章 机器学习概述",
      "description": "介绍机器学习的基本概念、发展历程和应用场景",
      "learning_objectives": ["理解机器学习的定义", "了解主要应用场景", "掌握基本术语"]
    }}
  ]
}}
"""

SECTION_GENERATION_PROMPT = """请根据以下课程信息和已生成的章节，为每章设计2-4个节：

课程名称：{course_name}
难度等级：{difficulty_level}

已生成的章节目录：
{chapters_text}

要求：
1. 每个 chapter_name 必须与上述章名称完全一致
2. 每章设计2-4个节，节与章之间逻辑递进
3. 每节有明确的学习目标（2-3个）

请严格按照以下JSON格式输出，字段名必须完全一致：
{{
  "sections": [
    {{
      "chapter_name": "第一章 机器学习概述",
      "name": "1.1 什么是机器学习",
      "description": "机器学习的定义、分类与基本术语",
      "learning_objectives": ["掌握机器学习定义", "区分监督学习与无监督学习"]
    }}
  ]
}}
"""

KNOWLEDGE_POINT_GENERATION_PROMPT = """请根据以下课程信息和已生成的节，为每节设计3-5个知识点：

课程名称：{course_name}

已生成的节目录：
{sections_text}

要求：
1. 每个 chapter_name / section_name 必须与上述节名称完全一致
2. 每节设计3-5个知识点，覆盖本节核心内容
3. 每个知识点标注难度（简单/中等/困难）
4. 每个知识点列出2-3个核心要点

请严格按照以下JSON格式输出，字段名必须完全一致：
{{
  "knowledge_points": [
    {{
      "chapter_name": "第一章 机器学习概述",
      "section_name": "1.1 什么是机器学习",
      "name": "机器学习的定义",
      "description": "阐述机器学习的基本定义及其与传统编程的区别",
      "key_points": ["从数据中学习而非规则", "模型、训练与泛化"],
      "difficulty": "简单"
    }}
  ]
}}
"""
