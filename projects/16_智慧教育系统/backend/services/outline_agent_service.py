"""Outline Agent 运行封装。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.schemas.outline import OutlineGenerateRequest
from utils.logger import get_logger
from utils.settings import settings


logger = get_logger(__name__)


class OutlineAgentRuntime:
    """复用现有 LangGraph Outline Agent，并提供稳定的结构化降级输出。"""

    def generate(self, payload: OutlineGenerateRequest) -> tuple[dict[str, Any], str]:
        if settings.outline_enable_llm:
            try:
                return self._generate_with_langgraph(payload), "llm"
            except Exception as exc:
                logger.exception(f"Outline Agent 调用失败，使用本地结构化降级输出: {exc}")
        return self._fallback_outline(payload), "fallback"

    def _generate_with_langgraph(self, payload: OutlineGenerateRequest) -> dict[str, Any]:
        from backend.agents.outline_workflow import create_outline_agent
        from backend.agents.states.outline_state import CourseOutline

        agent = create_outline_agent()
        response = agent.invoke(
            CourseOutline(
                course_name=payload.course_title,
                course_description=payload.course_description or f"{payload.course_title}课程",
                difficulty_level=payload.difficulty,
                target_audience=payload.target_students or payload.stage,
            )
        )
        return self._normalize_agent_response(payload, response)

    def _normalize_agent_response(
        self,
        payload: OutlineGenerateRequest,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        raw_chapters = [self._to_dict(item) for item in response.get("chapters") or []]
        raw_sections = [self._to_dict(item) for item in response.get("sections") or []]
        raw_points = [self._to_dict(item) for item in response.get("knowledge_points") or []]

        sections_by_chapter: dict[str, list[dict[str, Any]]] = defaultdict(list)
        points_by_section: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for point in raw_points:
            key = (point.get("chapter_name", ""), point.get("section_name", ""))
            points_by_section[key].append(
                {
                    "name": point.get("name", ""),
                    "description": point.get("description", ""),
                    "key_points": point.get("key_points", []),
                    "difficulty": point.get("difficulty", payload.difficulty),
                }
            )

        for section in raw_sections:
            chapter_name = section.get("chapter_name", "")
            section_name = section.get("name", "")
            sections_by_chapter[chapter_name].append(
                {
                    "title": section_name,
                    "description": section.get("description", ""),
                    "hours": max(1, payload.total_hours // max(1, len(raw_sections))),
                    "teaching_goals": section.get("learning_objectives", []),
                    "knowledge_points": points_by_section.get((chapter_name, section_name), []),
                    "teaching_methods": payload.teaching_methods or ["案例讲解", "课堂练习"],
                    "assessment": "课堂表现与阶段练习",
                }
            )

        chapters = []
        for index, chapter in enumerate(raw_chapters, 1):
            chapter_title = chapter.get("name", f"第{index}章 {payload.course_title}")
            chapters.append(
                {
                    "title": chapter_title,
                    "description": chapter.get("description", ""),
                    "hours": max(1, payload.total_hours // max(1, len(raw_chapters))),
                    "teaching_goals": chapter.get("learning_objectives", []),
                    "sections": sections_by_chapter.get(chapter_title, []),
                }
            )

        if not chapters:
            return self._fallback_outline(payload)

        flat_sections = [
            {**section, "chapter_title": chapter["title"]}
            for chapter in chapters
            for section in chapter["sections"]
        ]
        return self._assemble_outline(payload, chapters, flat_sections)

    def _fallback_outline(self, payload: OutlineGenerateRequest) -> dict[str, Any]:
        key_points = payload.key_points or ["核心概念", "方法体系", "实践应用", "综合评价"]
        difficult_points = payload.difficult_points or ["概念迁移", "综合应用"]
        goals = payload.teaching_goals or [
            f"理解{payload.course_title}的核心概念与基本框架",
            "能够将关键知识点应用到典型教学或实践场景",
            "形成分析问题、设计方案和反思改进的能力",
        ]

        chapter_count = min(6, max(3, len(key_points)))
        hours_per_chapter = max(1, payload.total_hours // chapter_count)
        chapters = []
        flat_sections = []

        for index in range(chapter_count):
            point = key_points[index % len(key_points)]
            chapter_title = f"第{index + 1}章 {point}"
            sections = [
                {
                    "title": f"{index + 1}.1 {point}基础认知",
                    "description": f"梳理{point}的基本概念、常见误区与学习路径。",
                    "hours": max(1, hours_per_chapter // 2),
                    "teaching_goals": [f"准确说出{point}的关键概念", "能识别典型应用场景"],
                    "knowledge_points": [
                        {
                            "name": f"{point}概念框架",
                            "description": f"建立{point}的概念结构与术语体系。",
                            "key_points": [point, "概念边界", "基础例题"],
                            "difficulty": payload.difficulty,
                        }
                    ],
                    "teaching_methods": payload.teaching_methods or ["情境导入", "案例讲解"],
                    "assessment": "课堂提问与随堂练习",
                },
                {
                    "title": f"{index + 1}.2 {point}实践应用",
                    "description": f"通过任务驱动方式完成{point}的迁移应用。",
                    "hours": max(1, hours_per_chapter - max(1, hours_per_chapter // 2)),
                    "teaching_goals": [f"完成{point}相关任务", "能够解释解题或设计过程"],
                    "knowledge_points": [
                        {
                            "name": f"{point}应用策略",
                            "description": f"围绕{point}设计实践任务与评价标准。",
                            "key_points": ["任务分解", "过程反馈", "结果评价"],
                            "difficulty": "中等",
                        }
                    ],
                    "teaching_methods": payload.teaching_methods or ["任务驱动", "小组协作"],
                    "assessment": "作品产出与过程记录",
                },
            ]
            chapter = {
                "title": chapter_title,
                "description": f"围绕{point}展开理论讲述、案例分析与实践训练。",
                "hours": hours_per_chapter,
                "teaching_goals": goals[:2],
                "sections": sections,
            }
            chapters.append(chapter)
            flat_sections.extend([{**section, "chapter_title": chapter_title} for section in sections])

        outline = self._assemble_outline(payload, chapters, flat_sections)
        outline["key_points"] = key_points
        outline["difficult_points"] = difficult_points
        outline["teaching_goals"] = goals
        return outline

    def _assemble_outline(
        self,
        payload: OutlineGenerateRequest,
        chapters: list[dict[str, Any]],
        flat_sections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        teaching_goals = payload.teaching_goals or [
            f"掌握{payload.course_title}的基本概念、方法和应用路径",
            "能够完成典型任务分析并形成结构化表达",
            "具备课程相关问题的探究、协作与评价能力",
        ]
        key_points = payload.key_points or [
            item["title"].split(" ", 1)[-1] for item in chapters[:4]
        ]
        difficult_points = payload.difficult_points or [
            "知识点之间的联系建构",
            "真实问题场景中的迁移应用",
        ]
        teaching_methods = payload.teaching_methods or ["讲授法", "案例教学", "任务驱动", "小组研讨"]
        assessment_methods = payload.assessment_methods or ["过程性评价", "阶段测验", "项目作品", "课堂参与"]
        references = payload.references or ["课程标准与教材", "教师自编案例", "开放教育资源"]

        return {
            "course_title": payload.course_title,
            "course_description": payload.course_description or f"{payload.course_title}结构化教学大纲",
            "target_students": payload.target_students or payload.stage,
            "stage": payload.stage,
            "total_hours": payload.total_hours,
            "difficulty": payload.difficulty,
            "teaching_goals": teaching_goals,
            "key_points": key_points,
            "difficult_points": difficult_points,
            "chapters": chapters,
            "sections": flat_sections,
            "teaching_requirements": [
                "课前完成基础资料阅读与问题收集",
                "课堂围绕核心概念、典型案例和实践任务展开",
                "课后完成反思记录、拓展练习和阶段性成果整理",
            ],
            "teaching_methods": teaching_methods,
            "assessment_methods": assessment_methods,
            "references": references,
            "ppt_outline": self._build_ppt_outline(payload, chapters),
        }

    def _build_ppt_outline(
        self,
        payload: OutlineGenerateRequest,
        chapters: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        slides = [
            {"slide_type": "cover", "title": payload.course_title, "layout": "首页"},
            {"slide_type": "intro", "title": "课程介绍", "layout": "课程介绍"},
            {"slide_type": "agenda", "title": "课程目录", "layout": "目录"},
            {"slide_type": "requirements", "title": "教学要求", "layout": "教学要求"},
        ]
        for chapter in chapters:
            slides.extend(
                [
                    {"slide_type": "chapter", "title": chapter["title"], "layout": "章节页"},
                    {"slide_type": "theory", "title": f"{chapter['title']} 理论讲述", "layout": "理论讲述模板"},
                    {"slide_type": "explanation", "title": f"{chapter['title']} 讲解设计", "layout": "带讲解模板"},
                    {"slide_type": "mixed", "title": f"{chapter['title']} 图文混编", "layout": "图文混编模板"},
                ]
            )
        return slides

    @staticmethod
    def _to_dict(item: Any) -> dict[str, Any]:
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if isinstance(item, dict):
            return item
        return dict(item)
