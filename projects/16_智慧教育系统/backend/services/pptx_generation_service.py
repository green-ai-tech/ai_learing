"""教学大纲 PPTX 生成服务。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pptx import Presentation

from backend.models.outline import OutlineModel
from backend.services.pptx_template_service import PptxTemplateService


STORAGE_ROOT = Path(__file__).resolve().parents[1] / "storage"
PPTX_DIR = STORAGE_ROOT / "pptx"


class PptxGenerationService:
    """基于模板结构生成课件 PPTX。"""

    def generate_pptx(self, outline: OutlineModel) -> Path:
        if not outline.outline_json:
            raise ValueError("大纲内容为空，无法生成 PPTX")

        PPTX_DIR.mkdir(parents=True, exist_ok=True)
        file_path = PPTX_DIR / f"outline_{outline.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pptx"

        data = outline.outline_json
        prs = Presentation()
        template = PptxTemplateService(prs)

        template.add_cover(data.get("course_title", "教学大纲"), data.get("target_students", ""))
        template.add_intro(
            "课程介绍",
            data.get("course_description", ""),
            [
                f"总课时：{data.get('total_hours', '')}",
                f"难度：{data.get('difficulty', '')}",
                f"目标学生：{data.get('target_students', '')}",
            ],
        )
        template.add_agenda("课程目录", data.get("chapters", []))
        template.add_requirements(
            "教学要求",
            data.get("teaching_requirements", []),
            data.get("teaching_methods", []),
            data.get("assessment_methods", []),
        )

        for chapter in data.get("chapters", []):
            template.add_chapter(chapter)
            template.add_theory(chapter)
            template.add_explanation(chapter)
            template.add_mixed(chapter)

        prs.save(file_path)
        return file_path
