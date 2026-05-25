"""教学智能体应用清单。"""

from __future__ import annotations

from typing import Optional


APP_STAGE_OPTIONS = ["所有", "教学前", "教学中", "教学后", "其他"]


RAW_APPS_DATA = [
    {
        "id": 1,
        "title": "课程大纲生成智能体",
        "description": "结构化的章节大纲、知识点逻辑图、推荐的教学素材（案例、视频、论文）。",
        "stage": "教学前",
        "tag": "教学前",
        "icon": "ListTree",
        "iconTone": "blue",
        "tagTone": "blue",
        "resource": "无",
        "showAction": False,
    },
    {
        "id": 2,
        "title": "个性化预习内容生成智能体",
        "description": "生成差异化的预习材料（微视频、阅读清单、自测题）。",
        "stage": "教学前",
        "tag": "教学前",
        "icon": "BookOpen",
        "iconTone": "teal",
        "tagTone": "blue",
        "resource": "教育部慕课网",
        "showAction": False,
    },
    {
        "id": 3,
        "title": "课件制作辅助智能体",
        "description": "输入教案文本，自动生成 PPT 初稿、动画演示建议、互动问题设计。",
        "stage": "教学前",
        "tag": "教学前",
        "icon": "Presentation",
        "iconTone": "purple",
        "tagTone": "blue",
        "resource": "无",
        "showAction": False,
    },
    {
        "id": 4,
        "title": "课堂互动助教智能体",
        "description": "自动发起投票、小测验、分组讨论任务。",
        "stage": "教学中",
        "tag": "教学中",
        "icon": "Vote",
        "iconTone": "orange",
        "tagTone": "green",
        "resource": "微信 / 钉钉互动",
        "showAction": False,
    },
    {
        "id": 5,
        "title": "知识点讲解智能体",
        "description": "针对某个难点，用不同的比喻、图示或案例多角度解释，适合录播课或混合式学习中的自助点播。",
        "stage": "教学中",
        "tag": "教学中",
        "icon": "Lightbulb",
        "iconTone": "yellow",
        "tagTone": "green",
        "resource": "虚拟教师",
        "showAction": False,
    },
    {
        "id": 6,
        "title": "课堂纪律与专注度监测智能体",
        "description": "通过摄像头，统计学生抬头率、参与度，生成课后报告，仅用于教学改进并注意隐私合规。",
        "stage": "教学中",
        "tag": "教学中",
        "icon": "Eye",
        "iconTone": "indigo",
        "tagTone": "green",
        "resource": "无",
        "showAction": False,
    },
    {
        "id": 7,
        "title": "实时答疑智能体",
        "description": "学生可通过文字 / 语音提问，优先匹配已有知识库回答，无法回答时转真人教师，并建议教师补充材料。",
        "stage": "教学中",
        "tag": "教学中",
        "icon": "CircleHelp",
        "iconTone": "cyan",
        "tagTone": "green",
        "resource": "RAG 知识库",
        "showAction": False,
    },
    {
        "id": 8,
        "title": "作业批改与评分智能体",
        "description": "支持客观题、主观题、代码题、数学公式、简答题的自动批改，并输出成绩统计、常见错误标签和学生得分详情。",
        "stage": "教学后",
        "tag": "教学后",
        "icon": "ClipboardCheck",
        "iconTone": "red",
        "tagTone": "violet",
        "resource": "根据错误推荐补习材料",
        "showAction": False,
    },
    {
        "id": 9,
        "title": "个性化错题本与推题智能体",
        "description": "记录错误，根据错误推荐练习题与学习资料。",
        "stage": "教学后",
        "tag": "教学后",
        "icon": "NotebookPen",
        "iconTone": "pink",
        "tagTone": "violet",
        "resource": "无",
        "showAction": False,
    },
    {
        "id": 10,
        "title": "实验报告评语生成智能体",
        "description": "根据学生的实验报告，给出评语与成绩，避免空洞的“很好 / 需努力”。",
        "stage": "教学后",
        "tag": "教学后",
        "icon": "FlaskConical",
        "iconTone": "emerald",
        "tagTone": "violet",
        "resource": "无",
        "showAction": False,
    },
    {
        "id": 11,
        "title": "专业学习辅导机器人",
        "description": "专攻某一学科（如数学、编程、英语语法），能逐步引导解题而非直接给答案。",
        "stage": "教学后",
        "tag": "教学后",
        "icon": "GraduationCap",
        "iconTone": "blue",
        "tagTone": "violet",
        "resource": "无",
        "showAction": False,
    },
    {
        "id": 12,
        "title": "学习规划智能体",
        "description": "根据大纲与学生要求，生成每日学习任务清单与目标。",
        "stage": "其他",
        "tag": "其他",
        "icon": "CalendarCheck",
        "iconTone": "teal",
        "tagTone": "slate",
        "resource": "无",
        "showAction": False,
    },
    {
        "id": 13,
        "title": "智能出卷智能体",
        "description": "根据知识点覆盖、难度分布、区分度要求，自动生成 A/B 卷，并附带标准答案和评分标准。",
        "stage": "其他",
        "tag": "其他",
        "icon": "FileQuestion",
        "iconTone": "orange",
        "tagTone": "slate",
        "resource": "无",
        "showAction": False,
    },
    {
        "id": 14,
        "title": "试题质量分析智能体",
        "description": "考后分析每道题的得分率、区分度、选项效度，标记异常题（如难度偏差或歧义选项）。",
        "stage": "其他",
        "tag": "其他",
        "icon": "BarChart3",
        "iconTone": "indigo",
        "tagTone": "slate",
        "resource": "无",
        "showAction": False,
    },
]


def list_apps(tag_filter: Optional[str] = None, stage_filter: Optional[str] = None):
    """按标签或教学阶段筛选应用。"""
    result = RAW_APPS_DATA

    if tag_filter:
        result = [app for app in result if app["tag"] == tag_filter]
    if stage_filter and stage_filter != "所有":
        result = [app for app in result if app["stage"] == stage_filter]

    return result
